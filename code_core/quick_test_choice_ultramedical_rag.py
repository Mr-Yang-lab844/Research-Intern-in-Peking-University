#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama-3.1-8B-UltraMedical 选择题全层扫描（RAG）
知识库：教科书 + 参考范围 + 同源训练集
检索 Top-K=3，特征数=10
开发集 50 条，测试集 50 条
"""

import json
import re
import random
import torch
import numpy as np
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
TRAIN_FILE = "./data/train.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
DEVICE = "cuda"
TEST_FILE = "./data/test.jsonl"
TOTAL_SAMPLES = 100
DEV_SIZE = 50
RANDOM_SEED = 42
MAX_NEW_TOKENS = 200
LAYERS = list(range(32))

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 构建 FAISS 索引（教科书+参考范围+同源训练集）
def get_index():
    index_path = "./faiss_choice_ultra"
    if os.path.exists(index_path):
        print("加载已有索引")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("构建索引...")
        docs = []
        for fp in [TEXTBOOK_FILE, REFERENCE_FILE, TRAIN_FILE]:
            with open(fp, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if "text" in data:
                        content = data["text"]
                    elif "test_name" in data:
                        content = data["description"]
                    elif "input" in data:
                        content = f"问题: {data['input']}\n答案: {data['output']}"
                    else:
                        continue
                    docs.append(Document(page_content=content))
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vs = FAISS.from_documents(docs, embedding_model)
        vs.save_local(index_path)
        return vs

vector_store = get_index()

def load_sae(layer):
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
    w = safetensors.torch.load_file(sae_path)
    return w['encoder.weight'].to(DEVICE).to(torch.float16), w['encoder.bias'].to(DEVICE).to(torch.float16)

def sae_encode(hidden, W, b):
    z = hidden @ W.T + b
    topk = torch.topk(z, SAE_TOP_K, dim=-1)
    f = torch.zeros_like(z)
    f.scatter_(-1, topk.indices, topk.values)
    return torch.relu(f)

def generate_answer(question, knowledge):
    prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{knowledge}

Question:
{question}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in ans:
        ans = ans.split("Answer:")[-1].strip()
    match = re.search(r'\b([A-E])\b', ans)
    return match.group(1) if match else ""

def get_feature(question, knowledge, layer, W, b):
    prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{knowledge}

Question:
{question}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0, -1, :]
    feat = sae_encode(hidden.unsqueeze(0), W, b).squeeze(0).cpu().numpy()
    return feat

# 加载测试集并抽样
with open(TEST_FILE, 'r') as f:
    all_samples = [json.loads(line) for line in f]
random.seed(RANDOM_SEED)
samples = random.sample(all_samples, TOTAL_SAMPLES)
dev_samples = samples[:DEV_SIZE]
test_samples = samples[DEV_SIZE:]

results = {}
for layer in LAYERS:
    print(f"\n=== 层 {layer} ===")
    W, b = load_sae(layer)

    # 开发集特征和标签
    dev_feats, dev_labels = [], []
    for s in tqdm(dev_samples, desc="开发集"):
        q = s["input"]
        true = s["output"]
        retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        pred = generate_answer(q, knowledge)
        is_correct = (pred == true)
        feat = get_feature(q, knowledge, layer, W, b)
        dev_feats.append(feat)
        dev_labels.append(0 if is_correct else 1)
    X_dev = np.array(dev_feats)
    y_dev = np.array(dev_labels)

    if y_dev.sum() == 0 or (len(y_dev)-y_dev.sum()) == 0:
        results[layer] = 0.5
        continue

    # 特征选择
    correct_mask = (y_dev == 0)
    error_mask = (y_dev == 1)
    mean_c = X_dev[correct_mask].mean(axis=0)
    mean_e = X_dev[error_mask].mean(axis=0)
    std_c = X_dev[correct_mask].std(axis=0) + 1e-8
    std_e = X_dev[error_mask].std(axis=0) + 1e-8
    t_score = np.abs(mean_e - mean_c) / np.sqrt(std_c**2 + std_e**2)
    t_score[np.isnan(t_score)] = 0
    top_k = 10
    top_idx = np.argsort(t_score)[-top_k:][::-1]
    top_w = t_score[top_idx]

    # 测试集
    test_feats, test_errors = [], []
    for s in tqdm(test_samples, desc="测试集"):
        q = s["input"]
        true = s["output"]
        retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        pred = generate_answer(q, knowledge)
        is_correct = (pred == true)
        feat = get_feature(q, knowledge, layer, W, b)
        test_feats.append(feat)
        test_errors.append(0 if is_correct else 1)
    X_test = np.array(test_feats)
    y_test = np.array(test_errors)
    risks = np.sum(X_test[:, top_idx] * top_w, axis=1)
    auc = roc_auc_score(y_test, risks)
    results[layer] = auc
    print(f"AUC = {auc:.4f}")

print("\n=== 选择题 AUC 汇总 ===")
for l, auc in sorted(results.items()):
    print(f"层 {l:2d}: {auc:.4f}")