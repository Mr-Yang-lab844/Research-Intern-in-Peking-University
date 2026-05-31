#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama-3.1-8B-UltraMedical 选择题 RAG 特征数量消融
知识库：教科书 + 参考范围 + 同源训练集
检索 Top-K=3
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

# ================= 配置 =================
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
TOTAL_SAMPLES = 200
DEV_SIZE = 100
RANDOM_SEED = 42
MAX_NEW_TOKENS = 200
LAYERS = [18, 20,22,24,27,29]   # 需要消融的层

def load_index():
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

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()
vector_store = load_index()

# 加载测试集并抽样
with open(TEST_FILE, 'r') as f:
    all_samples = [json.loads(line) for line in f]
random.seed(RANDOM_SEED)
samples = random.sample(all_samples, TOTAL_SAMPLES)
dev_samples = samples[:DEV_SIZE]
test_samples = samples[DEV_SIZE:]

# 预先为所有样本生成答案和正确性标签（因为不依赖层，可缓存）
cache_labels = "./cache/choice_labels.npy"
if os.path.exists(cache_labels):
    print("加载缓存的正确性标签")
    dev_labels = np.load(cache_labels + "_dev.npy")
    test_labels = np.load(cache_labels + "_test.npy")
else:
    print("预生成开发集标签...")
    dev_labels = []  # 1=错误, 0=正确
    for s in tqdm(dev_samples):
        q = s["input"]
        true = s["output"]
        retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        pred = generate_answer(q, knowledge)
        is_correct = (pred == true)
        dev_labels.append(0 if is_correct else 1)
    dev_labels = np.array(dev_labels)
    print("预生成测试集标签...")
    test_labels = []
    for s in tqdm(test_samples):
        q = s["input"]
        true = s["output"]
        retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        pred = generate_answer(q, knowledge)
        is_correct = (pred == true)
        test_labels.append(0 if is_correct else 1)
    test_labels = np.array(test_labels)
    os.makedirs("./cache", exist_ok=True)
    np.save(cache_labels + "_dev.npy", dev_labels)
    np.save(cache_labels + "_test.npy", test_labels)

# 对每个层进行消融
for layer in LAYERS:
    print(f"\n========== 层 {layer} 特征数量消融 ==========")
    cache_feat = f"./cache/choice_layer{layer}_feats.npy"
    if os.path.exists(cache_feat):
        print("加载缓存特征")
        all_feats = np.load(cache_feat)
        dev_feats = all_feats[:DEV_SIZE]
        test_feats = all_feats[DEV_SIZE:]
    else:
        W, b = load_sae(layer)
        print("提取开发集特征...")
        dev_feats = []
        for s in tqdm(dev_samples):
            q = s["input"]
            retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
            knowledge = "\n\n".join([d.page_content for d in retrieved])
            feat = get_feature(q, knowledge, layer, W, b)
            dev_feats.append(feat)
        dev_feats = np.array(dev_feats)
        print("提取测试集特征...")
        test_feats = []
        for s in tqdm(test_samples):
            q = s["input"]
            retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
            knowledge = "\n\n".join([d.page_content for d in retrieved])
            feat = get_feature(q, knowledge, layer, W, b)
            test_feats.append(feat)
        test_feats = np.array(test_feats)
        all_feats = np.vstack([dev_feats, test_feats])
        np.save(cache_feat, all_feats)

    # 计算区分度
    correct_mask = (dev_labels == 0)
    error_mask = (dev_labels == 1)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        print("开发集正负样本不足")
        continue
    mean_c = dev_feats[correct_mask].mean(axis=0)
    mean_e = dev_feats[error_mask].mean(axis=0)
    std_c = dev_feats[correct_mask].std(axis=0) + 1e-8
    std_e = dev_feats[error_mask].std(axis=0) + 1e-8
    pooled = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled
    t_score[np.isnan(t_score)] = 0
    sorted_idx = np.argsort(t_score)[::-1]

    print(f"\n特征数量消融 (层{layer}):")
    best_auc = 0
    best_k = 0
    for k in range(1, 31):
        top_idx = sorted_idx[:k]
        top_w = t_score[top_idx]
        risks = np.sum(test_feats[:, top_idx] * top_w, axis=1)
        auc = roc_auc_score(test_labels, risks)
        if auc > best_auc:
            best_auc = auc
            best_k = k
        print(f"Top-{k:2d} 特征: AUC = {auc:.4f}")
    print(f"最佳: Top-{best_k} 特征, AUC = {best_auc:.4f}")

    l1_risks = np.sum(test_feats, axis=1)
    auc_l1 = roc_auc_score(test_labels, l1_risks)
    print(f"L1 范数: AUC = {auc_l1:.4f}")