#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuatuoGPT-o1 选择题全层扫描（0-31层）
- 知识库：教科书 + 参考范围（不含同源训练集）
- 测试集：CMExam test.jsonl 抽样 100 条，前 50 条开发集，后 50 条测试集
- 每层使用 Top‑10 加权和风险分数，计算 AUC
"""

import json
import os
import random
import re
import numpy as np
import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ================= 配置 =================
MODEL_PATH = "./models/HuatuoGPT-o1-8B"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
TEST_FILE = "./data/test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
DEVICE = "cuda"
RANDOM_SEED = 42
TEST_SIZE = 100
DEV_SIZE = 50

# 构建或加载知识库索引（教科书+参考范围，不含同源训练集）
def get_choice_index():
    index_path = "./faiss_choice_basic_index"  # 新索引路径
    if os.path.exists(index_path):
        print(f"加载已有索引 {index_path}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        return vector_store
    else:
        print("构建知识库索引（教科书+参考范围）...")
        docs = []
        for file_path in [TEXTBOOK_FILE, REFERENCE_FILE]:
            print(f"  加载 {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if "text" in data:
                        content = data["text"]
                    elif "description" in data:
                        content = data["description"]
                    else:
                        continue
                    docs.append(Document(page_content=content, metadata={}))
        print(f"文档数: {len(docs)}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(index_path)
        print("索引已保存")
        return vector_store

print("加载 HuatuoGPT-o1-8B 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

vector_store = get_choice_index()

def load_sae(layer):
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE 文件不存在: {sae_path}")
    sae_weights = safetensors.torch.load_file(sae_path)
    W_enc = sae_weights['encoder.weight'].to(DEVICE).to(torch.float16)
    b_enc = sae_weights['encoder.bias'].to(DEVICE).to(torch.float16)
    return W_enc, b_enc

def sae_encode(hidden, W_enc, b_enc):
    z = hidden @ W_enc.T + b_enc
    topk_vals, topk_idx = torch.topk(z, SAE_TOP_K, dim=-1)
    features = torch.zeros_like(z)
    features.scatter_(-1, topk_idx, topk_vals)
    features = torch.relu(features)
    return features

def get_pre_generation_feature(question, knowledge, layer, W_enc, b_enc):
    # 使用与原始 Llama 相同的 prompt 格式（英文，要求输出字母）
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
    features = sae_encode(hidden.unsqueeze(0), W_enc, b_enc).squeeze(0).cpu().numpy()
    return features

def generate_answer(question, knowledge):
    # 用于获取正确答案标签（虽然模型准确率可能低，但我们需要真实正确性）
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
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    match = re.search(r'\b([A-E])\b', answer)
    return match.group(1) if match else ""

# 加载测试集并抽样
print("加载测试集...")
all_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_samples.append(json.loads(line))
random.seed(RANDOM_SEED)
test_samples = random.sample(all_samples, TEST_SIZE)
print(f"抽样 {len(test_samples)} 条，前 {DEV_SIZE} 条用作开发集")

layers = list(range(32))
results = {}

for layer in layers:
    print(f"\n=== 测试层 {layer} ===")
    W_enc, b_enc = load_sae(layer)
    
    # 开发集特征与正确性标签
    dev_features = []
    dev_labels = []  # 1=错误, 0=正确
    for i, sample in enumerate(tqdm(test_samples[:DEV_SIZE], desc="开发集")):
        question = sample["input"]
        true_label = sample["output"]
        retrieved = vector_store.similarity_search(question, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        pred = generate_answer(question, knowledge)
        is_correct = (pred == true_label)
        feat = get_pre_generation_feature(question, knowledge, layer, W_enc, b_enc)
        dev_features.append(feat)
        dev_labels.append(0 if is_correct else 1)
    X_dev = np.array(dev_features)
    y_dev = np.array(dev_labels)
    
    if y_dev.sum() == 0 or (len(y_dev) - y_dev.sum()) == 0:
        print("开发集缺乏错误或正确样本，跳过")
        results[layer] = 0.5
        continue
    
    # 特征筛选：t-statistic
    correct_mask = (y_dev == 0)
    error_mask = (y_dev == 1)
    mean_c = X_dev[correct_mask].mean(axis=0)
    mean_e = X_dev[error_mask].mean(axis=0)
    std_c = X_dev[correct_mask].std(axis=0) + 1e-8
    std_e = X_dev[error_mask].std(axis=0) + 1e-8
    pooled_std = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled_std
    t_score[np.isnan(t_score)] = 0
    top_k = 10
    top_idx = np.argsort(t_score)[-top_k:][::-1]
    top_w = t_score[top_idx]
    
    # 测试集
    test_risks = []
    test_true_error = []
    for sample in tqdm(test_samples[DEV_SIZE:], desc="测试集"):
        question = sample["input"]
        true_label = sample["output"]
        retrieved = vector_store.similarity_search(question, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        pred = generate_answer(question, knowledge)
        is_correct = (pred == true_label)
        feat = get_pre_generation_feature(question, knowledge, layer, W_enc, b_enc)
        risk = np.sum(feat[top_idx] * top_w)
        test_risks.append(risk)
        test_true_error.append(0 if is_correct else 1)
    auc = roc_auc_score(test_true_error, test_risks)
    results[layer] = auc
    print(f"AUC = {auc:.4f}")

print("\n=== HuatuoGPT-o1 选择题 SAE 各层 AUC 汇总 ===")
for l, auc in sorted(results.items()):
    print(f"层 {l:2d}: {auc:.4f}")