#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试不同 SAE 层在选择题任务上的表现
- 知识库：教科书 + 参考范围（已有 FAISS 索引）
- 测试集：CMExam test.jsonl（随机抽样 100 条）
- 前 50 条作为开发集（特征筛选），后 50 条作为测试集（评估 AUC）
- 每层使用 Top‑10 加权和风险分数
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
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
TEST_FILE = "./data/test.jsonl"          # CMExam 测试集
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
MAX_NEW_TOKENS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
TEST_SIZE = 200          # 总样本数
DEV_SIZE = 100            # 开发集大小

# ================= 加载模型和知识库 =================
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

print("加载知识库索引...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
vector_store = FAISS.load_local("./faiss_merged_index", embedding_model, allow_dangerous_deserialization=True)

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
    """选择题 prompt：知识 + 问题，提取生成答案前的最后一个 token"""
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
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    match = re.search(r'\b([A-E])\b', answer)
    return match.group(1) if match else ""

def extract_answer_letter(text):
    match = re.search(r'\b([A-E])\b', text)
    return match.group(1) if match else ""

# ================= 加载测试集并抽样 =================
print("加载测试集...")
all_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_samples.append(json.loads(line))
random.seed(RANDOM_SEED)
test_samples = random.sample(all_samples, TEST_SIZE)
print(f"抽样 {len(test_samples)} 条，其中前 {DEV_SIZE} 条用作开发集")

# 准备存储各层结果
layers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 17, 18, 19, 20,21,22,23,24,25,26,27,28,29,30,31]
results = {}

# ================= 逐层测试 =================
for layer in layers:
    print(f"\n=== 测试层 {layer} ===")
    W_enc, b_enc = load_sae(layer)
    
    # 开发集：提取特征并生成正确性标签
    dev_features = []
    dev_labels = []   # 1 表示错误，0 表示正确
    for i, sample in enumerate(tqdm(test_samples[:DEV_SIZE], desc="开发集")):
        question = sample["input"]
        true_label = sample["output"]
        # 检索知识
        retrieved = vector_store.similarity_search(question, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        # 生成答案（用于判断正确性）
        pred = generate_answer(question, knowledge)
        is_correct = (pred == true_label)
        # 提取特征
        feat = get_pre_generation_feature(question, knowledge, layer, W_enc, b_enc)
        dev_features.append(feat)
        dev_labels.append(0 if is_correct else 1)   # 1=错误
    
    X_dev = np.array(dev_features)
    y_dev = np.array(dev_labels)
    
    # 特征筛选：t-statistic
    correct_mask = (y_dev == 0)
    error_mask = (y_dev == 1)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        print("开发集中无错误样本，无法计算区分度")
        results[layer] = 0.5
        continue
    
    mean_correct = X_dev[correct_mask].mean(axis=0)
    mean_error = X_dev[error_mask].mean(axis=0)
    std_correct = X_dev[correct_mask].std(axis=0) + 1e-8
    std_error = X_dev[error_mask].std(axis=0) + 1e-8
    pooled_std = np.sqrt(std_correct**2 + std_error**2)
    t_score = np.abs(mean_error - mean_correct) / pooled_std
    t_score[np.isnan(t_score)] = 0
    top_k = 10
    top_indices = np.argsort(t_score)[-top_k:][::-1]
    top_weights = t_score[top_indices]
    print(f"Top 特征得分: {top_weights[:5]}")
    
    # 测试集：计算风险分数和真实错误标签
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
        risk = np.sum(feat[top_indices] * top_weights)
        test_risks.append(risk)
        test_true_error.append(0 if is_correct else 1)
    
    auc = roc_auc_score(test_true_error, test_risks)
    results[layer] = auc
    print(f"AUC = {auc:.4f}")

# ================= 输出汇总 =================
print("\n=== 选择题 SAE 各层 AUC 汇总 ===")
for layer, auc in sorted(results.items()):
    print(f"层 {layer:2d}: {auc:.4f}")