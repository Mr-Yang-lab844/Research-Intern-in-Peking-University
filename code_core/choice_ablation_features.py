#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选择题层20或22消融实验：改变特征数量（Top‑K 特征个数）
固定知识库：教科书+参考范围
固定检索 Top‑K=3
固定随机种子 42
固定开发集 50 条，测试集 50 条
输出各特征数量下的 AUC 以及 L1 范数基线
"""

import json
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

# ================= 固定配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
TEST_FILE = "./data/test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
MAX_NEW_TOKENS = 10
DEVICE = "cuda"
LAYER = 22 
RANDOM_SEED = 42
TOTAL_SAMPLES = 100   # 总样本数
DEV_SIZE = 50         # 开发集大小

# 待测试的特征数量列表
FEATURE_COUNTS = [1,2,3,4,5,6,7,8,9,10,15,20,25,30]

print("加载模型和知识库...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 加载 FAISS 索引（教科书+参考范围）
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
vector_store = FAISS.load_local("./faiss_merged_index", embedding_model, allow_dangerous_deserialization=True)

# 加载 SAE
sae_path = f"./models/Llama-Scope/L{LAYER}R-8x.safetensors"
sae_weights = safetensors.torch.load_file(sae_path)
W_enc = sae_weights['encoder.weight'].to(DEVICE).to(torch.float16)
b_enc = sae_weights['encoder.bias'].to(DEVICE).to(torch.float16)

def sae_encode(hidden):
    z = hidden @ W_enc.T + b_enc
    topk = torch.topk(z, SAE_TOP_K, dim=-1)
    f = torch.zeros_like(z)
    f.scatter_(-1, topk.indices, topk.values)
    return torch.relu(f)

def get_feature(question, knowledge):
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
    hidden = outputs.hidden_states[LAYER][0, -1, :]
    features = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
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

# 加载测试集并抽样
all_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_samples.append(json.loads(line))
random.seed(RANDOM_SEED)
samples = random.sample(all_samples, TOTAL_SAMPLES)
dev_samples = samples[:DEV_SIZE]
test_samples = samples[DEV_SIZE:]

# 预先为开发集和测试集提取特征和正确性标签（因为检索和生成只依赖知识库和模型，不依赖特征数量）
# 这样可以避免重复运行生成和检索，大幅加速
print("预提取开发集特征和标签...")
dev_features = []
dev_is_correct = []
for s in tqdm(dev_samples):
    q = s["input"]
    true = s["output"]
    retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    pred = generate_answer(q, knowledge)
    is_correct = (pred == true)
    feat = get_feature(q, knowledge)
    dev_features.append(feat)
    dev_is_correct.append(is_correct)

print("预提取测试集特征和标签...")
test_features = []
test_is_correct = []
for s in tqdm(test_samples):
    q = s["input"]
    true = s["output"]
    retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    pred = generate_answer(q, knowledge)
    is_correct = (pred == true)
    feat = get_feature(q, knowledge)
    test_features.append(feat)
    test_is_correct.append(is_correct)

dev_features = np.array(dev_features)
dev_labels = np.array([0 if c else 1 for c in dev_is_correct])  # 1=错误
test_features = np.array(test_features)
test_true_error = np.array([0 if c else 1 for c in test_is_correct])

# 计算全特征区分度（t-statistic）
correct_mask = (dev_labels == 0)
error_mask = (dev_labels == 1)
mean_correct = dev_features[correct_mask].mean(axis=0)
mean_error = dev_features[error_mask].mean(axis=0)
std_correct = dev_features[correct_mask].std(axis=0) + 1e-8
std_error = dev_features[error_mask].std(axis=0) + 1e-8
pooled_std = np.sqrt(std_correct**2 + std_error**2)
t_score = np.abs(mean_error - mean_correct) / pooled_std
t_score[np.isnan(t_score)] = 0

# 按t_score排序得到所有特征的排名
sorted_idx = np.argsort(t_score)[::-1]   # 降序

# 对不同特征数量计算AUC
results = {}
for k in FEATURE_COUNTS:
    top_idx = sorted_idx[:k]
    top_weights = t_score[top_idx]
    risks = np.sum(test_features[:, top_idx] * top_weights, axis=1)
    auc = roc_auc_score(test_true_error, risks)
    results[k] = auc
    print(f"特征数量 = {k:2d} -> AUC = {auc:.4f}")

# 计算L1范数基线
l1_risks = np.sum(test_features, axis=1)
auc_l1 = roc_auc_score(test_true_error, l1_risks)
print(f"L1 范数基线 AUC = {auc_l1:.4f}")

print("\n=== 消融实验汇总 ===")
for k, auc in results.items():
    print(f"Top-{k} 特征: {auc:.4f}")
print(f"L1 范数    : {auc_l1:.4f}")