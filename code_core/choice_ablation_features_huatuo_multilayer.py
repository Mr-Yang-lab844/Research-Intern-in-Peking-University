#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuatuoGPT-o1 选择题特征数量消融（支持多层）
每层使用 Top‑K 特征数量从 1 到 30，输出 AUC 及 L1 基线
知识库：教科书+参考范围（不含同源训练集）
检索 Top‑K=3，固定划分种子 42
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
import os

# ================= 配置 =================
MODEL_PATH = "./models/HuatuoGPT-o1-8B"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"  # 使用原始 Llama tokenizer 避免损坏问题
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
TEST_FILE = "./data/test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
DEVICE = "cuda"
RANDOM_SEED = 42
TOTAL_SAMPLES = 200
DEV_SIZE = 100

# 待测试的层列表
LAYERS = [18, 19, 20, 22, 24,28]   # 可选，你可以根据需要增减

# 特征数量范围
FEATURE_COUNTS = list(range(1, 31))

# 缓存目录
CACHE_DIR = "./cache/choice_huatuo"
os.makedirs(CACHE_DIR, exist_ok=True)

# 加载 tokenizer 和模型
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 加载知识库索引
print("加载知识库索引...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
vector_store = FAISS.load_local("./faiss_merged_index", embedding_model, allow_dangerous_deserialization=True)

# 加载测试集并固定划分
all_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_samples.append(json.loads(line))
random.seed(RANDOM_SEED)
samples = random.sample(all_samples, TOTAL_SAMPLES)
dev_samples = samples[:DEV_SIZE]
test_samples = samples[DEV_SIZE:]
print(f"开发集 {len(dev_samples)}，测试集 {len(test_samples)}")

# 辅助函数：生成答案（使用模型生成，不借助 vLLM，因为需要 hidden states）
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
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    match = re.search(r'\b([A-E])\b', answer)
    return match.group(1) if match else ""

# 预计算所有样本的生成答案和正确性标签（与层无关，只需一次）
cache_answer_file = os.path.join(CACHE_DIR, "answers.npz")
if os.path.exists(cache_answer_file):
    print("加载缓存的答案标签...")
    data = np.load(cache_answer_file)
    dev_is_correct = data['dev_is_correct']
    test_is_correct = data['test_is_correct']
else:
    print("预生成开发集答案...")
    dev_is_correct = []
    for s in tqdm(dev_samples):
        q = s["input"]
        true = s["output"]
        retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        pred = generate_answer(q, knowledge)
        dev_is_correct.append(pred == true)
    print("预生成测试集答案...")
    test_is_correct = []
    for s in tqdm(test_samples):
        q = s["input"]
        true = s["output"]
        retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        pred = generate_answer(q, knowledge)
        test_is_correct.append(pred == true)
    np.savez(cache_answer_file, dev_is_correct=dev_is_correct, test_is_correct=test_is_correct)

dev_labels = np.array([0 if c else 1 for c in dev_is_correct])   # 1=错误
test_true_error = np.array([0 if c else 1 for c in test_is_correct])

# 对每一层进行特征提取和消融
for LAYER in LAYERS:
    print(f"\n========== 处理层 {LAYER} ==========")
    # 加载 SAE
    sae_path = f"./models/Llama-Scope/L{LAYER}R-8x.safetensors"
    if not os.path.exists(sae_path):
        print(f"SAE 文件不存在: {sae_path}，跳过")
        continue
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

    # 缓存特征
    cache_feat_file = os.path.join(CACHE_DIR, f"layer{LAYER}_features.npz")
    if os.path.exists(cache_feat_file):
        print("加载缓存的特征...")
        data = np.load(cache_feat_file)
        dev_features = data['dev_features']
        test_features = data['test_features']
    else:
        print("提取开发集特征...")
        dev_features = []
        for s in tqdm(dev_samples):
            q = s["input"]
            retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
            knowledge = "\n\n".join([doc.page_content for doc in retrieved])
            feat = get_feature(q, knowledge)
            dev_features.append(feat)
        dev_features = np.array(dev_features)
        print("提取测试集特征...")
        test_features = []
        for s in tqdm(test_samples):
            q = s["input"]
            retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
            knowledge = "\n\n".join([doc.page_content for doc in retrieved])
            feat = get_feature(q, knowledge)
            test_features.append(feat)
        test_features = np.array(test_features)
        np.savez(cache_feat_file, dev_features=dev_features, test_features=test_features)

    # 计算区分度 t_score
    correct_mask = (dev_labels == 0)
    error_mask = (dev_labels == 1)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        print("开发集中缺少正确或错误样本，跳过")
        continue
    mean_c = dev_features[correct_mask].mean(axis=0)
    mean_e = dev_features[error_mask].mean(axis=0)
    std_c = dev_features[correct_mask].std(axis=0) + 1e-8
    std_e = dev_features[error_mask].std(axis=0) + 1e-8
    pooled_std = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled_std
    t_score[np.isnan(t_score)] = 0
    sorted_idx = np.argsort(t_score)[::-1]

    # 特征数量消融
    print("\n特征数量消融结果:")
    best_auc = 0
    best_k = 0
    for k in FEATURE_COUNTS:
        top_idx = sorted_idx[:k]
        top_weights = t_score[top_idx]
        risks = np.sum(test_features[:, top_idx] * top_weights, axis=1)
        auc = roc_auc_score(test_true_error, risks)
        if auc > best_auc:
            best_auc = auc
            best_k = k
        print(f"Top-{k:2d} 特征: AUC = {auc:.4f}")
    print(f"最佳: Top-{best_k} 特征, AUC = {best_auc:.4f}")

    # L1 基线
    l1_risks = np.sum(test_features, axis=1)
    auc_l1 = roc_auc_score(test_true_error, l1_risks)
    print(f"L1 范数   : AUC = {auc_l1:.4f}")