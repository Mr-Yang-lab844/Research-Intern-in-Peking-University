#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuatuoGPT-o1 选择题层22消融实验：固定特征数=4，改变检索数量(1-10)和随机种子(42,123,2024)
样本总量200条（开发100，测试100）
模型：HuatuoGPT-o1-8B，知识库：教科书+参考范围（不含同源训练集）
无缓存，每次运行重新计算。
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ================= 配置 =================
MODEL_PATH = "./models/HuatuoGPT-o1-8B"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
TEST_FILE = "./data/test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
SAE_TOP_K = 32
MAX_NEW_TOKENS = 10
DEVICE = "cuda"
LAYER = 22
TOTAL_SAMPLES = 200
DEV_SIZE = 100
TEST_SIZE = 100
FEATURE_COUNT = 4
TOP_K_LIST = list(range(1, 11))
SEED_LIST = [42, 123, 2024]

print("加载模型和知识库...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
vector_store = FAISS.load_local("./faiss_merged_index", embedding_model, allow_dangerous_deserialization=True)

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

# 加载所有样本
with open(TEST_FILE, "r", encoding="utf-8") as f:
    all_samples = [json.loads(line) for line in f]
print(f"总样本数: {len(all_samples)}")

results = {}
for top_k in TOP_K_LIST:
    for seed in SEED_LIST:
        print(f"\n--- TOP_K={top_k}, seed={seed} ---")
        
        random.seed(seed)
        samples = random.sample(all_samples, TOTAL_SAMPLES)
        dev_samples = samples[:DEV_SIZE]
        test_samples = samples[DEV_SIZE:]

        # 开发集特征和标签
        dev_features, dev_labels = [], []
        for s in tqdm(dev_samples, desc="开发集"):
            q = s["input"]
            true = s["output"]
            retrieved = vector_store.similarity_search(q, k=top_k)
            knowledge = "\n\n".join([doc.page_content for doc in retrieved])
            pred = generate_answer(q, knowledge)
            is_correct = (pred == true)
            feat = get_feature(q, knowledge)
            dev_features.append(feat)
            dev_labels.append(0 if is_correct else 1)
        dev_features = np.array(dev_features)
        dev_labels = np.array(dev_labels)

        correct_mask = (dev_labels == 0)
        error_mask = (dev_labels == 1)
        if correct_mask.sum() == 0 or error_mask.sum() == 0:
            print(f"警告: 开发集中正负样本不足, 跳过 (正确:{correct_mask.sum()}, 错误:{error_mask.sum()})")
            results[(top_k, seed)] = 0.5
            continue

        mean_c = dev_features[correct_mask].mean(axis=0)
        mean_e = dev_features[error_mask].mean(axis=0)
        std_c = dev_features[correct_mask].std(axis=0) + 1e-8
        std_e = dev_features[error_mask].std(axis=0) + 1e-8
        pooled_std = np.sqrt(std_c**2 + std_e**2)
        t_score = np.abs(mean_e - mean_c) / pooled_std
        t_score[np.isnan(t_score)] = 0
        top_indices = np.argsort(t_score)[-FEATURE_COUNT:][::-1]
        top_weights = t_score[top_indices]

        # 测试集评估
        test_features, test_labels = [], []
        for s in tqdm(test_samples, desc="测试集", leave=False):
            q = s["input"]
            true = s["output"]
            retrieved = vector_store.similarity_search(q, k=top_k)
            knowledge = "\n\n".join([doc.page_content for doc in retrieved])
            pred = generate_answer(q, knowledge)
            is_correct = (pred == true)
            feat = get_feature(q, knowledge)
            test_features.append(feat)
            test_labels.append(0 if is_correct else 1)
        test_features = np.array(test_features)
        test_true_error = np.array(test_labels)
        risks = np.sum(test_features[:, top_indices] * top_weights, axis=1)
        auc = roc_auc_score(test_true_error, risks)

        results[(top_k, seed)] = auc
        print(f"TOP_K={top_k}, seed={seed} -> AUC = {auc:.4f}")

print("\n=== 消融实验汇总 (层22, 特征数=4, 样本量=200) ===")
print("TOP_K\\Seed", end="")
for seed in SEED_LIST:
    print(f"\t{seed}", end="")
print()
for top_k in TOP_K_LIST:
    print(f"{top_k}", end="")
    for seed in SEED_LIST:
        auc = results.get((top_k, seed), 0.5)
        print(f"\t{auc:.4f}", end="")
    print()