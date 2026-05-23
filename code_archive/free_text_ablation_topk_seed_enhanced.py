#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自由文本层13或者层16消融（增强知识库）：固定特征数=21，遍历检索数量(1-7)和随机种子(42,123,2024)
输出AUC表格
"""

import json
import random
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

# 固定参数
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
ENHANCED_RANGE_FILE = "./knowledge/reference_ranges_enhanced.jsonl"
ALL_TEST_FILE = "./data/free_text_test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
SAE_TOP_K = 32
DEVICE = "cuda"
LAYER = 16
TOTAL_SAMPLES = 98
DEV_SIZE = 49
TOP_FEATURES = 25         # 根据各自消融实验最佳特征数

# 待测试参数
TOP_K_LIST = list(range(1, 8))
SEED_LIST = [42, 123, 2024]

# 加载或构建增强知识库索引
def get_enhanced_index():
    index_path = "./faiss_free_enhanced_index"
    if os.path.exists(index_path):
        print(f"加载已有索引 {index_path}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        return vector_store
    else:
        print("构建增强知识库索引...")
        docs = []
        for file_path in [TEXTBOOK_FILE, ENHANCED_RANGE_FILE]:
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

print("加载模型和知识库...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

vector_store = get_enhanced_index()

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

def get_feature(report, knowledge):
    prompt = f"检验报告：{report}\n\n知识：{knowledge}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[LAYER][0, -1, :]
    features = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    return features

# 加载所有样本
all_samples = []
with open(ALL_TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_samples.append(json.loads(line))
print(f"总样本数: {len(all_samples)}")

results = {}

for top_k in TOP_K_LIST:
    for seed in SEED_LIST:
        random.seed(seed)
        samples = random.sample(all_samples, TOTAL_SAMPLES)
        dev_samples = samples[:DEV_SIZE]
        test_samples = samples[DEV_SIZE:]

        # 开发集特征
        dev_features, dev_labels = [], []
        for s in tqdm(dev_samples, desc=f"TOP_K={top_k}, seed={seed} (dev)", leave=False):
            report = s["report"]
            retrieved = vector_store.similarity_search(report, k=top_k)
            knowledge = "\n\n".join([d.page_content for d in retrieved])
            feat = get_feature(report, knowledge)
            dev_features.append(feat)
            dev_labels.append(s["label"])
        dev_features = np.array(dev_features)
        dev_labels = np.array(dev_labels)

        # 特征选择 (t-statistic)
        correct_mask = (dev_labels == 1)
        error_mask = (dev_labels == 0)
        if correct_mask.sum() == 0 or error_mask.sum() == 0:
            print(f"警告: 开发集中正负样本不足，跳过 (seed={seed}, top_k={top_k})")
            results[(top_k, seed)] = 0.5
            continue
        mean_c = dev_features[correct_mask].mean(axis=0)
        mean_e = dev_features[error_mask].mean(axis=0)
        std_c = dev_features[correct_mask].std(axis=0) + 1e-8
        std_e = dev_features[error_mask].std(axis=0) + 1e-8
        pooled = np.sqrt(std_c**2 + std_e**2)
        t_score = np.abs(mean_e - mean_c) / pooled
        t_score[np.isnan(t_score)] = 0
        top_idx = np.argsort(t_score)[-TOP_FEATURES:][::-1]
        top_weights = t_score[top_idx]

        # 测试集评估
        test_risks, test_true_error = [], []
        for s in tqdm(test_samples, desc=f"TOP_K={top_k}, seed={seed} (test)", leave=False):
            report = s["report"]
            retrieved = vector_store.similarity_search(report, k=top_k)
            knowledge = "\n\n".join([d.page_content for d in retrieved])
            feat = get_feature(report, knowledge)
            risk = np.sum(feat[top_idx] * top_weights)
            test_risks.append(risk)
            test_true_error.append(1 - s["label"])
        auc = roc_auc_score(test_true_error, test_risks)
        results[(top_k, seed)] = auc
        print(f"TOP_K={top_k}, seed={seed} -> AUC = {auc:.4f}")

print("\n=== 消融实验汇总 (层13, 增强知识库, 特征数=21) ===")
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