#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始 Llama 自由文本 RAG（增强知识库）特征数量消融（层13和层16）
使用 196 条样本（开发98，测试98），固定划分，检索 Top‑K=3
输出各层特征数量 1-30 的 AUC 及 L1 基线
"""

import json
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

MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
ENHANCED_RANGE_FILE = "./knowledge/reference_ranges_enhanced.jsonl"
DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE = "./data/free_text_test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
DEVICE = "cuda"

# 要测试的层
LAYERS = [13, 16]

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

def get_enhanced_index():
    index_path = "./faiss_free_enhanced_index"
    if os.path.exists(index_path):
        print(f"加载已有索引 {index_path}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
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

vector_store = get_enhanced_index()

# 加载开发集和测试集（各98条，固定划分）
dev_samples = [json.loads(line) for line in open(DEV_FILE, encoding='utf-8')]
test_samples = [json.loads(line) for line in open(TEST_FILE, encoding='utf-8')]
print(f"开发集 {len(dev_samples)} 条，测试集 {len(test_samples)} 条")
assert len(dev_samples) == 98 and len(test_samples) == 98

# 预提取所有样本的检索知识（与层无关，但依赖于报告内容，可缓存）
print("预提取所有样本的检索知识...")
dev_knowledge = []
for s in tqdm(dev_samples):
    retrieved = vector_store.similarity_search(s["report"], k=TOP_K_RETRIEVAL)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    dev_knowledge.append(knowledge)
test_knowledge = []
for s in tqdm(test_samples):
    retrieved = vector_store.similarity_search(s["report"], k=TOP_K_RETRIEVAL)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    test_knowledge.append(knowledge)

for LAYER in LAYERS:
    print(f"\n========== 层 {LAYER} ==========")
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

    # 提取特征（使用缓存目录避免重复）
    cache_dir = f"./cache_llama_rag_layer{LAYER}"
    os.makedirs(cache_dir, exist_ok=True)
    dev_cache = os.path.join(cache_dir, "dev_features.npy")
    test_cache = os.path.join(cache_dir, "test_features.npy")
    if os.path.exists(dev_cache) and os.path.exists(test_cache):
        print("加载缓存特征...")
        dev_features = np.load(dev_cache)
        test_features = np.load(test_cache)
    else:
        print("提取开发集特征...")
        dev_features = []
        for i, s in enumerate(tqdm(dev_samples)):
            feat = get_feature(s["report"], dev_knowledge[i])
            dev_features.append(feat)
        dev_features = np.array(dev_features)
        np.save(dev_cache, dev_features)
        print("提取测试集特征...")
        test_features = []
        for i, s in enumerate(tqdm(test_samples)):
            feat = get_feature(s["report"], test_knowledge[i])
            test_features.append(feat)
        test_features = np.array(test_features)
        np.save(test_cache, test_features)

    dev_labels = np.array([s["label"] for s in dev_samples])
    test_labels = np.array([s["label"] for s in test_samples])
    test_true_error = 1 - test_labels

    correct_mask = (dev_labels == 1)
    error_mask = (dev_labels == 0)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        raise ValueError("开发集正负样本不足")
    mean_c = dev_features[correct_mask].mean(axis=0)
    mean_e = dev_features[error_mask].mean(axis=0)
    std_c = dev_features[correct_mask].std(axis=0) + 1e-8
    std_e = dev_features[error_mask].std(axis=0) + 1e-8
    pooled = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled
    t_score[np.isnan(t_score)] = 0
    sorted_idx = np.argsort(t_score)[::-1]

    print(f"\n特征数量消融 (层{LAYER}, 增强知识库, 196条样本):")
    best_auc = 0
    best_k = 0
    for k in range(1, 31):
        top_idx = sorted_idx[:k]
        top_w = t_score[top_idx]
        risks = np.sum(test_features[:, top_idx] * top_w, axis=1)
        auc = roc_auc_score(test_true_error, risks)
        if auc > best_auc:
            best_auc = auc
            best_k = k
        print(f"Top-{k:2d} 特征: AUC = {auc:.4f}")
    print(f"最佳: Top-{best_k} 特征, AUC = {best_auc:.4f}")

    l1_risks = np.sum(test_features, axis=1)
    auc_l1 = roc_auc_score(test_true_error, l1_risks)
    print(f"L1 范数   : AUC = {auc_l1:.4f}")