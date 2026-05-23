#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuatuoGPT-o1 自由文本特征数量消融（增强知识库） - 98+98 样本版
对候选层 3, 6, 27 分别进行特征数量 1-30 消融，输出 AUC 及 L1 基线
使用独立缓存目录 ./cache_huatuo_free_ablation，避免与其他实验冲突
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
import shutil

# ================= 配置 =================
MODEL_PATH = "./models/HuatuoGPT-o1-8B"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
ENHANCED_RANGE_FILE = "./knowledge/reference_ranges_enhanced.jsonl"
DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE = "./data/free_text_test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
DEVICE = "cuda"

# 候选层列表（可修改）
LAYERS = [3, 6, 27]

# 使用独立缓存目录，避免与其他实验冲突
CACHE_DIR = "./cache_huatuo_free_ablation"
# 如果目录已存在，删除它（强制重新提取特征）
if os.path.exists(CACHE_DIR):
    print(f"删除旧缓存目录 {CACHE_DIR}，强制重新提取特征...")
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

# ================= 加载模型 =================
print("加载 HuatuoGPT-o1-8B 模型...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# ================= 增强知识库索引 =================
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
        return vector_store

vector_store = get_enhanced_index()

# ================= 加载数据集（验证样本数） =================
dev_samples = [json.loads(line) for line in open(DEV_FILE, encoding='utf-8')]
test_samples = [json.loads(line) for line in open(TEST_FILE, encoding='utf-8')]
print(f"开发集 {len(dev_samples)} 条，测试集 {len(test_samples)} 条")
assert len(dev_samples) == 98, f"开发集应为 98 条，实际 {len(dev_samples)}"
assert len(test_samples) == 98, f"测试集应为 98 条，实际 {len(test_samples)}"

# ================= 特征提取函数（不使用旧缓存，但会使用新的独立缓存目录） =================
def get_features_for_layer(layer):
    # 缓存文件路径
    cache_dev = os.path.join(CACHE_DIR, f"layer{layer}_dev_features.npy")
    cache_test = os.path.join(CACHE_DIR, f"layer{layer}_test_features.npy")
    
    # 如果缓存存在，直接加载
    if os.path.exists(cache_dev) and os.path.exists(cache_test):
        print(f"层 {layer} 使用缓存特征")
        dev_features = np.load(cache_dev)
        test_features = np.load(cache_test)
        return dev_features, test_features
    
    print(f"层 {layer} 提取特征...")
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
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
        hidden = outputs.hidden_states[layer][0, -1, :]
        features = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
        return features

    dev_features = []
    for s in tqdm(dev_samples, desc=f"层{layer} 开发集"):
        retrieved = vector_store.similarity_search(s["report"], k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        feat = get_feature(s["report"], knowledge)
        dev_features.append(feat)
    dev_features = np.array(dev_features)
    np.save(cache_dev, dev_features)

    test_features = []
    for s in tqdm(test_samples, desc=f"层{layer} 测试集"):
        retrieved = vector_store.similarity_search(s["report"], k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        feat = get_feature(s["report"], knowledge)
        test_features.append(feat)
    test_features = np.array(test_features)
    np.save(cache_test, test_features)

    return dev_features, test_features

# ================= 逐层消融 =================
for LAYER in LAYERS:
    print(f"\n========== 层 {LAYER} ==========")
    dev_features, test_features = get_features_for_layer(LAYER)
    dev_labels = np.array([s["label"] for s in dev_samples])
    test_labels = np.array([s["label"] for s in test_samples])
    test_true_error = 1 - test_labels  # 1表示错误（不忠实）

    # 计算区分度
    correct_mask = (dev_labels == 1)
    error_mask = (dev_labels == 0)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        print("开发集正负样本不足，跳过")
        continue
    mean_c = dev_features[correct_mask].mean(axis=0)
    mean_e = dev_features[error_mask].mean(axis=0)
    std_c = dev_features[correct_mask].std(axis=0) + 1e-8
    std_e = dev_features[error_mask].std(axis=0) + 1e-8
    pooled = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled
    t_score[np.isnan(t_score)] = 0
    sorted_idx = np.argsort(t_score)[::-1]

    print(f"\n特征数量消融 (层{LAYER}, 增强知识库, 98+98):")
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

    # L1 基线
    l1_risks = np.sum(test_features, axis=1)
    auc_l1 = roc_auc_score(test_true_error, l1_risks)
    print(f"L1 范数   : AUC = {auc_l1:.4f}")