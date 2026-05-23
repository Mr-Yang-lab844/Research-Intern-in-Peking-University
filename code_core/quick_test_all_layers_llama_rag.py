#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试原始 Llama 自由文本所有层（增强知识库），TOP_K=3，加权和，使用 196 条样本（开发98+测试98）"""

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

def load_sae(layer):
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
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

def get_feature(report, knowledge, layer, W_enc, b_enc):
    prompt = f"检验报告：{report}\n\n知识：{knowledge}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0, -1, :]
    features = sae_encode(hidden.unsqueeze(0), W_enc, b_enc).squeeze(0).cpu().numpy()
    return features

dev_samples = [json.loads(line) for line in open(DEV_FILE)]
test_samples = [json.loads(line) for line in open(TEST_FILE)]
print(f"开发集 {len(dev_samples)} 条，测试集 {len(test_samples)} 条")
assert len(dev_samples) == 98 and len(test_samples) == 98, "请确保 free_text_dev.jsonl 和 free_text_test.jsonl 各含 98 条样本"

layers = list(range(0, 32))
results = {}

for layer in layers:
    print(f"\n=== 测试层 {layer} ===")
    W_enc, b_enc = load_sae(layer)
    
    # 开发集
    dev_features, dev_labels = [], []
    for sample in tqdm(dev_samples, desc="开发集"):
        report = sample["report"]
        retrieved = vector_store.similarity_search(report, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        feat = get_feature(report, knowledge, layer, W_enc, b_enc)
        dev_features.append(feat)
        dev_labels.append(sample["label"])
    X_dev = np.array(dev_features)
    y_dev = np.array(dev_labels)
    
    correct = X_dev[y_dev==1]
    error = X_dev[y_dev==0]
    if len(correct)==0 or len(error)==0:
        print("开发集中缺乏正样本或负样本，跳过")
        results[layer] = 0.5
        continue
    mean_c = correct.mean(axis=0)
    mean_e = error.mean(axis=0)
    std_c = correct.std(axis=0) + 1e-8
    std_e = error.std(axis=0) + 1e-8
    pooled = np.sqrt(std_c**2 + std_e**2)
    t = np.abs(mean_e - mean_c) / pooled
    t[np.isnan(t)] = 0
    top_k = 10
    top_idx = np.argsort(t)[-top_k:][::-1]
    top_w = t[top_idx]
    print(f"Top 特征得分 (前5): {top_w[:5]}")
    
    # 测试集
    risks, true_err = [], []
    for sample in tqdm(test_samples, desc="测试集"):
        report = sample["report"]
        retrieved = vector_store.similarity_search(report, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        feat = get_feature(report, knowledge, layer, W_enc, b_enc)
        risk = np.sum(feat[top_idx] * top_w)
        risks.append(risk)
        true_err.append(1 - sample["label"])
    auc = roc_auc_score(true_err, risks)
    results[layer] = auc
    print(f"AUC = {auc:.4f}")

print("\n=== 原始 Llama 自由文本 RAG 全层扫描 AUC 汇总 (196条) ===")
for l, auc in sorted(results.items()):
    print(f"层 {l:2d}: {auc:.4f}")