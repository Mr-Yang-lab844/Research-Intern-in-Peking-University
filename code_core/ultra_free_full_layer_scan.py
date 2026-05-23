#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama-3.1-8B-UltraMedical 自由文本全层扫描（无 RAG + 有 RAG）
- 数据集：free_text_dev.jsonl (98条) + free_text_test.jsonl (98条)
- 无 RAG：自然语言 prompt "在某一次体检中，我的{report}，正常吗？"
- 有 RAG：增强知识库（教科书+离散化参考范围），检索 Top‑K=3，prompt "检验报告：{report}\n\n知识：{knowledge}\n\n解读："
- 每层使用 Top-10 特征加权和，计算 AUC
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

# ================= 配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE = "./data/free_text_test.jsonl"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
ENHANCED_RANGE_FILE = "./knowledge/reference_ranges_enhanced.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
LAYERS = list(range(32))

# ================= 加载模型 =================
print("Loading Llama-3.1-8B-UltraMedical...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# ================= 加载 SAE 权重 =================
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

# ================= 无 RAG 特征提取 =================
def get_feature_no_rag(report_text, layer, W, b):
    prompt = f"在某一次体检中，我的{report_text}，正常吗？"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0, -1, :]
    feat = sae_encode(hidden.unsqueeze(0), W, b).squeeze(0).cpu().numpy()
    return feat

# ================= 有 RAG 知识库索引 =================
def get_enhanced_index():
    index_path = "./faiss_free_enhanced_ultra"
    if os.path.exists(index_path):
        print("加载已有增强知识库索引")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("构建增强知识库索引...")
        docs = []
        for fp in [TEXTBOOK_FILE, ENHANCED_RANGE_FILE]:
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if "text" in data:
                        content = data["text"]
                    elif "description" in data:
                        content = data["description"]
                    else:
                        continue
                    docs.append(Document(page_content=content))
        print(f"文档数: {len(docs)}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vs = FAISS.from_documents(docs, embedding_model)
        vs.save_local(index_path)
        return vs

vector_store = get_enhanced_index()

def get_feature_rag(report_text, layer, W, b):
    retrieved = vector_store.similarity_search(report_text, k=TOP_K_RETRIEVAL)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    prompt = f"检验报告：{report_text}\n\n知识：{knowledge}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0, -1, :]
    feat = sae_encode(hidden.unsqueeze(0), W, b).squeeze(0).cpu().numpy()
    return feat

# ================= 加载数据集（各98条） =================
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

dev_samples = load_jsonl(DEV_FILE)
test_samples = load_jsonl(TEST_FILE)
print(f"开发集: {len(dev_samples)} 条, 测试集: {len(test_samples)} 条")
assert len(dev_samples) == 98 and len(test_samples) == 98

# ================= 全层扫描函数 =================
def full_layer_scan(get_feature_func, experiment_name):
    print(f"\n========== {experiment_name} ==========")
    results = {}
    for layer in LAYERS:
        print(f"\n--- 层 {layer} ---")
        W, b = load_sae(layer)

        # 开发集
        dev_feats, dev_labels = [], []
        for s in tqdm(dev_samples, desc="开发集"):
            report = s['report']
            true_label = s['label']   # 1=正确,0=错误
            error_label = 1 - true_label
            feat = get_feature_func(report, layer, W, b)
            dev_feats.append(feat)
            dev_labels.append(error_label)
        X_dev = np.array(dev_feats)
        y_dev = np.array(dev_labels)

        correct_mask = (y_dev == 0)
        error_mask = (y_dev == 1)
        if correct_mask.sum() == 0 or error_mask.sum() == 0:
            print("开发集正负样本不足，跳过")
            results[layer] = 0.5
            continue

        mean_c = X_dev[correct_mask].mean(axis=0)
        mean_e = X_dev[error_mask].mean(axis=0)
        std_c = X_dev[correct_mask].std(axis=0) + 1e-8
        std_e = X_dev[error_mask].std(axis=0) + 1e-8
        pooled = np.sqrt(std_c**2 + std_e**2)
        t_score = np.abs(mean_e - mean_c) / pooled
        t_score[np.isnan(t_score)] = 0
        top_k = 10
        top_idx = np.argsort(t_score)[-top_k:][::-1]
        top_w = t_score[top_idx]

        # 测试集
        test_feats, test_errors = [], []
        for s in tqdm(test_samples, desc="测试集"):
            report = s['report']
            true_label = s['label']
            error_label = 1 - true_label
            feat = get_feature_func(report, layer, W, b)
            test_feats.append(feat)
            test_errors.append(error_label)
        X_test = np.array(test_feats)
        y_test = np.array(test_errors)
        risks = np.sum(X_test[:, top_idx] * top_w, axis=1)
        auc = roc_auc_score(y_test, risks)
        results[layer] = auc
        print(f"AUC = {auc:.4f}")

    print(f"\n=== {experiment_name} AUC 汇总 ===")
    for l, auc in sorted(results.items()):
        print(f"层 {l:2d}: {auc:.4f}")
    return results

# ================= 运行两个实验 =================
# 无 RAG
results_no_rag = full_layer_scan(get_feature_no_rag, "无 RAG 全层扫描")

# 有 RAG
results_rag = full_layer_scan(get_feature_rag, "有 RAG 全层扫描 (Top-K=3)")