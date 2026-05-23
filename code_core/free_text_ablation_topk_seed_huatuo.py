#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuatuoGPT-o1 自由文本层27检索数量消融（增强知识库，固定划分）
- 使用原始 dev/test 文件，不合并打乱
- 固定特征数=10（来自特征数量消融最佳结果）
- 检索 Top‑K=1..10，输出各 Top‑K 下的 AUC
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
MODEL_PATH = "./models/HuatuoGPT-o1-8B"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
ENHANCED_RANGE_FILE = "./knowledge/reference_ranges_enhanced.jsonl"
DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE = "./data/free_text_test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
SAE_TOP_K = 32
DEVICE = "cuda"
LAYER = 27
TOP_FEATURES = 10            # 固定特征数（最佳）
TOP_K_LIST = list(range(1, 11))

# 独立缓存目录
CACHE_DIR = "./cache_huatuo_free_topk_fixed"
os.makedirs(CACHE_DIR, exist_ok=True)

# 加载原始开发集和测试集
dev_samples = [json.loads(line) for line in open(DEV_FILE, encoding='utf-8')]
test_samples = [json.loads(line) for line in open(TEST_FILE, encoding='utf-8')]
print(f"开发集 {len(dev_samples)} 条，测试集 {len(test_samples)} 条")
assert len(dev_samples) == 98 and len(test_samples) == 98

# 加载知识库索引
def get_enhanced_index():
    index_path = "./faiss_free_enhanced_index"
    if os.path.exists(index_path):
        print(f"加载已有索引 {index_path}")
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

print("加载模型和知识库...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()
vector_store = get_enhanced_index()

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

def get_feature(report, knowledge):
    prompt = f"检验报告：{report}\n\n知识：{knowledge}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[LAYER][0, -1, :]
    features = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    return features

# 预先提取开发集特征（与检索数量无关，但依赖知识，需要为每个 top_k 分别提取？开发集的特征随检索知识变化，所以必须在循环内提取。但我们可以缓存每个 top_k 的特征）
results = {}
for top_k in TOP_K_LIST:
    cache_file = os.path.join(CACHE_DIR, f"topk_{top_k}_cache.npz")
    if os.path.exists(cache_file):
        print(f"加载缓存: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        auc = data['auc']
        results[top_k] = auc
        print(f"TOP_K={top_k} -> AUC = {auc:.4f} (cached)")
        continue

    print(f"\n=== TOP_K={top_k} ===")
    # 开发集特征和标签
    dev_features = []
    dev_labels = []   # 原始标签：1=正确，0=错误
    for s in tqdm(dev_samples, desc="开发集"):
        report = s['report']
        label = s['label']
        retrieved = vector_store.similarity_search(report, k=top_k)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        feat = get_feature(report, knowledge)
        dev_features.append(feat)
        dev_labels.append(label)
    dev_features = np.array(dev_features)
    dev_labels = np.array(dev_labels)
    dev_true_error = 1 - dev_labels   # 1=错误

    # 特征选择 (t-statistic)
    correct_mask = (dev_true_error == 0)
    error_mask = (dev_true_error == 1)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        print("开发集正负样本不足，跳过")
        results[top_k] = 0.5
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
    test_risks = []
    test_true_error = []
    for s in tqdm(test_samples, desc="测试集"):
        report = s['report']
        label = s['label']
        error_label = 1 - label
        retrieved = vector_store.similarity_search(report, k=top_k)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        feat = get_feature(report, knowledge)
        risk = np.sum(feat[top_idx] * top_weights)
        test_risks.append(risk)
        test_true_error.append(error_label)
    auc = roc_auc_score(test_true_error, test_risks)

    # 保存缓存
    np.savez(cache_file, auc=auc,
             dev_features=dev_features, dev_labels=dev_labels,
             test_risks=test_risks, test_labels=test_true_error,
             top_idx=top_idx, top_weights=top_weights)
    results[top_k] = auc
    print(f"TOP_K={top_k} -> AUC = {auc:.4f} (saved)")

print("\n=== 层27检索数量消融 (固定划分，特征数=10) ===")
print("TOP_K\\AUC")
for top_k, auc in sorted(results.items()):
    print(f"{top_k}\t{auc:.4f}")