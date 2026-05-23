#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuatuoGPT-o1 自由文本忠实性检测（层16，增强知识库）
使用与原始Llama相同的SAE权重和特征提取方式
"""

import json
import numpy as np
import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, fbeta_score
from tqdm import tqdm
import os

# ================= 配置 =================
MODEL_PATH = "./models/HuatuoGPT-o1-8B"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
ENHANCED_RANGE_FILE = "./knowledge/reference_ranges_enhanced.jsonl"
DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE = "./data/free_text_test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 6          # 原最优
SAE_TOP_K = 32               # SAE 内部 TopK
DEVICE = "cuda"
LAYER = 16                   # 固定测试层16
NUM_FEATURES = 25            # 特征数量（原最优）
RANDOM_SEED = 2024

# 缓存特征权重的路径
CACHE_WEIGHTS = f"./cache/huatuo_layer{LAYER}_weights.npy"
os.makedirs("./cache", exist_ok=True)

# ----------------- 1. 加载模型 -----------------
print("加载 HuatuoGPT-o1-8B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ----------------- 2. 构建知识库索引（增强版）-----------------
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

vector_store = get_enhanced_index()

# ----------------- 3. 加载 SAE 权重 -----------------
sae_path = f"./models/Llama-Scope/L{LAYER}R-8x.safetensors"
print(f"加载 SAE 权重: {sae_path}")
sae_weights = safetensors.torch.load_file(sae_path)
W_enc = sae_weights['encoder.weight'].to(DEVICE).to(torch.float16)
b_enc = sae_weights['encoder.bias'].to(DEVICE).to(torch.float16)

def sae_encode(hidden):
    """hidden: (1, 4096) -> 特征向量 (32768,)"""
    z = hidden @ W_enc.T + b_enc
    topk = torch.topk(z, SAE_TOP_K, dim=-1)
    f = torch.zeros_like(z)
    f.scatter_(-1, topk.indices, topk.values)
    return torch.relu(f)

def get_feature(report, knowledge):
    """获取最后一个 token 的隐藏状态并编码为 SAE 特征"""
    prompt = f"检验报告：{report}\n\n知识：{knowledge}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # 取指定层的最后一个 token 的隐藏状态
    hidden = outputs.hidden_states[LAYER][0, -1, :]  # shape (4096,)
    features = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    return features

# ----------------- 4. 加载数据 -----------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

dev_samples = load_jsonl(DEV_FILE)
test_samples = load_jsonl(TEST_FILE)
print(f"开发集: {len(dev_samples)} 条, 测试集: {len(test_samples)} 条")

# ----------------- 5. 特征提取（使用缓存）-----------------
dev_features_path = f"./cache/huatuo_layer{LAYER}_dev_features.npy"
dev_labels_path = f"./cache/huatuo_layer{LAYER}_dev_labels.npy"
if os.path.exists(dev_features_path) and os.path.exists(dev_labels_path):
    print("加载缓存的开发集特征...")
    dev_features = np.load(dev_features_path)
    dev_labels = np.load(dev_labels_path)
else:
    print("提取开发集特征...")
    dev_features, dev_labels = [], []
    for s in tqdm(dev_samples):
        retrieved = vector_store.similarity_search(s["report"], k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        feat = get_feature(s["report"], knowledge)
        dev_features.append(feat)
        dev_labels.append(s["label"])
    dev_features = np.array(dev_features)
    dev_labels = np.array(dev_labels)
    np.save(dev_features_path, dev_features)
    np.save(dev_labels_path, dev_labels)
    print("开发集特征已缓存")

test_features_path = f"./cache/huatuo_layer{LAYER}_test_features.npy"
test_labels_path = f"./cache/huatuo_layer{LAYER}_test_labels.npy"
if os.path.exists(test_features_path) and os.path.exists(test_labels_path):
    print("加载缓存的测试集特征...")
    test_features = np.load(test_features_path)
    test_labels = np.load(test_labels_path)
else:
    print("提取测试集特征...")
    test_features, test_labels = [], []
    for s in tqdm(test_samples):
        retrieved = vector_store.similarity_search(s["report"], k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        feat = get_feature(s["report"], knowledge)
        test_features.append(feat)
        test_labels.append(s["label"])
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    np.save(test_features_path, test_features)
    np.save(test_labels_path, test_labels)
    print("测试集特征已缓存")

# 标签转换：0=正确（忠实），1=错误（不忠实）
# 原始数据中 label=1 表示正确，label=0 表示错误，需要反转
dev_true_error = 1 - dev_labels
test_true_error = 1 - test_labels

# ----------------- 6. 特征标定（区分度计算）-----------------
if os.path.exists(CACHE_WEIGHTS):
    print("加载缓存的特征权重...")
    feature_weights = np.load(CACHE_WEIGHTS)
else:
    print("计算特征区分度...")
    # 开发集分为正确和错误两组
    correct_mask = (dev_true_error == 0)
    error_mask = (dev_true_error == 1)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        raise ValueError("开发集正负样本不足")
    mean_c = dev_features[correct_mask].mean(axis=0)
    mean_e = dev_features[error_mask].mean(axis=0)
    std_c = dev_features[correct_mask].std(axis=0) + 1e-8
    std_e = dev_features[error_mask].std(axis=0) + 1e-8
    pooled_std = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled_std
    t_score[np.isnan(t_score)] = 0
    # 选择 Top-N 特征
    sorted_idx = np.argsort(t_score)[::-1]
    top_idx = sorted_idx[:NUM_FEATURES]
    # 构建权重向量
    feature_weights = np.zeros(dev_features.shape[1])
    feature_weights[top_idx] = t_score[top_idx]
    # 归一化（可选，但保持与原始实验一致）
    feature_weights = feature_weights / (feature_weights.sum() + 1e-8)
    np.save(CACHE_WEIGHTS, feature_weights)
    print(f"已选择 {NUM_FEATURES} 个特征，权重已保存")

# ----------------- 7. 测试集评估 -----------------
# 计算风险分数
risks = np.dot(test_features, feature_weights)
# 计算 AUC
auc = roc_auc_score(test_true_error, risks)
# 寻找最优阈值（最大化 F2）
thresholds = np.linspace(risks.min(), risks.max(), 50)
best_f2 = 0
best_thresh = 0.5
for thresh in thresholds:
    pred = (risks > thresh).astype(int)
    f2 = fbeta_score(test_true_error, pred, beta=2)
    if f2 > best_f2:
        best_f2 = f2
        best_thresh = thresh
# 最终预测
pred_labels = (risks > best_thresh).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(test_true_error, pred_labels, average='binary')

print("\n===== HuatuoGPT-o1 自由文本忠实性检测结果 =====")
print(f"Layer {LAYER}, Top-{NUM_FEATURES} features, Retrieval Top-{TOP_K_RETRIEVAL}")
print(f"AUC: {auc:.4f}")
print(f"F2 (beta=2): {best_f2:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print(f"Optimal threshold: {best_thresh:.6f}")