#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama-3.1-8B-UltraMedical 自由文本无 RAG 全层扫描
使用自然语言 prompt: "在某一次体检中，我的{指标}是{数值}{单位}，正常吗？"
仅前向传播，提取最后一层最后一个 token 的隐藏状态，SAE 编码，计算每层 AUC
"""

import json
import torch
import numpy as np
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

# ================= 配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"   # 兼容 tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE = "./data/free_text_test.jsonl"
SAE_TOP_K = 32
LAYERS = list(range(32))

# ================= 加载模型 =================
print("Loading Llama-3.1-8B-UltraMedical...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

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
    return torch.relu(features)

def get_feature(report_text, layer, W_enc, b_enc):
    """
    构造自然语言问句: "在某一次体检中，我的{指标}是{数值}{单位}，正常吗？"
    注意: report_text 已经是类似 "血糖 5.0 mmol/L" 的字符串，直接嵌入即可
    """
    # 去掉可能的标点
    report_clean = report_text.strip()
    prompt = f"在某一次体检中，我的{report_clean}，正常吗？"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0, -1, :]
    features = sae_encode(hidden.unsqueeze(0), W_enc, b_enc).squeeze(0).cpu().numpy()
    return features

# ================= 加载数据集 =================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

dev_samples = load_jsonl(DEV_FILE)
test_samples = load_jsonl(TEST_FILE)
print(f"开发集: {len(dev_samples)} 条, 测试集: {len(test_samples)} 条")

# ================= 逐层扫描 =================
results = {}
for layer in LAYERS:
    print(f"\n=== 测试层 {layer} ===")
    W_enc, b_enc = load_sae(layer)

    # 开发集特征和标签（error_label = 1 - true_label，因为 dev 中 label=1 表示正确）
    dev_features = []
    dev_labels = []   # 1 表示错误（不忠实），0 表示正确
    for sample in tqdm(dev_samples, desc="开发集"):
        report = sample["report"]
        true_label = sample["label"]   # 1=正确, 0=错误
        error_label = 1 - true_label
        feat = get_feature(report, layer, W_enc, b_enc)
        dev_features.append(feat)
        dev_labels.append(error_label)
    dev_features = np.array(dev_features)
    dev_labels = np.array(dev_labels)

    # 特征选择（t-statistic）
    correct_mask = (dev_labels == 0)
    error_mask = (dev_labels == 1)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        print(f"开发集正负样本不足，跳过层{layer}")
        results[layer] = 0.5
        continue

    mean_c = dev_features[correct_mask].mean(axis=0)
    mean_e = dev_features[error_mask].mean(axis=0)
    std_c = dev_features[correct_mask].std(axis=0) + 1e-8
    std_e = dev_features[error_mask].std(axis=0) + 1e-8
    pooled_std = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled_std
    t_score[np.isnan(t_score)] = 0

    top_k = 10
    top_indices = np.argsort(t_score)[-top_k:][::-1]
    top_weights = t_score[top_indices]

    # 测试集
    test_features = []
    test_true_error = []   # 1 表示错误（不忠实）
    for sample in tqdm(test_samples, desc="测试集"):
        report = sample["report"]
        label = sample["label"]   # 1=正确, 0=错误
        error_label = 1 - label
        feat = get_feature(report, layer, W_enc, b_enc)
        test_features.append(feat)
        test_true_error.append(error_label)
    test_features = np.array(test_features)
    test_true_error = np.array(test_true_error)

    risks = np.sum(test_features[:, top_indices] * top_weights, axis=1)
    auc = roc_auc_score(test_true_error, risks)
    results[layer] = auc
    print(f"AUC = {auc:.4f}")

print("\n=== Llama-3.1-8B-UltraMedical 自由文本无 RAG 全层扫描 AUC 汇总 ===")
for l in sorted(results.keys()):
    print(f"层 {l:2d}: {results[l]:.4f}")