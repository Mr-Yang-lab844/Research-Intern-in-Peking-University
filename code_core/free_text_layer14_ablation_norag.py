#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuatuoGPT-o1 自由文本无 RAG 特征数量消融（层14） - 使用原始固定划分
- 直接使用 free_text_dev.jsonl (98条) 和 free_text_test.jsonl (98条)
- 自然语言 prompt: "在某一次体检中，我的{report}，正常吗？"
- 特征数量 1-30，输出各 AUC 及 L1 基线
"""

import json
import numpy as np
import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

MODEL_PATH = "./models/HuatuoGPT-o1-8B"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
DEVICE = "cuda"

DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE = "./data/free_text_test.jsonl"
SAE_TOP_K = 32
LAYER = 14

CACHE_DIR = "./cache_huatuo_free_layer14_fixed2"
os.makedirs(CACHE_DIR, exist_ok=True)

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

def load_sae(layer):
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
    w = safetensors.torch.load_file(sae_path)
    return w['encoder.weight'].to(DEVICE).to(torch.float16), w['encoder.bias'].to(DEVICE).to(torch.float16)

W_enc, b_enc = load_sae(LAYER)

def sae_encode(hidden):
    z = hidden @ W_enc.T + b_enc
    topk = torch.topk(z, SAE_TOP_K, dim=-1)
    f = torch.zeros_like(z)
    f.scatter_(-1, topk.indices, topk.values)
    return torch.relu(f)

def get_feature_norag(report_text):
    prompt = f"在某一次体检中，我的{report_text}，正常吗？"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[LAYER][0, -1, :]
    feat = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    return feat

# 直接使用原始开发集和测试集（不合并、不打乱）
dev_samples = [json.loads(line) for line in open(DEV_FILE, encoding='utf-8')]
test_samples = [json.loads(line) for line in open(TEST_FILE, encoding='utf-8')]
print(f"开发集 {len(dev_samples)} 条，测试集 {len(test_samples)} 条")
assert len(dev_samples) == 98 and len(test_samples) == 98

# 提取特征（缓存）
cache_file = os.path.join(CACHE_DIR, "features.npz")
if os.path.exists(cache_file):
    print("加载缓存特征...")
    data = np.load(cache_file)
    dev_features = data['dev_features']
    test_features = data['test_features']
    dev_labels = data['dev_labels']
    test_labels = data['test_labels']
else:
    print("提取开发集特征...")
    dev_features, dev_labels = [], []
    for s in tqdm(dev_samples):
        report = s['report']
        label = s['label']
        feat = get_feature_norag(report)
        dev_features.append(feat)
        dev_labels.append(label)
    dev_features = np.array(dev_features)
    dev_labels = np.array(dev_labels)

    print("提取测试集特征...")
    test_features, test_labels = [], []
    for s in tqdm(test_samples):
        report = s['report']
        label = s['label']
        feat = get_feature_norag(report)
        test_features.append(feat)
        test_labels.append(label)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    np.savez(cache_file,
             dev_features=dev_features, dev_labels=dev_labels,
             test_features=test_features, test_labels=test_labels)

dev_true_error = 1 - dev_labels
test_true_error = 1 - test_labels

# 特征选择
correct_mask = (dev_true_error == 0)
error_mask = (dev_true_error == 1)
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

print("\n特征数量消融 (无 RAG, 层14, 原始固定划分):")
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