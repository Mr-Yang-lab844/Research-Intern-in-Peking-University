#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama-3.1-8B-UltraMedical 自由文本无 RAG 特征数量消融 (层2)
"""

import json
import torch
import numpy as np
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE = "./data/free_text_test.jsonl"
SAE_TOP_K = 32
DEVICE = "cuda"
LAYER = 15

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

def get_feature(report, layer, W, b):
    prompt = f"在某一次体检中，我的{report}，正常吗？"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outs = model(**inputs, output_hidden_states=True)
    hidden = outs.hidden_states[layer][0, -1, :]
    feat = sae_encode(hidden.unsqueeze(0), W, b).squeeze(0).cpu().numpy()
    return feat

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

dev = load_jsonl(DEV_FILE)
test = load_jsonl(TEST_FILE)
print(f"开发集 {len(dev)} 条，测试集 {len(test)} 条")

cache_dev = f"./cache/layer{LAYER}_dev_feats.npy"
cache_test = f"./cache/layer{LAYER}_test_feats.npy"
os.makedirs("./cache", exist_ok=True)

if os.path.exists(cache_dev) and os.path.exists(cache_test):
    print("加载缓存特征...")
    dev_feats = np.load(cache_dev)
    test_feats = np.load(cache_test)
    dev_labels = np.array([1 - s['label'] for s in dev])  # 1=错误
    test_labels = np.array([1 - s['label'] for s in test])
else:
    print("提取开发集特征...")
    W, b = load_sae(LAYER)
    dev_feats = []
    for s in tqdm(dev):
        feat = get_feature(s['report'], LAYER, W, b)
        dev_feats.append(feat)
    dev_feats = np.array(dev_feats)
    dev_labels = np.array([1 - s['label'] for s in dev])
    print("提取测试集特征...")
    test_feats = []
    for s in tqdm(test):
        feat = get_feature(s['report'], LAYER, W, b)
        test_feats.append(feat)
    test_feats = np.array(test_feats)
    test_labels = np.array([1 - s['label'] for s in test])
    np.save(cache_dev, dev_feats)
    np.save(cache_test, test_feats)

# 计算 t-score
correct_mask = (dev_labels == 0)
error_mask = (dev_labels == 1)
mean_c = dev_feats[correct_mask].mean(axis=0)
mean_e = dev_feats[error_mask].mean(axis=0)
std_c = dev_feats[correct_mask].std(axis=0) + 1e-8
std_e = dev_feats[error_mask].std(axis=0) + 1e-8
pooled = np.sqrt(std_c**2 + std_e**2)
t_score = np.abs(mean_e - mean_c) / pooled
t_score[np.isnan(t_score)] = 0
sorted_idx = np.argsort(t_score)[::-1]

print("\n特征数量消融 (层15):")
best_auc = 0
best_k = 0
for k in range(1, 31):
    top_idx = sorted_idx[:k]
    top_w = t_score[top_idx]
    risks = np.sum(test_feats[:, top_idx] * top_w, axis=1)
    auc = roc_auc_score(test_labels, risks)
    if auc > best_auc:
        best_auc = auc
        best_k = k
    print(f"Top-{k:2d} 特征: AUC = {auc:.4f}")
print(f"最佳: Top-{best_k} 特征, AUC = {best_auc:.4f}")

# L1 基线
l1_risks = np.sum(test_feats, axis=1)
auc_l1 = roc_auc_score(test_labels, l1_risks)
print(f"L1 范数: AUC = {auc_l1:.4f}")