#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama-3.1-8B-UltraMedical 选择题（无 RAG）SAE 全层扫描 + 准确率测试
"""

import json
import re
import random
import torch
import numpy as np
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
TEST_FILE = "./data/test.jsonl"
SAE_TOP_K = 32
TOTAL_SAMPLES = 100
DEV_SIZE = 50
RANDOM_SEED = 42
MAX_NEW_TOKENS = 500
LAYERS = list(range(32))

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

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

def get_feature_and_answer(question, layer, W, b):
    prompt = f"{question}\n\n请只输出正确答案的字母（例如：B）。"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    last_hidden = outputs.hidden_states[-1][-1][:, -1, :]
    feat = sae_encode(last_hidden, W, b).squeeze(0).cpu().numpy()
    full = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    match = re.search(r'\b([A-E])\b', full)
    pred = match.group(1) if match else ""
    return feat, pred

# 加载数据
with open(TEST_FILE) as f:
    all_samples = [json.loads(line) for line in f]
random.seed(RANDOM_SEED)
samples = random.sample(all_samples, TOTAL_SAMPLES)
dev_samples = samples[:DEV_SIZE]
test_samples = samples[DEV_SIZE:]

# 准确率（在测试集上）
correct = 0
for s in test_samples:
    _, pred = get_feature_and_answer(s["input"], 0, *load_sae(0))  # 随便用一层，只为了得到答案
    if pred == s["output"]:
        correct += 1
print(f"模型自身准确率（测试集 {len(test_samples)} 条）: {correct}/{len(test_samples)} = {correct/len(test_samples)*100:.2f}%")

# 全层扫描
results = {}
for layer in LAYERS:
    print(f"\n=== 层 {layer} ===")
    W, b = load_sae(layer)
    dev_feats, dev_labels = [], []
    for s in tqdm(dev_samples, desc="开发集"):
        feat, pred = get_feature_and_answer(s["input"], layer, W, b)
        dev_feats.append(feat)
        dev_labels.append(0 if pred == s["output"] else 1)
    X_dev = np.array(dev_feats)
    y_dev = np.array(dev_labels)
    if y_dev.sum() == 0 or (len(y_dev)-y_dev.sum()) == 0:
        results[layer] = 0.5
        continue
    correct_mask = (y_dev == 0)
    error_mask = (y_dev == 1)
    mean_c = X_dev[correct_mask].mean(axis=0)
    mean_e = X_dev[error_mask].mean(axis=0)
    std_c = X_dev[correct_mask].std(axis=0) + 1e-8
    std_e = X_dev[error_mask].std(axis=0) + 1e-8
    t_score = np.abs(mean_e - mean_c) / np.sqrt(std_c**2 + std_e**2)
    t_score[np.isnan(t_score)] = 0
    top_idx = np.argsort(t_score)[-10:][::-1]
    top_w = t_score[top_idx]
    test_feats, test_errors = [], []
    for s in tqdm(test_samples, desc="测试集"):
        feat, pred = get_feature_and_answer(s["input"], layer, W, b)
        test_feats.append(feat)
        test_errors.append(0 if pred == s["output"] else 1)
    X_test = np.array(test_feats)
    y_test = np.array(test_errors)
    risks = np.sum(X_test[:, top_idx] * top_w, axis=1)
    auc = roc_auc_score(y_test, risks)
    results[layer] = auc
    print(f"AUC = {auc:.4f}")

print("\n=== AUC 汇总 ===")
for l, auc in sorted(results.items()):
    print(f"层 {l:2d}: {auc:.4f}")