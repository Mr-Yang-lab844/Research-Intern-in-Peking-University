#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自由文本层16消融：特征数量1-10，加权和，TOP_K=3
输出AUC及L1范数基线
"""

import json
import numpy as np
import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 固定参数
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE = "./data/free_text_test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
DEVICE = "cuda"
LAYER = 16

print("加载模型和知识库...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
vector_store = FAISS.load_local("./faiss_merged_index", embedding_model, allow_dangerous_deserialization=True)

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

# 加载数据
dev_samples = [json.loads(line) for line in open(DEV_FILE)]
test_samples = [json.loads(line) for line in open(TEST_FILE)]
print(f"开发集 {len(dev_samples)} 条，测试集 {len(test_samples)} 条")

# 预提取特征（仅一次，避免重复检索）
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

print("提取测试集特征...")
test_features, test_labels = [], []
for s in tqdm(test_samples):
    retrieved = vector_store.similarity_search(s["report"], k=TOP_K_RETRIEVAL)
    knowledge = "\n\n".join([d.page_content for d in retrieved])
    feat = get_feature(s["report"], knowledge)
    test_features.append(feat)
    test_labels.append(s["label"])
test_features = np.array(test_features)
test_true_error = 1 - np.array(test_labels)  # 1=错误

# 计算全特征区分度
correct_mask = (dev_labels == 1)
error_mask = (dev_labels == 0)
if correct_mask.sum()==0 or error_mask.sum()==0:
    raise ValueError("开发集正负样本不足")
mean_c = dev_features[correct_mask].mean(axis=0)
mean_e = dev_features[error_mask].mean(axis=0)
std_c = dev_features[correct_mask].std(axis=0) + 1e-8
std_e = dev_features[error_mask].std(axis=0) + 1e-8
pooled = np.sqrt(std_c**2 + std_e**2)
t_score = np.abs(mean_e - mean_c) / pooled
t_score[np.isnan(t_score)] = 0
sorted_idx = np.argsort(t_score)[::-1]

# 对不同特征数量计算AUC
print("\n特征数量消融:")
for k in [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]:
    top_idx = sorted_idx[:k]
    top_w = t_score[top_idx]
    risks = np.sum(test_features[:, top_idx] * top_w, axis=1)
    auc = roc_auc_score(test_true_error, risks)
    print(f"Top-{k:2d} 特征: AUC = {auc:.4f}")

# L1 范数基线
l1_risks = np.sum(test_features, axis=1)
auc_l1 = roc_auc_score(test_true_error, l1_risks)
print(f"L1 范数   : AUC = {auc_l1:.4f}")