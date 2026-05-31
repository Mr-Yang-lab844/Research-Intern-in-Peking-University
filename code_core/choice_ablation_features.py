#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始 Llama-3.1-8B-Instruct 选择题特征数量消融
层：22,23,24,25,26,27,28
特征数量：1～30
固定检索 Top‑K=3，随机种子 42，开发集 100，测试集 100
知识库：教科书+参考范围
"""

import json
import random
import re
import numpy as np
import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ================= 固定配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
TEST_FILE = "./data/test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
MAX_NEW_TOKENS = 10
DEVICE = "cuda"
RANDOM_SEED = 42
TOTAL_SAMPLES = 200
DEV_SIZE = 100

# 待扫描的层列表
LAYERS = [22, 23, 24, 25, 26, 27, 28]

# 特征数量 1～30
FEATURE_COUNTS = list(range(1, 31))

print("加载模型和tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 加载 FAISS 索引（教科书+参考范围）
print("加载知识库索引...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
vector_store = FAISS.load_local("./faiss_merged_index", embedding_model, allow_dangerous_deserialization=True)

# 加载测试集并抽样
with open(TEST_FILE, "r", encoding="utf-8") as f:
    all_samples = [json.loads(line) for line in f]
random.seed(RANDOM_SEED)
samples = random.sample(all_samples, TOTAL_SAMPLES)
dev_samples = samples[:DEV_SIZE]
test_samples = samples[DEV_SIZE:]

# 预先计算每个样本的正确答案标签（与层无关，可复用）
def generate_answer(question, knowledge):
    prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{knowledge}

Question:
{question}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    match = re.search(r'\b([A-E])\b', answer)
    return match.group(1) if match else ""

print("预生成开发集和测试集正确性标签...")
dev_labels = []   # 1=错误, 0=正确
for s in tqdm(dev_samples):
    q = s["input"]
    true = s["output"]
    retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    pred = generate_answer(q, knowledge)
    dev_labels.append(0 if pred == true else 1)

test_labels = []
for s in tqdm(test_samples):
    q = s["input"]
    true = s["output"]
    retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    pred = generate_answer(q, knowledge)
    test_labels.append(0 if pred == true else 1)

dev_labels = np.array(dev_labels)
test_labels = np.array(test_labels)

# 对每个层进行消融
for layer in LAYERS:
    print(f"\n========== 层 {layer} 特征数量消融 ==========")

    # 加载该层的 SAE
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

    def get_feature(question, knowledge):
        prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{knowledge}

Question:
{question}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer][0, -1, :]   # 注意：layer是整数，需要索引
        features = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
        return features

    # 提取开发集特征
    print("提取开发集特征...")
    dev_feats = []
    for s in tqdm(dev_samples):
        q = s["input"]
        retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        feat = get_feature(q, knowledge)
        dev_feats.append(feat)
    dev_feats = np.array(dev_feats)

    # 提取测试集特征
    print("提取测试集特征...")
    test_feats = []
    for s in tqdm(test_samples):
        q = s["input"]
        retrieved = vector_store.similarity_search(q, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        feat = get_feature(q, knowledge)
        test_feats.append(feat)
    test_feats = np.array(test_feats)

    # 计算 t 分数（区分度）
    correct_mask = (dev_labels == 0)
    error_mask = (dev_labels == 1)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        print("开发集中正确或错误样本数量为0，跳过该层")
        continue

    mean_c = dev_feats[correct_mask].mean(axis=0)
    mean_e = dev_feats[error_mask].mean(axis=0)
    std_c = dev_feats[correct_mask].std(axis=0) + 1e-8
    std_e = dev_feats[error_mask].std(axis=0) + 1e-8
    pooled_std = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled_std
    t_score[np.isnan(t_score)] = 0

    # 按 t_score 降序排序
    sorted_idx = np.argsort(t_score)[::-1]

    # 对每个特征数量计算 AUC
    print("特征数量消融结果:")
    for k in FEATURE_COUNTS:
        top_idx = sorted_idx[:k]
        top_weights = t_score[top_idx]
        risks = np.sum(test_feats[:, top_idx] * top_weights, axis=1)
        auc = roc_auc_score(test_labels, risks)
        print(f"Top-{k:2d} 特征: AUC = {auc:.4f}")

    # L1 范数基线
    l1_risks = np.sum(test_feats, axis=1)
    auc_l1 = roc_auc_score(test_labels, l1_risks)
    print(f"L1 范数: AUC = {auc_l1:.4f}")