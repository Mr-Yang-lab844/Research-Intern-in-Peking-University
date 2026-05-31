#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充实验（基线对比 + 多 token 特征探索）
- 选择题基线（原始隐藏状态、SAE 无标定 L1、随机特征）
- 自由文本基线（原始隐藏状态、SAE 无标定 L1、随机特征）
- 多 token 特征探索（位置 AUC、组合、新思路）

依赖：
- 已预训练模型、SAE 权重、知识库索引、测试数据文件
- 不需要预先运行其他补充实验脚本
"""

import json
import time
import numpy as np
import torch
import safetensors.torch
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os

# ================= 全局配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./cache_supplement"
os.makedirs(CACHE_DIR, exist_ok=True)

# ================= 选择题基线（原始 Llama-3.1-8B-Instruct）=================
def choice_baselines():
    print("\n" + "="*60)
    print("选择题基线（原始 Llama-3.1-8B-Instruct）")
    print("="*60)
    
    MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    # 知识库索引
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5", model_kwargs={'device': DEVICE})
    vector_store = FAISS.load_local("./faiss_merged_index", embedding_model, allow_dangerous_deserialization=True)
    
    # 测试集（200条，后100条）
    with open("./data/test.jsonl", "r") as f:
        all_samples = [json.loads(line) for line in f]
    random.seed(42)
    samples = random.sample(all_samples, 200)
    test_samples = samples[100:]
    
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
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        match = re.search(r'\b([A-E])\b', answer)
        return match.group(1) if match else ""
    
    # 计算测试集标签（错误=1，正确=0）
    test_labels = []
    for s in tqdm(test_samples, desc="选择题-计算标签"):
        q = s["input"]
        true = s["output"]
        retrieved = vector_store.similarity_search(q, k=3)  # 与报告一致
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        pred = generate_answer(q, knowledge)
        test_labels.append(0 if pred == true else 1)
    test_labels = np.array(test_labels)
    
    # 1. 原始隐藏状态基线
    raw_hiddens = []
    for s in tqdm(test_samples, desc="选择题-提取原始隐藏状态"):
        q = s["input"]
        retrieved = vector_store.similarity_search(q, k=3)
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{knowledge}

Question:
{q}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
        raw_hiddens.append(hidden)
    raw_hiddens = np.array(raw_hiddens)
    
    l1_risks = np.sum(np.abs(raw_hiddens), axis=1)
    l2_risks = np.sqrt(np.sum(raw_hiddens**2, axis=1))
    std_risks = np.std(raw_hiddens, axis=1)
    
    auc_l1 = roc_auc_score(test_labels, l1_risks)
    auc_l2 = roc_auc_score(test_labels, l2_risks)
    auc_std = roc_auc_score(test_labels, std_risks)
    print(f"\n选择题原始隐藏状态 L1 范数 AUC = {auc_l1:.4f}")
    print(f"选择题原始隐藏状态 L2 范数 AUC = {auc_l2:.4f}")
    print(f"选择题原始隐藏状态 标准差 AUC = {auc_std:.4f}")
    
    # 2. SAE 特征全等权 L1 范数（层24，无标定）
    # 加载 SAE
    sae_path = f"./models/Llama-Scope/L24R-8x.safetensors"
    sae_w = safetensors.torch.load_file(sae_path)
    W_enc = sae_w['encoder.weight'].to(DEVICE).to(torch.float16)
    b_enc = sae_w['encoder.bias'].to(DEVICE).to(torch.float16)
    
    def sae_encode(hidden):
        z = hidden @ W_enc.T + b_enc
        topk = torch.topk(z, 32, dim=-1)
        f = torch.zeros_like(z)
        f.scatter_(-1, topk.indices, topk.values)
        return torch.relu(f)
    
    test_feats = []
    for s in tqdm(test_samples, desc="选择题-提取 SAE 特征（层24）"):
        q = s["input"]
        retrieved = vector_store.similarity_search(q, k=3)  # 检索个数（与标定一致）
        knowledge = "\n\n".join([d.page_content for d in retrieved])
        prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{knowledge}

Question:
{q}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[24][0, -1, :]
        feat = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
        test_feats.append(feat)
    test_feats = np.array(test_feats)
    sae_l1_risks = np.sum(np.abs(test_feats), axis=1)
    auc_sae_l1 = roc_auc_score(test_labels, sae_l1_risks)
    print(f"选择题 SAE 特征全等权 L1 范数（无标定） AUC = {auc_sae_l1:.4f}")
    
    # 3. 随机特征（3个，10次平均）
    n_features = 3
    n_iters = 10
    random_aucs = []
    for _ in range(n_iters):
        rand_idx = np.random.choice(test_feats.shape[1], size=n_features, replace=False)
        rand_weights = np.random.randn(n_features)
        risks = np.sum(test_feats[:, rand_idx] * rand_weights, axis=1)
        auc = roc_auc_score(test_labels, risks)
        random_aucs.append(auc)
    avg_auc = np.mean(random_aucs)
    std_auc = np.std(random_aucs)
    print(f"选择题随机特征（{n_features} 个，{n_iters} 次）AUC = {avg_auc:.4f} ± {std_auc:.4f}")
    
    # 标定特征（0.8889）来自主实验，此处不计算，仅打印
    print(f"本文标定特征（Top-3，检索个数10） AUC = 0.8889 （引用自主实验）")

# ================= 自由文本基线（UltraMedical 无 RAG，层15）=================
def free_baselines():
    print("\n" + "="*60)
    print("自由文本基线（UltraMedical 无 RAG）")
    print("="*60)
    
    MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
    TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    FREE_TEST_FILE = "./data/free_text_test.jsonl"
    
    # 加载 SAE（层15）
    def load_sae(layer):
        sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
        w = safetensors.torch.load_file(sae_path)
        return w['encoder.weight'].to(DEVICE).to(torch.float16), w['encoder.bias'].to(DEVICE).to(torch.float16)
    W_enc, b_enc = load_sae(15)
    def sae_encode(hidden):
        z = hidden @ W_enc.T + b_enc
        topk = torch.topk(z, 32, dim=-1)
        f = torch.zeros_like(z)
        f.scatter_(-1, topk.indices, topk.values)
        return torch.relu(f)
    
    # 加载标定好的特征权重
    calib_file = os.path.join(CACHE_DIR, "free_weights.npy")
    if not os.path.exists(calib_file):
        raise FileNotFoundError("请先运行主实验生成标定文件 free_weights.npy")
    data = np.load(calib_file, allow_pickle=True).item()
    free_top_idx = data['top_idx'].tolist()
    free_weights = data['top_w'].tolist()
    
    # 加载测试集
    with open(FREE_TEST_FILE, 'r') as f:
        test = [json.loads(line) for line in f]
    test_labels = np.array([1 - s['label'] for s in test])
    
    # 1. 原始隐藏状态基线
    raw_hiddens = []
    for s in tqdm(test, desc="自由文本-提取原始隐藏状态"):
        prompt = f"在某一次体检中，我的{s['report']}，正常吗？"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
        raw_hiddens.append(hidden)
    raw_hiddens = np.array(raw_hiddens)
    l1_risks = np.sum(np.abs(raw_hiddens), axis=1)
    l2_risks = np.sqrt(np.sum(raw_hiddens**2, axis=1))
    std_risks = np.std(raw_hiddens, axis=1)
    auc_l1 = roc_auc_score(test_labels, l1_risks)
    auc_l2 = roc_auc_score(test_labels, l2_risks)
    auc_std = roc_auc_score(test_labels, std_risks)
    print(f"自由文本原始隐藏状态 L1 范数 AUC = {auc_l1:.4f}")
    print(f"自由文本原始隐藏状态 L2 范数 AUC = {auc_l2:.4f}")
    print(f"自由文本原始隐藏状态 标准差 AUC = {auc_std:.4f}")
    
    # 2. SAE 特征全等权 L1 范数（无标定）
    test_feats = []
    for s in tqdm(test, desc="自由文本-提取 SAE 特征（层15）"):
        prompt = f"在某一次体检中，我的{s['report']}，正常吗？"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[15][0, -1, :]
        feat = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
        test_feats.append(feat)
    test_feats = np.array(test_feats)
    sae_l1_risks = np.sum(np.abs(test_feats), axis=1)
    auc_sae_l1 = roc_auc_score(test_labels, sae_l1_risks)
    print(f"自由文本 SAE 特征全等权 L1 范数（无标定） AUC = {auc_sae_l1:.4f}")
    
    # 3. 随机特征（4个，10次平均）
    n_features = 4
    n_iters = 10
    random_aucs = []
    for _ in range(n_iters):
        rand_idx = np.random.choice(test_feats.shape[1], size=n_features, replace=False)
        rand_weights = np.random.randn(n_features)
        risks = np.sum(test_feats[:, rand_idx] * rand_weights, axis=1)
        auc = roc_auc_score(test_labels, risks)
        random_aucs.append(auc)
    avg_auc = np.mean(random_aucs)
    std_auc = np.std(random_aucs)
    print(f"自由文本随机特征（{n_features} 个，{n_iters} 次）AUC = {avg_auc:.4f} ± {std_auc:.4f}")
    
    print(f"本文标定特征（Top-4） AUC = 0.6335 （引用自主实验）")

def multitoken_exploration():
    print("\n" + "="*60)
    print("多 token 特征探索")
    print("="*60)
    
    MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
    TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    SAE_LAYER = 15
    SAE_TOP_K = 32
    FREE_FEATURE_COUNT = 4
    FREE_TEST_FILE = "./data/free_text_test.jsonl"
    
    def load_sae(layer):
        sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
        w = safetensors.torch.load_file(sae_path)
        return w['encoder.weight'].to(DEVICE).to(torch.float16), w['encoder.bias'].to(DEVICE).to(torch.float16)
    W_enc, b_enc = load_sae(SAE_LAYER)
    
    def sae_encode(hidden):
        z = hidden @ W_enc.T + b_enc
        topk = torch.topk(z, SAE_TOP_K, dim=-1)
        f = torch.zeros_like(z)
        f.scatter_(-1, topk.indices, topk.values)
        return torch.relu(f)
    
    # 加载标定权重
    calib_file = os.path.join(CACHE_DIR, "free_weights.npy")
    data = np.load(calib_file, allow_pickle=True).item()
    free_top_idx = data['top_idx'].tolist()
    free_weights = data['top_w'].tolist()
    
    with open(FREE_TEST_FILE, 'r') as f:
        test = [json.loads(line) for line in f]
    test_labels = np.array([1 - s['label'] for s in test])
    
    MAX_TOKENS = 10
    all_feats = []
    for s in tqdm(test, desc="提取多 token 特征"):
        prompt = f"在某一次体检中，我的{s['report']}，正常吗？"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_seq = outputs.hidden_states[SAE_LAYER][0, -MAX_TOKENS:, :]
        feats = []
        for h in hidden_seq:
            f = sae_encode(h.unsqueeze(0)).squeeze(0).cpu().numpy()
            feats.append(f)
        all_feats.append(np.array(feats))
    
    # ================= 1. 各 token 位置 AUC =================
    print("\n=== 各 token 位置的 AUC ===")
    for d in range(1, MAX_TOKENS+1):
        risks = []
        for feats in all_feats:
            if len(feats) < d:
                idx = -1
            else:
                idx = -d
            feat = feats[idx]
            risk = np.sum(feat[free_top_idx] * free_weights)
            risks.append(risk)
        auc = roc_auc_score(test_labels, risks)
        print(f"偏移量 -{d} (倒数第{d}个token): AUC = {auc:.4f}")
    
    # ================= 2. 基础多 token 组合 =================
    def combine_mean(feats, n):
        return np.mean(feats[-n:], axis=0)
    def combine_max(feats, n):
        return np.max(feats[-n:], axis=0)
    def combine_weighted(feats, n, decay=0.8):
        weights = np.array([decay ** (n-1-i) for i in range(n)])
        weights = weights / weights.sum()
        return np.sum(feats[-n:] * weights[:, np.newaxis], axis=0)
    
    print("\n=== 多 token 基础组合 AUC ===")
    for n in [2,3,4,5,6,7,8,9,10]:
        for name, func in [("mean", combine_mean), ("max", combine_max), ("weighted", combine_weighted)]:
            risks = []
            for feats in all_feats:
                if len(feats) < n:
                    feat = feats[-1]
                else:
                    feat = func(feats, n)
                risk = np.sum(feat[free_top_idx] * free_weights)
                risks.append(risk)
            auc = roc_auc_score(test_labels, risks)
            print(f"n={n}, {name}: AUC={auc:.4f}")
    
    # ================= 3. 更多组合探索 =================
    def combine_linear_decay(feats, n):
        weights = np.arange(1, n+1)
        weights = weights / weights.sum()
        return np.sum(feats[-n:] * weights[:, np.newaxis], axis=0)
    
    def combine_square_decay(feats, n):
        weights = np.arange(1, n+1)**2
        weights = weights / weights.sum()
        return np.sum(feats[-n:] * weights[:, np.newaxis], axis=0)
    
    def combine_l2_weighted(feats, n):
        block = feats[-n:]
        l2_norms = np.linalg.norm(block, axis=1)
        weights = l2_norms / (l2_norms.sum() + 1e-8)
        return np.sum(block * weights[:, np.newaxis], axis=0)
    
    def combine_max_skip(feats, positions):
        selected = [feats[p] for p in positions if -p <= len(feats)]
        return np.max(selected, axis=0)
    
    print("\n=== 更多组合探索 ===")
    # 3.1 不同衰减系数
    for decay in [0.6, 0.7, 0.8, 0.9]:
        for n in [3,4,5,6,7,8,9,10]:
            risks = []
            for feats in all_feats:
                if len(feats) < n:
                    feat = feats[-1]
                else:
                    weights = np.array([decay ** (n-1-i) for i in range(n)])
                    weights = weights / weights.sum()
                    feat = np.sum(feats[-n:] * weights[:, np.newaxis], axis=0)
                risk = np.sum(feat[free_top_idx] * free_weights)
                risks.append(risk)
            auc = roc_auc_score(test_labels, risks)
            print(f"n={n}, decay={decay}: AUC={auc:.4f}")
    
    # 3.2 线性衰减 & 平方衰减
    for n in [3,4,5,6,7,8,9,10]:
        for func, name in [(combine_linear_decay, 'linear'), (combine_square_decay, 'square')]:
            risks = []
            for feats in all_feats:
                if len(feats) < n:
                    feat = feats[-1]
                else:
                    feat = func(feats, n)
                risk = np.sum(feat[free_top_idx] * free_weights)
                risks.append(risk)
            auc = roc_auc_score(test_labels, risks)
            print(f"n={n}, {name}: AUC={auc:.4f}")
    
    # 3.3 L2 范数加权
    for n in [3,4,5,6,7,8,9,10]:
        risks = []
        for feats in all_feats:
            if len(feats) < n:
                feat = feats[-1]
            else:
                feat = combine_l2_weighted(feats, n)
            risk = np.sum(feat[free_top_idx] * free_weights)
            risks.append(risk)
        auc = roc_auc_score(test_labels, risks)
        print(f"n={n}, L2_weighted: AUC={auc:.4f}")
    
    # 3.4 选择性位置最大值
    skip_patterns = [[-1,-2], [-1,-3], [-1,-4], [-1,-2,-3], [-1,-3,-5], [-1,-2,-4]]
    for pattern in skip_patterns:
        risks = []
        for feats in all_feats:
            valid = [p for p in pattern if -p <= len(feats)]
            if not valid:
                feat = feats[-1]
            else:
                selected = [feats[p] for p in valid]
                feat = np.max(selected, axis=0)
            risk = np.sum(feat[free_top_idx] * free_weights)
            risks.append(risk)
        auc = roc_auc_score(test_labels, risks)
        print(f"positions={pattern}: AUC={auc:.4f}")
    
    # ================= 4. 新思路探索 =================
    def safe_ratio(feats, n=5):
        block = feats[-n:]
        ratio = np.mean(np.mean(block > 0.5, axis=1))
        return ratio
    
    def trajectory_variance(feats, n=5):
        block = feats[-n:]
        return np.mean(np.std(block, axis=0))
    
    def l2_weighted_pooling(feats, n=5):
        block = feats[-n:]
        weights = np.linalg.norm(block, axis=1)
        weights = weights / (weights.sum() + 1e-8)
        weighted_avg = np.sum(block * weights[:, np.newaxis], axis=0)
        return weighted_avg
    
    def cosine_similarity_variance(feats, n=5):
        block = feats[-n:]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        normalized = block / (norms + 1e-8)
        sim_matrix = normalized @ normalized.T
        triu_indices = np.triu_indices(n, k=1)
        return np.mean(sim_matrix[triu_indices])
    
    print("\n=== 新思路探索 ===")
    for n in [3, 4, 5, 6, 8, 10]:
        # 4.1 轨迹方差
        risks = []
        for feats in all_feats:
            if len(feats) < n:
                feat_val = np.mean(np.std(feats, axis=0))
            else:
                feat_val = trajectory_variance(feats, n)
            risks.append(feat_val)
        auc = roc_auc_score(test_labels, risks)
        print(f"n={n}, trajectory_variance: AUC={auc:.4f}")
    
        # 4.2 L2 加权池化（先池化，再计算加权和）
        risks = []
        for feats in all_feats:
            if len(feats) < n:
                weights = np.linalg.norm(feats, axis=1)
                weights = weights / (weights.sum() + 1e-8)
                weighted_avg = np.sum(feats * weights[:, np.newaxis], axis=0)
            else:
                weighted_avg = l2_weighted_pooling(feats, n)
            risk = np.sum(weighted_avg[free_top_idx] * free_weights)
            risks.append(risk)
        auc = roc_auc_score(test_labels, risks)
        print(f"n={n}, l2_weighted_pooling: AUC={auc:.4f}")
    
        # 4.3 安全比例
        risks = []
        for feats in all_feats:
            if len(feats) < n:
                feat_val = np.mean(np.mean(feats > 0.5, axis=1))
            else:
                feat_val = safe_ratio(feats, n)
            risks.append(feat_val)
        auc = roc_auc_score(test_labels, risks)
        print(f"n={n}, safe_ratio: AUC={auc:.4f}")
    
        # 4.4 多样性（余弦相似度）
        risks = []
        for feats in all_feats:
            if len(feats) < n:
                feat_val = 1 - cosine_similarity_variance(feats, len(feats))
            else:
                feat_val = 1 - cosine_similarity_variance(feats, n)
            risks.append(feat_val)
        auc = roc_auc_score(test_labels, risks)
        print(f"n={n}, diversity: AUC={auc:.4f}")
    
    print("\n多 token 特征探索全部完成。")
# ================= 主函数 =================
if __name__ == "__main__":
    # 1. 选择题基线
    choice_baselines()
    # 2. 自由文本基线
    free_baselines()
    # 3. 多 token 特征探索
    multitoken_exploration()
    print("\n所有补充实验完成！")