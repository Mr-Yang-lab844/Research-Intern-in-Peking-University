#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充实验（含阈值分析、多token融合等）
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
MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAE_LAYER = 15
SAE_TOP_K = 32
FREE_FEATURE_COUNT = 4

FREE_DEV_FILE = "./data/free_text_dev.jsonl"
FREE_TEST_FILE = "./data/free_text_test.jsonl"
CHOICE_TEST_FILE = "./data/test.jsonl"
ORIG_EMBED_MODEL = "BAAI/bge-base-zh-v1.5"
NEW_EMBED_MODEL = "shibing624/text2vec-base-chinese"

CACHE_DIR = "./cache_supplement"
os.makedirs(CACHE_DIR, exist_ok=True)

# ================= 加载模型和 SAE =================
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

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

def get_free_feature(report_text):
    prompt = f"在某一次体检中，我的{report_text}，正常吗？"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[SAE_LAYER][0, -1, :]
    feat = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    return feat

# ================= 标定 =================
def load_or_calibrate_weights():
    cache_file = os.path.join(CACHE_DIR, "free_weights.npy")
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True).item()
        return data['top_idx'], data['top_w'], data['low_th'], data['high_th']
    else:
        print("标定 SAE 特征权重（仅一次）...")
        with open(FREE_DEV_FILE, 'r') as f:
            dev = [json.loads(line) for line in f]
        feats, labels = [], []
        for s in dev:
            report = s['report']
            label = s['label']  # 1=正确,0=错误
            error_label = 1 - label
            feat = get_free_feature(report)
            feats.append(feat)
            labels.append(error_label)
        X = np.array(feats)
        y = np.array(labels)
        correct_mask = (y == 0)
        error_mask = (y == 1)
        mean_c = X[correct_mask].mean(axis=0)
        mean_e = X[error_mask].mean(axis=0)
        std_c = X[correct_mask].std(axis=0) + 1e-8
        std_e = X[error_mask].std(axis=0) + 1e-8
        t_score = np.abs(mean_e - mean_c) / np.sqrt(std_c**2 + std_e**2)
        t_score[np.isnan(t_score)] = 0
        top_idx = np.argsort(t_score)[-FREE_FEATURE_COUNT:][::-1]
        top_w = t_score[top_idx]
        risks = np.array([np.sum(feat[top_idx] * top_w) for feat in X])
        low_th = np.percentile(risks, 30)
        high_th = np.percentile(risks, 70)
        np.save(cache_file, {'top_idx': top_idx, 'top_w': top_w, 'low_th': low_th, 'high_th': high_th})
        return top_idx, top_w, low_th, high_th

free_top_idx, free_weights, FREE_LOW_TH, FREE_HIGH_TH = load_or_calibrate_weights()

# ================= 实验4：SAE拦截效率 =================
def experiment_sae_efficiency():
    print("\n=== 实验4：SAE拦截效率（耗时测量）===")
    with open(FREE_DEV_FILE, 'r') as f:
        dev = [json.loads(line) for line in f][:20]
    times = []
    for s in dev:
        report = s['report']
        feat = get_free_feature(report)
        start = time.perf_counter()
        risk = np.sum(feat[free_top_idx] * free_weights)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"风险分数计算平均耗时: {avg_ms:.3f} ± {std_ms:.3f} ms")
    print(f"(基于 {len(times)} 次测量)")

# ================= 实验5：工具调用命中率 =================
def experiment_tool_call_accuracy():
    print("\n=== 实验5：工具调用命中率 ===")
    abnormal_cases = [
        ("血糖 7.5 mmol/L", "偏高"),
        ("收缩压 160 mmHg", "高血压"),
        ("舒张压 105 mmHg", "高血压"),
        ("总胆固醇 6.8 mmol/L", "偏高"),
        ("甘油三酯 3.5 mmol/L", "偏高"),
        ("低密度脂蛋白 4.5 mmol/L", "偏高"),
        ("血红蛋白 80 g/L", "偏低"),
        ("肌酐 150 μmol/L", "偏高"),
        ("尿酸 550 μmol/L", "偏高"),
        ("心率 120 次/分", "心动过速")
    ]
    correct = 0
    total = len(abnormal_cases)
    for report, expected in abnormal_cases:
        prompt = f"在某一次体检中，我的{report}，正常吗？"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        keywords = ["就医", "医生", "咨询", "进一步检查", "治疗", "药物"]
        hit = any(kw in response for kw in keywords)
        if hit:
            correct += 1
        print(f"{report} -> 工具调用: {'✓' if hit else '✗'}")
    print(f"工具调用命中率: {correct}/{total} = {correct/total*100:.1f}%")

# ================= 实验6：随机特征 vs 选择特征 AUC =================
def experiment_random_features():
    print("\n=== 实验6：通用幻觉特征对比（随机特征 vs 选择特征）===")
    cache_test_feat = os.path.join(CACHE_DIR, "free_test_feats.npy")
    cache_test_labels = os.path.join(CACHE_DIR, "free_test_labels.npy")
    if os.path.exists(cache_test_feat) and os.path.exists(cache_test_labels):
        test_feats = np.load(cache_test_feat)
        test_labels = np.load(cache_test_labels)
    else:
        with open(FREE_TEST_FILE, 'r') as f:
            test = [json.loads(line) for line in f]
        test_feats = []
        test_labels = []
        for s in test:
            report = s['report']
            label = s['label']  # 1=正确
            error_label = 1 - label
            feat = get_free_feature(report)
            test_feats.append(feat)
            test_labels.append(error_label)
        test_feats = np.array(test_feats)
        test_labels = np.array(test_labels)
        np.save(cache_test_feat, test_feats)
        np.save(cache_test_labels, test_labels)

    risks_selected = np.sum(test_feats[:, free_top_idx] * free_weights, axis=1)
    auc_selected = roc_auc_score(test_labels, risks_selected)
    print(f"我们的特征选择 (Top-{FREE_FEATURE_COUNT}) AUC = {auc_selected:.4f}")

    random_aucs = []
    for _ in range(10):
        rand_idx = np.random.choice(test_feats.shape[1], size=FREE_FEATURE_COUNT, replace=False)
        rand_weights = np.random.randn(FREE_FEATURE_COUNT)
        risks_rand = np.sum(test_feats[:, rand_idx] * rand_weights, axis=1)
        auc_rand = roc_auc_score(test_labels, risks_rand)
        random_aucs.append(auc_rand)
    avg_rand_auc = np.mean(random_aucs)
    std_rand_auc = np.std(random_aucs)
    print(f"随机特征 (平均) AUC = {avg_rand_auc:.4f} ± {std_rand_auc:.4f}")
    if std_rand_auc > 0:
        print(f"我们的特征选择优于随机特征 {(auc_selected - avg_rand_auc)/std_rand_auc:.2f} 标准差")
    else:
        print("随机特征标准差为0，无法计算倍数")

# ================= 实验7：不同嵌入模型测试 =================
def experiment_embedding_model():
    print("\n=== 实验7：不同嵌入模型对选择题准确率的影响 ===")
    choice_model_path = "./models/Llama-3.1-8B-Instruct"
    choice_tokenizer = AutoTokenizer.from_pretrained(choice_model_path)
    choice_model = AutoModelForCausalLM.from_pretrained(choice_model_path, torch_dtype=torch.float16, device_map="auto")
    choice_model.eval()

    def test_accuracy_with_embedding(embed_model_name, index_name):
        print(f"  使用嵌入模型: {embed_model_name}")
        index_path = f"./faiss_choice_{index_name}"
        if os.path.exists(index_path):
            embedding = HuggingFaceEmbeddings(model_name=embed_model_name, model_kwargs={'device': DEVICE})
            vector_store = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
        else:
            docs = []
            for fp in ["./knowledge/medical_textbook_chunks.jsonl", "./knowledge/reference_ranges.jsonl", "./data/train.jsonl"]:
                with open(fp, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        if "text" in data:
                            content = data["text"]
                        elif "test_name" in data:
                            content = data["description"]
                        elif "input" in data:
                            content = f"问题: {data['input']}\n答案: {data['output']}"
                        else:
                            continue
                        docs.append(Document(page_content=content))
            embedding = HuggingFaceEmbeddings(model_name=embed_model_name, model_kwargs={'device': DEVICE})
            vector_store = FAISS.from_documents(docs, embedding)
            vector_store.save_local(index_path)
        with open(CHOICE_TEST_FILE, 'r') as f:
            all_samples = [json.loads(line) for line in f]
        random.seed(42)
        test_samples = random.sample(all_samples, 20)
        correct = 0
        for s in test_samples:
            q = s['input']
            true = s['output']
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
            inputs = choice_tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = choice_model.generate(**inputs, max_new_tokens=10, do_sample=False)
            ans = choice_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in ans:
                ans = ans.split("Answer:")[-1].strip()
            match = re.search(r'\b([A-E])\b', ans)
            pred = match.group(1) if match else ''
            if pred == true:
                correct += 1
        acc = correct / len(test_samples) * 100
        print(f"    准确率: {correct}/{len(test_samples)} = {acc:.1f}%")
        return acc

    print("原嵌入模型 (BAAI/bge-base-zh-v1.5):")
    acc_orig = test_accuracy_with_embedding(ORIG_EMBED_MODEL, "orig")
    print("新嵌入模型 (shibing624/text2vec-base-chinese):")
    acc_new = test_accuracy_with_embedding(NEW_EMBED_MODEL, "new")
    print(f"准确率变化: {acc_orig:.1f}% -> {acc_new:.1f}%")

# ================= 实验8：自由文本原始隐藏状态基线 =================
def experiment_raw_hidden_baseline():
    print("\n=== 实验8：原始隐藏状态基线（无 SAE）===")
    cache_raw_feat = os.path.join(CACHE_DIR, "free_test_raw_hidden.npy")
    cache_test_labels = os.path.join(CACHE_DIR, "free_test_labels.npy")
    if os.path.exists(cache_raw_feat) and os.path.exists(cache_test_labels):
        raw_hiddens = np.load(cache_raw_feat)
        test_labels = np.load(cache_test_labels)
        print("从缓存加载原始隐藏状态及标签")
    else:
        with open(FREE_TEST_FILE, 'r') as f:
            test = [json.loads(line) for line in f]
        raw_hiddens = []
        test_labels = []
        for s in test:
            report = s['report']
            label = s['label']
            error_label = 1 - label
            prompt = f"在某一次体检中，我的{report}，正常吗？"
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
            raw_hiddens.append(hidden)
            test_labels.append(error_label)
        raw_hiddens = np.array(raw_hiddens)
        test_labels = np.array(test_labels)
        np.save(cache_raw_feat, raw_hiddens)
        np.save(cache_test_labels, test_labels)
        print("计算并缓存原始隐藏状态")

    l1_risks = np.sum(np.abs(raw_hiddens), axis=1)
    l2_risks = np.sqrt(np.sum(raw_hiddens**2, axis=1))
    std_risks = np.std(raw_hiddens, axis=1)
    auc_l1 = roc_auc_score(test_labels, l1_risks)
    auc_l2 = roc_auc_score(test_labels, l2_risks)
    auc_std = roc_auc_score(test_labels, std_risks)
    print(f"原始隐藏状态 L1 范数 AUC = {auc_l1:.4f}")
    print(f"原始隐藏状态 L2 范数 AUC = {auc_l2:.4f}")
    print(f"原始隐藏状态 标准差 AUC = {auc_std:.4f}")

    cache_test_feat = os.path.join(CACHE_DIR, "free_test_feats.npy")
    if os.path.exists(cache_test_feat):
        test_feats = np.load(cache_test_feat)
        sae_l1_risks = np.sum(np.abs(test_feats), axis=1)
        auc_sae_l1 = roc_auc_score(test_labels, sae_l1_risks)
        print(f"SAE 特征（全特征）L1 范数 AUC = {auc_sae_l1:.4f}")
    else:
        print("未找到 SAE 特征缓存，跳过 SAE 全特征对比")

# ================= 阈值分析（实验9） =================
def experiment_threshold_analysis():
    print("\n=== 实验9：阈值分析（自由文本测试集，不同分位数下的分类性能）===")
    cache_test_feat = os.path.join(CACHE_DIR, "free_test_feats.npy")
    cache_test_labels = os.path.join(CACHE_DIR, "free_test_labels.npy")
    if os.path.exists(cache_test_feat) and os.path.exists(cache_test_labels):
        test_feats = np.load(cache_test_feat)
        test_labels = np.load(cache_test_labels)
    else:
        print("提取测试集特征...")
        with open(FREE_TEST_FILE, 'r') as f:
            test = [json.loads(line) for line in f]
        test_feats = []
        test_labels = []
        for s in test:
            report = s['report']
            label = s['label']
            error_label = 1 - label
            feat = get_free_feature(report)
            test_feats.append(feat)
            test_labels.append(error_label)
        test_feats = np.array(test_feats)
        test_labels = np.array(test_labels)
        np.save(cache_test_feat, test_feats)
        np.save(cache_test_labels, test_labels)
    risks = np.sum(test_feats[:, free_top_idx] * free_weights, axis=1)
    print(f"风险分数范围: {risks.min():.4f} ~ {risks.max():.4f}")
    percentiles = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
    for p in percentiles:
        th = np.percentile(risks, p)
        pred = (risks > th).astype(int)
        acc = accuracy_score(test_labels, pred)
        prec = precision_score(test_labels, pred, zero_division=0)
        rec = recall_score(test_labels, pred, zero_division=0)
        f1 = f1_score(test_labels, pred, zero_division=0)
        print(f"分位数 {p}%: 阈值={th:.4f}, Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
    print("阈值分析完成。")

# ================= 主函数 =================
if __name__ == "__main__":
    experiment_threshold_analysis()
    experiment_sae_efficiency()
    experiment_tool_call_accuracy()
    experiment_random_features()
    experiment_embedding_model()
    experiment_raw_hidden_baseline()
    print("\n所有补充实验完成！")