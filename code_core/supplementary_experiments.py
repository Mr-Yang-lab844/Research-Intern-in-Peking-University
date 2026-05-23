#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充实验：自动完成计划书中的 4-7 项指标计算
- 4. SAE拦截效率：测量风险分数计算的平均耗时
- 5. 工具调用命中率：检测模型在高风险指标时是否建议就医
- 6. 通用幻觉特征对比：随机特征 vs 我们的特征选择 AUC
- 7. 不同嵌入模型测试：更换嵌入模型对选择题准确率的影响
"""

import json
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.metrics import roc_auc_score
import os
import random

# ================= 配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAE 配置（来自最终演示：层15，特征数4）
SAE_LAYER = 15
SAE_TOP_K = 32
FREE_FEATURE_COUNT = 4

# 文件路径
FREE_DEV_FILE = "./data/free_text_dev.jsonl"
FREE_TEST_FILE = "./data/free_text_test.jsonl"
CHOICE_TEST_FILE = "./data/test.jsonl"
CHOICE_INDEX_PATH = "./faiss_choice_index"   # 选择题原知识库索引（教科书+参考范围+同源训练集）
ORIG_EMBED_MODEL = "BAAI/bge-base-zh-v1.5"
NEW_EMBED_MODEL = "shibing624/text2vec-base-chinese"  # 用于测试不同嵌入模型

# 缓存特征（避免重复计算）
CACHE_DIR = "./cache_supplement"
os.makedirs(CACHE_DIR, exist_ok=True)

# ================= 加载模型和 SAE =================
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

def load_sae(layer):
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
    import safetensors.torch
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
    """提取无 RAG 自由文本特征（使用自然语言 prompt）"""
    prompt = f"在某一次体检中，我的{report_text}，正常吗？"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[SAE_LAYER][0, -1, :]
    feat = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    return feat

# 加载已标定的特征权重和阈值（从最终演示脚本中获取，这里假设已存在）
# 如果不存在，我们临时从开发集标定
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
        # 计算阈值（开发集风险分数的30%和70%分位数）
        risks = np.array([np.sum(feat[top_idx] * top_w) for feat in X])
        low_th = np.percentile(risks, 30)
        high_th = np.percentile(risks, 70)
        np.save(cache_file, {'top_idx': top_idx, 'top_w': top_w, 'low_th': low_th, 'high_th': high_th})
        return top_idx, top_w, low_th, high_th

free_top_idx, free_weights, FREE_LOW_TH, FREE_HIGH_TH = load_or_calibrate_weights()

# ================= 实验4：SAE拦截效率 =================
def experiment_sae_efficiency():
    print("\n=== 实验4：SAE拦截效率（耗时测量）===")
    # 使用开发集前20个样本，测量风险分数计算的平均时间
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
    # 定义需要工具的场景：指标明显异常时应建议就医
    # 我们构造10个异常指标输入，检查模型输出中是否包含“就医”、“咨询医生”等关键词
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
        # 检查是否包含工具关键词（就医、医生、咨询等）
        keywords = ["就医", "医生", "咨询", "进一步检查", "治疗", "药物"]
        hit = any(kw in response for kw in keywords)
        if hit:
            correct += 1
        print(f"{report} -> 工具调用: {'✓' if hit else '✗'}")
    print(f"工具调用命中率: {correct}/{total} = {correct/total*100:.1f}%")

# ================= 实验6：通用幻觉特征对比 =================
def experiment_random_features():
    print("\n=== 实验6：通用幻觉特征对比（随机特征 vs 选择特征）===")
    # 需要在自由文本测试集上计算随机特征集的AUC
    # 加载测试集特征（如果已缓存，否则提取）
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
    
    # 我们的特征选择 AUC
    risks_selected = np.sum(test_feats[:, free_top_idx] * free_weights, axis=1)
    auc_selected = roc_auc_score(test_labels, risks_selected)
    print(f"我们的特征选择 (Top-{FREE_FEATURE_COUNT}) AUC = {auc_selected:.4f}")
    
    # 随机特征：随机选取相同数量的特征索引，随机赋权（正态分布），计算平均AUC（10次）
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
    print(f"我们的特征选择优于随机特征 {(auc_selected - avg_rand_auc)/std_rand_auc:.2f} 标准差")

# ================= 实验7：不同嵌入模型测试 =================
def experiment_embedding_model():
    print("\n=== 实验7：不同嵌入模型对选择题准确率的影响 ===")
    # 使用原嵌入模型和新嵌入模型分别构建索引，测试选择题准确率（抽样20条）
    def test_accuracy_with_embedding(embed_model_name, index_name):
        print(f"  使用嵌入模型: {embed_model_name}")
        # 如果索引已存在，直接加载；否则构建
        index_path = f"./faiss_choice_{index_name}"
        if os.path.exists(index_path):
            embedding = HuggingFaceEmbeddings(model_name=embed_model_name, model_kwargs={'device': DEVICE})
            vector_store = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
        else:
            # 构建索引（仅第一次）
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
        # 抽样20个测试题
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
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in ans:
                ans = ans.split("Answer:")[-1].strip()
            import re
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

# ================= 运行所有补充实验 =================
if __name__ == "__main__":
    experiment_sae_efficiency()
    experiment_tool_call_accuracy()
    experiment_random_features()
    experiment_embedding_model()
    print("\n所有补充实验完成！")