#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一体化 RAG + SAE Pipeline (改进版)
- 提取模型开始生成前的最后一个 token 隐藏状态
- 使用逻辑回归组合特征 (Top-30)
- 支持 --layer 参数
"""

import json
import argparse
import os
import re
import numpy as np
import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, fbeta_score, precision_score, recall_score
from tqdm import tqdm

# ================= 固定配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE = "./data/free_text_test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOP_K_RETRIEVAL = 3
SAE_TOP_K = 32
TOP_FEATURES = 30          # 逻辑回归使用的特征数量
TARGET_LAYER = 20          # 会被命令行覆盖
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 辅助函数 =================
def load_knowledge_base():
    """加载教科书和参考范围，构建 FAISS 索引（缓存）"""
    index_path = "./faiss_merged_index"
    if os.path.exists(index_path):
        print(f"加载已有索引: {index_path}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        return vector_store
    else:
        print("构建知识库索引...")
        docs = []
        with open(TEXTBOOK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                docs.append(Document(page_content=data["text"], metadata={"source": "textbook"}))
        with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                content = f"{data['test_name']}：正常范围 {data['normal_range']} {data['unit']}。{data['description']}"
                docs.append(Document(page_content=content, metadata={"source": "reference"}))
        print(f"共加载 {len(docs)} 个文档")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(index_path)
        print(f"索引已保存至 {index_path}")
        return vector_store

def load_sae(layer):
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE 权重文件不存在: {sae_path}")
    sae_weights = safetensors.torch.load_file(sae_path)
    W_enc = sae_weights['encoder.weight'].to(DEVICE).to(torch.float16)
    b_enc = sae_weights['encoder.bias'].to(DEVICE).to(torch.float16)
    return W_enc, b_enc

def sae_encode(hidden, W_enc, b_enc):
    z = hidden @ W_enc.T + b_enc
    topk_vals, topk_idx = torch.topk(z, SAE_TOP_K, dim=-1)
    features = torch.zeros_like(z)
    features.scatter_(-1, topk_idx, topk_vals)
    features = torch.relu(features)
    return features

def get_pre_generation_feature(report_text, knowledge_context, model, tokenizer, W_enc, b_enc):
    """提取模型开始生成解读前的最后一个 token 的隐藏状态"""
    prompt = f"检验报告：{report_text}\n\n知识：{knowledge_context}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[TARGET_LAYER][0, -1, :]   # 最后一个 token
    features = sae_encode(hidden.unsqueeze(0), W_enc, b_enc).squeeze(0).cpu().numpy()
    return features

def generate_interpretation(report_text, knowledge_context, model, tokenizer):
    prompt = f"你是医学专家。根据以下医学知识，对检验结果给出简短解读（1-2句话）并给出建议。\n\n知识：{knowledge_context}\n\n检验报告：{report_text}\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "解读：" in answer:
        answer = answer.split("解读：")[-1].strip()
    return answer

# ================= 主流程 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, choices=[5,10,15,20,25,30], help="SAE 层数")
    args = parser.parse_args()
    global TARGET_LAYER
    TARGET_LAYER = args.layer

    print(f"开始 Pipeline (SAE 层 {TARGET_LAYER})")

    # 1. 加载模型和 tokenizer
    print("加载 LLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    # 2. 加载 SAE
    print("加载 SAE...")
    W_enc, b_enc = load_sae(TARGET_LAYER)

    # 3. 加载知识库索引
    vector_store = load_knowledge_base()

    # 4. 加载开发集，提取特征（使用知识库检索）
    print("加载开发集...")
    dev_samples = []
    with open(DEV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            dev_samples.append(json.loads(line))
    print(f"开发集样本数: {len(dev_samples)}")

    dev_features = []
    dev_labels = []
    for sample in tqdm(dev_samples, desc="提取开发集特征"):
        report = sample["report"]
        # 为开发集样本检索知识（使用原始 report 作为查询）
        retrieved = vector_store.similarity_search(report, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        feat = get_pre_generation_feature(report, knowledge, model, tokenizer, W_enc, b_enc)
        dev_features.append(feat)
        dev_labels.append(sample["label"])

    X_dev = np.array(dev_features)
    y_dev = np.array(dev_labels)

    # 特征选择：根据开发集上的区分度（t-statistic）选择 Top‑K 特征
    correct_mask = (y_dev == 1)
    error_mask = (y_dev == 0)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        raise ValueError("开发集必须同时包含正确(label=1)和错误(label=0)样本")

    mean_correct = X_dev[correct_mask].mean(axis=0)
    mean_error = X_dev[error_mask].mean(axis=0)
    std_correct = X_dev[correct_mask].std(axis=0) + 1e-8
    std_error = X_dev[error_mask].std(axis=0) + 1e-8
    pooled_std = np.sqrt(std_correct**2 + std_error**2)
    t_score = np.abs(mean_error - mean_correct) / pooled_std
    t_score[np.isnan(t_score)] = 0

    top_k = TOP_FEATURES
    top_indices = np.argsort(t_score)[-top_k:][::-1]
    top_scores = t_score[top_indices]
    print(f"Top {top_k} 特征索引及得分:")
    for idx, sc in zip(top_indices[:10], top_scores[:10]):
        print(f"  {idx}: {sc:.4f}")

    # 使用选出的特征训练逻辑回归分类器（预测是否为错误样本）
    X_dev_selected = X_dev[:, top_indices]
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_dev_selected, 1 - y_dev)   # 目标：1 表示错误（因为 risk 越高越可能错误）
    # 保存模型和特征索引
    os.makedirs(f"./results/layer{TARGET_LAYER}", exist_ok=True)
    feature_info = {
        "layer": TARGET_LAYER,
        "top_indices": top_indices.tolist(),
        "coefficients": clf.coef_[0].tolist(),
        "intercept": clf.intercept_[0].tolist()
    }
    with open(f"./results/layer{TARGET_LAYER}/features.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    # 5. 加载测试集并评估
    print("加载测试集...")
    test_samples = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))
    print(f"测试集样本数: {len(test_samples)}")

    test_risks = []
    test_true_error = []   # 1 表示错误，0 表示正确
    test_generated = []

    for sample in tqdm(test_samples, desc="评估测试集"):
        report = sample["report"]
        true_label = sample["label"]   # 1 正确，0 错误
        # 检索知识
        retrieved = vector_store.similarity_search(report, k=TOP_K_RETRIEVAL)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        # 生成解读（用于展示，但不影响特征提取）
        interpretation = generate_interpretation(report, knowledge, model, tokenizer)
        # 提取特征（使用生成前的 prompt）
        feat = get_pre_generation_feature(report, knowledge, model, tokenizer, W_enc, b_enc)
        X_test_selected = feat[top_indices].reshape(1, -1)
        risk = clf.predict_proba(X_test_selected)[0, 1]   # 预测为错误样本的概率
        test_risks.append(risk)
        test_true_error.append(1 - true_label)   # 转换为 1=错误
        test_generated.append(interpretation)

    test_risks = np.array(test_risks)
    y_true = np.array(test_true_error)

    # 打印前10个样本的 risk 和真实标签
    print("\n前10个测试样本的风险分数及真实标签（1=错误）:")
    for i in range(min(10, len(test_risks))):
        print(f"  risk={test_risks[i]:.4f}, true_error={y_true[i]}")

    auc = roc_auc_score(y_true, test_risks)
    print(f"\n=== 评估结果 (层 {TARGET_LAYER}) ===")
    print(f"AUC: {auc:.4f}")

    # 阈值优化 (F2)
    thresholds = np.linspace(test_risks.min(), test_risks.max(), 101)
    best_f2 = 0
    best_th = 0
    for th in thresholds:
        pred = (test_risks >= th).astype(int)
        f2 = fbeta_score(y_true, pred, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_th = th

    pred_opt = (test_risks >= best_th).astype(int)
    precision = precision_score(y_true, pred_opt)
    recall = recall_score(y_true, pred_opt)
    print(f"最优阈值: {best_th:.4f}")
    print(f"F2 分数: {best_f2:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")

    # 保存评估结果
    results = {
        "layer": TARGET_LAYER,
        "auc": auc,
        "best_threshold": best_th,
        "f2": best_f2,
        "precision": precision,
        "recall": recall,
        "risk_scores": test_risks.tolist(),
        "true_labels": y_true.tolist(),
        "generated_interpretations": test_generated
    }
    with open(f"./results/layer{TARGET_LAYER}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("结果已保存至 ./results/")

if __name__ == "__main__":
    main()