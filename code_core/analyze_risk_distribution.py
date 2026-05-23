#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计选择题（TOP_K=20）和自由文本 SAE 风险分数的分布
- 选择题：层22，特征10，检索 Top-K=20，知识库=教科书+参考范围+同源训练集
- 自由文本：层16，特征11，检索 Top-K=3，知识库=教科书+参考范围
输出均值、方差、分位数等
"""

import os
import json
import random
import re
import numpy as np
import torch
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

# ================= 通用配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
SAE_TOP_K = 32

# 知识库文件（选择题包含同源训练集，自由文本只用教科书+参考范围）
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
TRAIN_FILE = "./data/train.jsonl"          # 同源训练集（选择题用）

# 加载模型
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# ----------------------------- 选择题专用知识库（包含同源训练集）-----------------------------
def load_choice_knowledge_base():
    index_path = "./faiss_choice_index"
    if os.path.exists(index_path):
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        return vector_store
    else:
        print("Building choice knowledge base (including train.jsonl)...")
        docs = []
        for file_path in [TEXTBOOK_FILE, REFERENCE_FILE, TRAIN_FILE]:
            print(f"  Loading {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if "text" in data:
                        content = data["text"]
                    elif "test_name" in data:
                        content = f"{data['test_name']}：正常范围 {data['normal_range']} {data['unit']}。{data['description']}"
                    elif "input" in data:
                        content = f"问题: {data['input']}\n答案: {data['output']}"
                    else:
                        continue
                    docs.append(Document(page_content=content, metadata={}))
        print(f"Total docs: {len(docs)}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(index_path)
        return vector_store

choice_vector_store = load_choice_knowledge_base()

# ----------------------------- 选择题部分（TOP_K=20）-----------------------------
print("\n=== 选择题风险分数统计 (TOP_K=20) ===")

CHOICE_LAYER = 22
CHOICE_FEATURE_COUNT = 10
CHOICE_TOP_K = 20               # 修改为20
RANDOM_SEED = 42
TOTAL_SAMPLES = 100
DEV_SIZE = 50

# 加载 SAE 权重
sae_path = f"./models/Llama-Scope/L{CHOICE_LAYER}R-8x.safetensors"
sae_weights = safetensors.torch.load_file(sae_path)
W_enc = sae_weights['encoder.weight'].to(DEVICE).to(torch.float16)
b_enc = sae_weights['encoder.bias'].to(DEVICE).to(torch.float16)

def sae_encode(hidden):
    z = hidden @ W_enc.T + b_enc
    topk = torch.topk(z, SAE_TOP_K, dim=-1)
    f = torch.zeros_like(z)
    f.scatter_(-1, topk.indices, topk.values)
    return torch.relu(f)

def get_choice_feature(question, knowledge):
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
    hidden = outputs.hidden_states[CHOICE_LAYER][0, -1, :]
    features = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    return features

def generate_choice_answer(question, knowledge):
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

# 加载测试集
with open("./data/test.jsonl", "r", encoding="utf-8") as f:
    all_samples = [json.loads(line) for line in f]
random.seed(RANDOM_SEED)
samples = random.sample(all_samples, TOTAL_SAMPLES)
dev_samples = samples[:DEV_SIZE]
test_samples = samples[DEV_SIZE:]

# 开发集特征提取
dev_features = []
dev_labels = []   # 1=错误
for s in tqdm(dev_samples, desc="Choice dev"):
    q = s["input"]
    true = s["output"]
    retrieved = choice_vector_store.similarity_search(q, k=CHOICE_TOP_K)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    pred = generate_choice_answer(q, knowledge)
    is_correct = (pred == true)
    feat = get_choice_feature(q, knowledge)
    dev_features.append(feat)
    dev_labels.append(0 if is_correct else 1)

dev_features = np.array(dev_features)
dev_labels = np.array(dev_labels)

# 特征选择
correct_mask = (dev_labels == 0)
error_mask = (dev_labels == 1)
mean_c = dev_features[correct_mask].mean(axis=0)
mean_e = dev_features[error_mask].mean(axis=0)
std_c = dev_features[correct_mask].std(axis=0) + 1e-8
std_e = dev_features[error_mask].std(axis=0) + 1e-8
pooled = np.sqrt(std_c**2 + std_e**2)
t_score = np.abs(mean_e - mean_c) / pooled
t_score[np.isnan(t_score)] = 0
top_idx = np.argsort(t_score)[-CHOICE_FEATURE_COUNT:][::-1]
top_weights = t_score[top_idx]

# 测试集风险分数
test_risks = []
test_labels = []   # 1=错误
for s in tqdm(test_samples, desc="Choice test"):
    q = s["input"]
    true = s["output"]
    retrieved = choice_vector_store.similarity_search(q, k=CHOICE_TOP_K)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    pred = generate_choice_answer(q, knowledge)
    is_correct = (pred == true)
    feat = get_choice_feature(q, knowledge)
    risk = np.sum(feat[top_idx] * top_weights)
    test_risks.append(risk)
    test_labels.append(0 if is_correct else 1)

test_risks = np.array(test_risks)
test_labels = np.array(test_labels)

print(f"Choice risk scores (n={len(test_risks)})")
print(f"  Mean: {test_risks.mean():.4f}")
print(f"  Variance: {test_risks.var():.4f}")
print(f"  Std: {test_risks.std():.4f}")
print(f"  Min: {test_risks.min():.4f}")
print(f"  Max: {test_risks.max():.4f}")
print("  Percentiles:")
for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
    print(f"    {p}%: {np.percentile(test_risks, p):.4f}")

# ================= 自由文本部分（保持不变，TOP_K=3）=================
print("\n=== 自由文本风险分数统计 ===")

FREE_LAYER = 16
FREE_FEATURE_COUNT = 11
FREE_TOP_K = 3
DEV_FILE = "./data/free_text_dev.jsonl"
TEST_FILE_FREE = "./data/free_text_test.jsonl"

# 自由文本知识库（仅教科书+参考范围）
def load_free_knowledge_base():
    index_path = "./faiss_merged_index"
    if os.path.exists(index_path):
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        return vector_store
    else:
        print("Building free knowledge base (textbook+reference)...")
        docs = []
        with open(TEXTBOOK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                docs.append(Document(page_content=data["text"], metadata={}))
        with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                content = f"{data['test_name']}：正常范围 {data['normal_range']} {data['unit']}。{data['description']}"
                docs.append(Document(page_content=content, metadata={}))
        print(f"Total docs: {len(docs)}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(index_path)
        return vector_store

free_vector_store = load_free_knowledge_base()

sae_path_free = f"./models/Llama-Scope/L{FREE_LAYER}R-8x.safetensors"
sae_weights_free = safetensors.torch.load_file(sae_path_free)
W_enc_free = sae_weights_free['encoder.weight'].to(DEVICE).to(torch.float16)
b_enc_free = sae_weights_free['encoder.bias'].to(DEVICE).to(torch.float16)

def get_free_feature(report, knowledge):
    prompt = f"检验报告：{report}\n\n知识：{knowledge}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[FREE_LAYER][0, -1, :]
    features = sae_encode_free(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    return features

def sae_encode_free(hidden):
    z = hidden @ W_enc_free.T + b_enc_free
    topk = torch.topk(z, SAE_TOP_K, dim=-1)
    f = torch.zeros_like(z)
    f.scatter_(-1, topk.indices, topk.values)
    return torch.relu(f)

dev_free = []
with open(DEV_FILE, "r", encoding="utf-8") as f:
    for line in f:
        dev_free.append(json.loads(line))
test_free = []
with open(TEST_FILE_FREE, "r", encoding="utf-8") as f:
    for line in f:
        test_free.append(json.loads(line))

dev_features_free = []
dev_labels_free = []
for s in tqdm(dev_free, desc="Free dev"):
    report = s["report"]
    true_label = s["label"]
    retrieved = free_vector_store.similarity_search(report, k=FREE_TOP_K)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    feat = get_free_feature(report, knowledge)
    dev_features_free.append(feat)
    dev_labels_free.append(0 if true_label == 1 else 1)

dev_features_free = np.array(dev_features_free)
dev_labels_free = np.array(dev_labels_free)

correct_mask_free = (dev_labels_free == 0)
error_mask_free = (dev_labels_free == 1)
if correct_mask_free.sum() > 0 and error_mask_free.sum() > 0:
    mean_c_free = dev_features_free[correct_mask_free].mean(axis=0)
    mean_e_free = dev_features_free[error_mask_free].mean(axis=0)
    std_c_free = dev_features_free[correct_mask_free].std(axis=0) + 1e-8
    std_e_free = dev_features_free[error_mask_free].std(axis=0) + 1e-8
    pooled_free = np.sqrt(std_c_free**2 + std_e_free**2)
    t_score_free = np.abs(mean_e_free - mean_c_free) / pooled_free
    t_score_free[np.isnan(t_score_free)] = 0
    top_idx_free = np.argsort(t_score_free)[-FREE_FEATURE_COUNT:][::-1]
    top_weights_free = t_score_free[top_idx_free]
else:
    print("Warning: free dev set lacks positive/negative samples, using default weights.")
    top_idx_free = [28843, 11033, 4945, 32434, 9495, 30755, 1483, 13022, 18938, 28720, 0]
    top_weights_free = [0.4675, 0.3923, 0.3853, 0.3308, 0.3303, 0.3206, 0.3184, 0.3162, 0.2998, 0.2786, 0.0]

test_risks_free = []
test_labels_free = []
for s in tqdm(test_free, desc="Free test"):
    report = s["report"]
    true_label = s["label"]
    retrieved = free_vector_store.similarity_search(report, k=FREE_TOP_K)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    feat = get_free_feature(report, knowledge)
    risk = np.sum(feat[top_idx_free] * top_weights_free)
    test_risks_free.append(risk)
    test_labels_free.append(0 if true_label == 1 else 1)

test_risks_free = np.array(test_risks_free)
test_labels_free = np.array(test_labels_free)

print(f"Free text risk scores (n={len(test_risks_free)})")
print(f"  Mean: {test_risks_free.mean():.4f}")
print(f"  Variance: {test_risks_free.var():.4f}")
print(f"  Std: {test_risks_free.std():.4f}")
print(f"  Min: {test_risks_free.min():.4f}")
print(f"  Max: {test_risks_free.max():.4f}")
print("  Percentiles:")
for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
    print(f"    {p}%: {np.percentile(test_risks_free, p):.4f}")