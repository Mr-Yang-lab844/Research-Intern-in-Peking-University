#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG + SAE (Layer 20) with full knowledge base.
Uses pre-selected top 10 features and their discrimination scores (from feature_analysis.npz)
to compute a risk score. Evaluates on the same test set (100 samples) for demonstration.
"""

import json
import torch
import re
import random
import numpy as np
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, fbeta_score, precision_score, recall_score

# ================= 配置 =================
KNOWLEDGE_FILE = "./data/train.jsonl"
TEST_FILE = "./data/test.jsonl"
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
SAE_WEIGHT_FILE = "./models/Llama-Scope/L20R-8x.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 3
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
MAX_NEW_TOKENS = 10
TEST_SIZE = 100          # 抽样数量（设为 None 则全测）
RANDOM_SEED = 42
SAE_TOP_K = 32
HIDDEN_DIM = 4096

# 从之前分析得到的最具区分度的特征索引及其 score（来自 feature_analysis.npz 的 top_indices 和 top_scores）
TOP_FEATURE_INDICES = [28843, 11033, 4945, 32434, 9495, 30755, 1483, 13022, 18938, 28720]
FEATURE_SCORES = [0.4675, 0.3923, 0.3853, 0.3308, 0.3303, 0.3206, 0.3184, 0.3162, 0.2998, 0.2786]

# ================= 加载 LLM =================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ================= 构建知识库 =================
print(f"Loading knowledge base from {KNOWLEDGE_FILE}...")
docs = []
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        question = data["input"]
        answer = data["output"]
        doc_text = f"问题: {question}\n答案: {answer}"
        docs.append(Document(page_content=doc_text, metadata={}))
print(f"Loaded {len(docs)} documents.")

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': DEVICE}
)
print("Building FAISS vector store...")
vector_store = FAISS.from_documents(docs, embedding_model)
print("Vector store ready.")

# ================= 加载 SAE =================
print(f"Loading SAE from {SAE_WEIGHT_FILE}...")
sae_weights = safetensors.torch.load_file(SAE_WEIGHT_FILE)
W_enc = sae_weights['encoder.weight'].to(DEVICE).to(torch.float16)
b_enc = sae_weights['encoder.bias'].to(DEVICE).to(torch.float16)
print(f"SAE: hidden_dim={HIDDEN_DIM}, feature_dim={W_enc.shape[0]}, TopK={SAE_TOP_K}")

def sae_encode(hidden_state: torch.Tensor) -> torch.Tensor:
    z = hidden_state @ W_enc.T + b_enc
    topk_values, topk_indices = torch.topk(z, SAE_TOP_K, dim=-1)
    features = torch.zeros_like(z)
    features.scatter_(-1, topk_indices, topk_values)
    features = torch.relu(features)
    return features

# ================= 辅助函数 =================
def extract_answer_letter(raw_output: str) -> str:
    match = re.search(r'\b([A-E])\b', raw_output)
    return match.group(1) if match else ""

def get_last_token_hidden_state(prompt: str, target_layer: int = 20):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    layer_hidden = outputs.hidden_states[target_layer]
    last_token_hidden = layer_hidden[0, -1, :]
    return last_token_hidden

def generate_answer(question: str, context: str) -> str:
    prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{context}

Question:
{question}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return extract_answer_letter(answer)

# ================= 加载测试集 =================
print(f"Loading test set from {TEST_FILE}...")
all_test_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_test_samples.append(json.loads(line))
if TEST_SIZE is not None:
    random.seed(RANDOM_SEED)
    test_samples = random.sample(all_test_samples, TEST_SIZE)
else:
    test_samples = all_test_samples
print(f"Using {len(test_samples)} test samples.")

# ================= 收集风险分数和正确性 =================
print("\nCollecting SAE features (only top 10 indices) and computing risk scores...")
risk_scores = []
correctness = []

for idx, sample in enumerate(tqdm(test_samples, desc="Processing")):
    question = sample["input"]
    true_label = sample["output"]

    # 检索
    retrieved_docs = vector_store.similarity_search(question, k=TOP_K)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt_text = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{context}

Question:
{question}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""

    hidden = get_last_token_hidden_state(prompt_text, target_layer=20)
    features = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu()  # (feature_dim,)
    # 提取选中的特征
    selected = features[TOP_FEATURE_INDICES].numpy()
    # 风险分数 = 加权和（使用预先计算的特征区分度得分作为权重）
    risk = np.sum(selected * FEATURE_SCORES)
    risk_scores.append(risk)
    
    pred_label = generate_answer(question, context)
    is_correct = (pred_label == true_label)
    correctness.append(is_correct)

risk_scores = np.array(risk_scores)
correctness = np.array(correctness, dtype=bool)

# ================= 评估风险分数的检测能力 =================
# 真实标签：1 表示错误
y_true = (~correctness).astype(int)

# AUC
auc = roc_auc_score(y_true, risk_scores)   # 风险越高，越可能错误，所以 AUC 应 >0.5
print(f"\n=== Risk Score Detection Performance ===")
print(f"AUC: {auc:.4f}")

# 阈值优化（F2 分数）
thresholds = np.linspace(risk_scores.min(), risk_scores.max(), 101)
best_f2 = 0
best_th = 0
for th in thresholds:
    pred = (risk_scores >= th).astype(int)
    f2 = fbeta_score(y_true, pred, beta=2)
    if f2 > best_f2:
        best_f2 = f2
        best_th = th

pred_opt = (risk_scores >= best_th).astype(int)
precision = precision_score(y_true, pred_opt)
recall = recall_score(y_true, pred_opt)
print(f"Optimal threshold: {best_th:.4f}")
print(f"F2 score: {best_f2:.4f}")
print(f"Precision (error detection): {precision:.4f}")
print(f"Recall (error detection): {recall:.4f}")

# RAG 模型本身的准确率
rag_acc = correctness.sum() / len(correctness)
print(f"\nRAG model accuracy: {rag_acc:.2%}")

# 保存结果
np.savez("rag_sae_final.npz", risk_scores=risk_scores, correctness=correctness,
         top_indices=TOP_FEATURE_INDICES, top_scores=FEATURE_SCORES, auc=auc)
print("Saved results to rag_sae_final.npz")