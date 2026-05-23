#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature attribution for SAE (fixed division by zero)
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
from sklearn.metrics import roc_auc_score

# ================= 配置 =================
KNOWLEDGE_FILE = "./data/train.jsonl"
TEST_FILE = "./data/test.jsonl"
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
SAE_WEIGHT_FILE = "./models/Llama-Scope/L20R-8x.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 3
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
MAX_NEW_TOKENS = 10
TEST_SIZE = 100
RANDOM_SEED = 42
SAE_TOP_K = 32
HIDDEN_DIM = 4096

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
feature_dim = W_enc.shape[0]
print(f"SAE: hidden_dim={HIDDEN_DIM}, feature_dim={feature_dim}, TopK={SAE_TOP_K}")

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
random.seed(RANDOM_SEED)
test_samples = random.sample(all_test_samples, TEST_SIZE)
print(f"Using {TEST_SIZE} random samples (seed={RANDOM_SEED})")

# ================= 收集特征和正确性 =================
print("\nCollecting SAE features and correctness...")
feature_vectors = []
correctness_list = []

for idx, sample in enumerate(tqdm(test_samples, desc="Processing")):
    question = sample["input"]
    true_label = sample["output"]

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
    features = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu()  # [feature_dim]
    feature_vectors.append(features.numpy())
    
    pred_label = generate_answer(question, context)
    is_correct = (pred_label == true_label)
    correctness_list.append(is_correct)

# ================= 特征区分度分析（安全处理 NaN） =================
print("\nComputing per-feature discrimination scores...")
feature_matrix = np.array(feature_vectors)  # (n_samples, feature_dim)
correct_mask = np.array(correctness_list, dtype=bool)
error_mask = ~correct_mask

# 如果某一类样本数为0，则跳过（这里两类都有）
if correct_mask.sum() == 0 or error_mask.sum() == 0:
    print("Warning: One of the classes has zero samples. Cannot compute discrimination.")
    exit()

correct_activations = feature_matrix[correct_mask]
error_activations = feature_matrix[error_mask]

mean_correct = correct_activations.mean(axis=0)
mean_error = error_activations.mean(axis=0)
std_correct = correct_activations.std(axis=0)
std_error = error_activations.std(axis=0)

# 计算 pooled std，避免除零
pooled_std = np.sqrt(std_correct**2 + std_error**2)
# 将 std 接近于0的特征得分设为0
score = np.zeros_like(mean_correct)
valid = pooled_std > 1e-8
if valid.any():
    score[valid] = np.abs(mean_error[valid] - mean_correct[valid]) / pooled_std[valid]

# 选择 Top-K 特征（忽略得分为0的）
K = 10
top_indices = np.argsort(score)[-K:][::-1]
top_scores = score[top_indices]

print(f"Top {K} features and their discrimination scores:")
for i, (idx, sc) in enumerate(zip(top_indices, top_scores)):
    print(f"  {i+1}. Feature {idx}: score = {sc:.4f}")

# ================= 使用选出的特征重新计算风险分数 =================
selected_weights = score[top_indices]
selected_features = feature_matrix[:, top_indices]
new_risk = np.sum(selected_features * selected_weights, axis=1)

# 评估新风险分数的区分能力（AUC）
# 注意：风险越高应越可能错误，因此使用负相关
auc = roc_auc_score(correctness_list, -new_risk)
print(f"\nAUC using top {K} features: {auc:.4f}")

# 比较简单 L1 范数的 AUC
l1_risks = np.sum(feature_matrix, axis=1)
auc_l1 = roc_auc_score(correctness_list, -l1_risks)
print(f"AUC using L1 norm: {auc_l1:.4f}")

# 保存结果
np.savez("feature_analysis.npz", feature_matrix=feature_matrix, correctness=correctness_list,
         top_indices=top_indices, top_scores=top_scores, new_risk=new_risk, l1_risk=l1_risks)
print("Saved results to feature_analysis.npz")