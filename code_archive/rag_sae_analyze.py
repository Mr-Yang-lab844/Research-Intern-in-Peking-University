#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG + SAE (Layer 20) with full training knowledge base and risk analysis.
Computes average risk for correct vs incorrect answers.
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

# ================= 构建完整知识库 =================
print(f"Loading knowledge base from {KNOWLEDGE_FILE}...")
docs = []
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        question = data["input"]
        answer = data["output"]
        doc_text = f"问题: {question}\n答案: {answer}"
        docs.append(Document(page_content=doc_text, metadata={}))
print(f"Loaded {len(docs)} documents. (No truncation)")

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': DEVICE}
)
print("Building FAISS vector store (this may take a few minutes)...")
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
    layer_hidden = outputs.hidden_states[target_layer]  # [1, seq_len, hid_dim]
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

# ================= 加载并抽样测试集 =================
print(f"Loading test set from {TEST_FILE}...")
all_test_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_test_samples.append(json.loads(line))
print(f"Total test samples: {len(all_test_samples)}")

random.seed(RANDOM_SEED)
test_samples = random.sample(all_test_samples, TEST_SIZE)
print(f"Using {TEST_SIZE} random samples (seed={RANDOM_SEED})")

# ================= 评估并记录风险与正确性 =================
print("\nEvaluating RAG + SAE (full knowledge base)...")
correct = 0
total = len(test_samples)
risk_scores = []
correctness = []

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
    features = sae_encode(hidden.unsqueeze(0))
    risk = features.abs().sum().item()
    risk_scores.append(risk)

    pred_label = generate_answer(question, context)
    is_correct = (pred_label == true_label)
    correctness.append(is_correct)

    if is_correct:
        correct += 1
    else:
        if idx < 10:
            print(f"\n[Error] Q: {question[:80]}... True: {true_label}, Pred: {pred_label}, Risk: {risk:.4f}")

accuracy = correct / total * 100
print(f"\n=== Accuracy: {correct}/{total} = {accuracy:.2f}% ===")
print(f"=== Average Risk Score (L1) over all samples: {np.mean(risk_scores):.4f} ===")

# 分别计算正确和错误样本的平均风险
correct_risks = [risk for risk, corr in zip(risk_scores, correctness) if corr]
error_risks = [risk for risk, corr in zip(risk_scores, correctness) if not corr]

print(f"Correct samples: {len(correct_risks)}, Avg risk = {np.mean(correct_risks):.4f}")
print(f"Error samples: {len(error_risks)}, Avg risk = {np.mean(error_risks):.4f}")

# 保存详细数据供后续分析
np.savez("rag_sae_results.npz", risk_scores=risk_scores, correctness=correctness)