#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG using Huatuo knowledge base (5866 medical QA pairs) on CMExam test set.
Knowledge: ./knowledge/medical_knowledge.jsonl (pure text paragraphs)
Test set: ./data/test.jsonl (CMExam format)
Embedding: Chinese model (BAAI/bge-base-zh-v1.5)
"""

import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
import os
import re

# ================= 配置 =================
KNOWLEDGE_FILE = "./knowledge/medical_knowledge.jsonl"   # 华佗知识库
TEST_FILE = "./data/test.jsonl"                         # CMExam 测试集
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"

TOP_K = 3
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"    # 中文嵌入模型
MAX_NEW_TOKENS = 10

# 抽样数量（测试集太大可以只测前 N 条，0 表示全测）
TEST_SAMPLES = 100      # 设置为 0 则全测

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 加载 LLM =================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ================= 构建华佗知识库 =================
print(f"Loading knowledge base from {KNOWLEDGE_FILE}...")
docs = []
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # 假设每行格式为 {"text": "..."} 或直接是字符串？根据之前构建，每条是一个纯文本段落。
        # 检查字段：可能是 {"text": "..."}，也可能是 {"answer": "..."}。我们保守取第一个值。
        if "text" in data:
            content = data["text"]
        elif "answer" in data:
            content = data["answer"]
        else:
            # 取第一个字符串值
            content = list(data.values())[0] if data else ""
        if content:
            docs.append(Document(page_content=content, metadata={}))
print(f"Loaded {len(docs)} knowledge items.")

# 如果知识库太大，可以限制数量（例如前2000条）加快构建，但这里先全用
if len(docs) > 2000:
    print(f"Truncating to first 2000 items for speed (you can remove this limit).")
    docs = docs[:2000]

print("Loading embedding model (may download first time)...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': DEVICE}
)

print("Building FAISS vector store...")
vector_store = FAISS.from_documents(docs, embedding_model)
print("Vector store ready.")

# ================= 加载测试集 =================
print(f"Loading test set from {TEST_FILE}...")
test_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        test_samples.append(json.loads(line))
if TEST_SAMPLES > 0:
    test_samples = test_samples[:TEST_SAMPLES]
print(f"Test samples: {len(test_samples)}")

# ================= 辅助函数 =================
def extract_answer_letter(raw_output: str) -> str:
    match = re.search(r'\b([A-E])\b', raw_output)
    return match.group(1) if match else ""

def generate_answer(question: str, context: str) -> str:
    prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided medical knowledge.

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

# ================= 评估 =================
print("\nEvaluating RAG with Huatuo knowledge base...")
correct = 0
total = len(test_samples)

for idx, sample in enumerate(tqdm(test_samples, desc="Processing")):
    question = sample["input"]        # 包含问题和选项
    true_label = sample["output"]
    
    # 检索
    retrieved_docs = vector_store.similarity_search(question, k=TOP_K)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    pred_label = generate_answer(question, context)
    
    if pred_label == true_label:
        correct += 1
    else:
        if idx < 10:   # 仅打印前10个错误样例
            print(f"\n[Error] Q: {question[:80]}... True: {true_label}, Pred: {pred_label}")

accuracy = correct / total * 100
print(f"\n=== Accuracy with Huatuo knowledge: {correct}/{total} = {accuracy:.2f}% ===")