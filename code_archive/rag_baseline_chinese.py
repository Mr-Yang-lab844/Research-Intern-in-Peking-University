#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 RAG 基线（无 SAE），用于与 SAE 版本对照。
使用全量训练集（52741条），从测试集中随机抽样 100 条（seed=42）。
"""

import json
import torch
import re
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

# 配置
KNOWLEDGE_FILE = "./data/train.jsonl"
TEST_FILE = "./data/test.jsonl"
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
TOP_K = 3
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
MAX_NEW_TOKENS = 10
TEST_SIZE = 100
RANDOM_SEED = 42

# 加载模型
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 加载全量知识库
print(f"Loading knowledge base from {KNOWLEDGE_FILE}...")
docs = []
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        doc_text = f"问题: {data['input']}\n答案: {data['output']}"
        docs.append(Document(page_content=doc_text, metadata={}))
print(f"Loaded {len(docs)} documents (all).")

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': 'cuda'})
print("Building FAISS vector store...")
vector_store = FAISS.from_documents(docs, embedding_model)
print("Vector store ready.")

# 加载测试集并随机抽样
print(f"Loading test set from {TEST_FILE}...")
all_test = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_test.append(json.loads(line))
random.seed(RANDOM_SEED)
test_samples = random.sample(all_test, TEST_SIZE)
print(f"Test samples: {len(test_samples)} (seed={RANDOM_SEED})")

# 辅助函数
def extract_letter(text):
    match = re.search(r'\b([A-E])\b', text)
    return match.group(1) if match else ""

def generate(question, context):
    prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{context}

Question:
{question}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    if "Answer:" in ans:
        ans = ans.split("Answer:")[-1].strip()
    return extract_letter(ans)

# 评估
correct = 0
for sample in tqdm(test_samples):
    question = sample["input"]
    true = sample["output"]
    retrieved = vector_store.similarity_search(question, k=TOP_K)
    context = "\n\n".join([doc.page_content for doc in retrieved])
    pred = generate(question, context)
    if pred == true:
        correct += 1

acc = correct / len(test_samples) * 100
print(f"\n=== Accuracy (without SAE): {correct}/{len(test_samples)} = {acc:.2f}% ===")