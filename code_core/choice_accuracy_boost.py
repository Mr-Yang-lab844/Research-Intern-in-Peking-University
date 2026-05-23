#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选择题准确率扫描：测试不同检索数量 TOP_K
"""

import json
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

# ================= 配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
KB_FILES = [
    "./knowledge/medical_textbook_chunks.jsonl",
    "./knowledge/reference_ranges.jsonl",
    "./data/train.jsonl"
]
TEST_FILE = "./data/test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
MAX_NEW_TOKENS = 10
TEST_SIZE = 100
RANDOM_SEED = 42
DEVICE = "cuda"

# 待测试的检索数量列表
TOP_K_LIST = [3, 5, 7, 10, 15, 20]

# 加载 LLM
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# 构建知识库（只做一次）
print("Building knowledge base...")
docs = []
for file_path in KB_FILES:
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
print(f"Total documents: {len(docs)}")

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': DEVICE}
)
print("Building FAISS index...")
vector_store = FAISS.from_documents(docs, embedding_model)
print("Index ready.")

# 加载测试集
print("Loading test set...")
all_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_samples.append(json.loads(line))
random.seed(RANDOM_SEED)
test_samples = random.sample(all_samples, TEST_SIZE)
print(f"Test samples: {len(test_samples)}")

# 辅助函数
def extract_answer_letter(raw_output: str) -> str:
    match = re.search(r'\b([A-E])\b', raw_output)
    return match.group(1) if match else ""

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
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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

# 扫描不同 TOP_K
print("\n=== Accuracy Sweep ===")
for top_k in TOP_K_LIST:
    correct = 0
    for sample in tqdm(test_samples, desc=f"TOP_K={top_k}", leave=False):
        question = sample["input"]
        true_label = sample["output"]
        retrieved = vector_store.similarity_search(question, k=top_k)
        context = "\n\n".join([doc.page_content for doc in retrieved])
        pred = generate_answer(question, context)
        if pred == true_label:
            correct += 1
    acc = correct / len(test_samples) * 100
    print(f"TOP_K={top_k:2d} : Accuracy = {correct}/{len(test_samples)} = {acc:.2f}%")