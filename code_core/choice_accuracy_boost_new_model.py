#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调试版2：增加 max_tokens=2000，严格提取 ## Final Response，打印输出尾部"""

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
MODEL_PATH = "./models/HuatuoGPT-o1-8B"
KB_FILES = [
    "./knowledge/medical_textbook_chunks.jsonl",
    "./knowledge/reference_ranges.jsonl",
    "./data/train.jsonl"
]
TEST_FILE = "./data/test.jsonl"
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
MAX_NEW_TOKENS = 2000          # 增加，确保模型能输出完整 ## Final Response
TEST_SIZE = 20
RANDOM_SEED = 42
DEVICE = "cuda"
TOP_K = 3

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# 构建知识库（复用之前的代码）
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
all_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_samples.append(json.loads(line))
random.seed(RANDOM_SEED)
test_samples = random.sample(all_samples, TEST_SIZE)
print(f"Test samples: {len(test_samples)}\n")

def extract_answer_strict(full_output: str) -> str:
    """严格模式：只有找到 ## Final Response 且其中包含字母才返回，否则返回空"""
    if "## Final Response" not in full_output:
        return ""
    response_part = full_output.split("## Final Response")[-1].strip()
    # 匹配独立字母 A-E
    match = re.search(r'\b([A-E])\b', response_part)
    if match:
        return match.group(1)
    # 尝试匹配 "B." "B、" 等格式
    match = re.search(r'([A-E])[\s\.\、\)\]\:]', response_part)
    if match:
        return match.group(1)
    return ""

def generate_answer(question: str, context: str) -> tuple:
    user_content = f"""Question: {question}

Knowledge:
{context}

Output only the letter of the correct answer."""
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_answer_strict(full_output)
    return pred, full_output

print(f"=== Debugging with TOP_K={TOP_K} (max_tokens=2000, strict extraction) ===\n")
correct = 0
for i, sample in enumerate(tqdm(test_samples, desc="Processing")):
    question = sample["input"]
    true_label = sample["output"]
    retrieved = vector_store.similarity_search(question, k=TOP_K)
    context = "\n\n".join([doc.page_content for doc in retrieved])
    pred, full_out = generate_answer(question, context)
    if pred == true_label:
        correct += 1
        status = "✓"
    else:
        status = "✗"
    print(f"\n[{i+1}] {status} | True: {true_label} | Pred: {pred if pred else '(empty)'}")
    # 打印输出最后 500 字符，便于观察答案部分
    print("Output (last 500 chars):")
    print(full_out[-500:])
    print("-" * 80)

acc = correct / TEST_SIZE * 100
print(f"\nTOP_K={TOP_K} : Accuracy = {correct}/{TEST_SIZE} = {acc:.2f}%")