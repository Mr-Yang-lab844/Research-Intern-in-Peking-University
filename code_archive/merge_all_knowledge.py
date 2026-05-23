#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并四个知识库（使用已有的 medical_knowledge.jsonl 作为华佗知识源）
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
CMEXAM_TRAIN = "./data/train.jsonl"
HUATUO_FILE = "./knowledge/medical_knowledge.jsonl"   # 修正路径
TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
TEST_FILE = "./data/test.jsonl"
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
TOP_K = 3
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
MAX_NEW_TOKENS = 10
TEST_SAMPLES = 100
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 加载所有知识库 =================
print("加载知识库...")
docs = []

# 1. CMExam 训练集
print("加载 CMExam train.jsonl...")
with open(CMEXAM_TRAIN, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        content = f"问题: {data['input']}\n正确答案: {data['output']}"
        docs.append(Document(page_content=content, metadata={"source": "cmexam_train"}))

# 2. 华佗知识库
print("加载华佗 medical_knowledge.jsonl...")
with open(HUATUO_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # 兼容不同格式
        if "text" in data:
            content = data["text"]
        elif "answer" in data:
            content = data["answer"]
        else:
            content = list(data.values())[0] if data else ""
        if content:
            docs.append(Document(page_content=content, metadata={"source": "huatuo"}))

# 3. 教科书分块
print("加载教科书 chunks...")
with open(TEXTBOOK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        docs.append(Document(page_content=data["text"], metadata={"source": "textbook"}))

# 4. 参考范围
print("加载参考范围...")
with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        content = f"{data['test_name']}：正常范围 {data['normal_range']} {data['unit']}。{data['description']}"
        docs.append(Document(page_content=content, metadata={"source": "reference"}))

print(f"总共加载 {len(docs)} 个文档")

# 构建 FAISS 索引
print("构建 FAISS 索引...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
vector_store = FAISS.from_documents(docs, embedding_model)
print("索引构建完成")

# ================= 加载测试集 =================
print("加载测试集...")
all_test = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_test.append(json.loads(line))
random.seed(RANDOM_SEED)
test_samples = random.sample(all_test, TEST_SAMPLES)
print(f"测试集大小: {len(test_samples)}")

# ================= 加载模型 =================
print("加载 LLM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# ================= 辅助函数 =================
def extract_letter(text):
    match = re.search(r'\b([A-E])\b', text)
    return match.group(1) if match else ""

def generate_answer(question, context):
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
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return extract_letter(answer)

# ================= 评估 =================
print("\n开始评估选择题准确率...")
correct = 0
for sample in tqdm(test_samples):
    question = sample["input"]
    true_label = sample["output"]
    retrieved = vector_store.similarity_search(question, k=TOP_K)
    context = "\n\n".join([doc.page_content for doc in retrieved])
    pred = generate_answer(question, context)
    if pred == true_label:
        correct += 1

accuracy = correct / len(test_samples) * 100
print(f"\n=== 选择题准确率 (合并所有知识库): {correct}/{len(test_samples)} = {accuracy:.2f}% ===")