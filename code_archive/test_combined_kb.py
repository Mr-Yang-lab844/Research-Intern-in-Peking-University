#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分别加载教科书知识库和参考范围知识库
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

# ================= 加载知识库 =================
print("加载知识库...")
docs = []

# 加载教科书
with open(TEXTBOOK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        content = data["text"]
        docs.append(Document(page_content=content, metadata={"source": "textbook"}))

# 加载参考范围
with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        content = f"{data['test_name']}：正常范围 {data['normal_range']} {data['unit']}。{data['description']}"
        docs.append(Document(page_content=content, metadata={"source": "reference"}))

print(f"共加载 {len(docs)} 个文档")

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

# ================= 选择题评估 =================
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
print(f"\n=== 选择题准确率: {correct}/{len(test_samples)} = {accuracy:.2f}% ===")

# ================= 自由文本报告解读演示 =================
print("\n" + "="*60)
print("自由文本报告解读演示")
print("="*60)

demo_reports = [
    "血糖：7.2 mmol/L",
    "总胆固醇：6.1 mmol/L，参考范围 <5.2",
]

for report in demo_reports:
    print(f"\n用户输入: {report}")
    retrieved = vector_store.similarity_search(report, k=2)
    context = "\n\n".join([doc.page_content for doc in retrieved])
    prompt = f"你是医学专家。根据以下医学知识，对检验结果给出简短解读（1-2句话）并给出建议。\n\n知识：{context}\n\n检验报告：{report}\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "解读：" in answer:
        answer = answer.split("解读：")[-1].strip()
    print(f"模型解读: {answer}")