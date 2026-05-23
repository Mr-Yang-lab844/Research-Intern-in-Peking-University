#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Baseline for MedXpertQA (Text subset) - 使用外部医学知识库
- 知识库来源：huatuo26M 抽取的纯文本段落（medical_knowledge.jsonl）
- 测试集：data/Text/test.jsonl（MedXpertQA）
- 无数据泄露
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

# ================= 配置 =================
DATA_ROOT = "./data"
TEST_FILE = os.path.join(DATA_ROOT, "Text/test.jsonl")   # 测试集
KB_FILE = "./knowledge/medical_knowledge.jsonl"          # 外部知识库（请修改为实际路径）

MODEL_PATH = "./models/Llama-3.1-8B-Instruct"

# 检索参数
TOP_K = 3
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 10
TEMPERATURE = 0.0

# 测试集抽样（与之前保持一致，但不从测试集中切分知识库）
TEST_SIZE = 100        # 测试样本数
RANDOM_SEED = 42

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

# ================= 加载测试数据 =================
print(f"Loading test samples from {TEST_FILE}...")
all_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_samples.append(json.loads(line))

random.seed(RANDOM_SEED)
random.shuffle(all_samples)
test_samples = all_samples[:TEST_SIZE]
print(f"测试样本数: {len(test_samples)}")

# ================= 加载外部知识库 =================
def load_external_knowledge(kb_file: str):
    docs = []
    with open(kb_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            if text and len(text) > 20:
                docs.append(Document(page_content=text))
    print(f"加载知识库文档数: {len(docs)} 来自 {kb_file}")
    return docs

print("Loading external knowledge base...")
kb_docs = load_external_knowledge(KB_FILE)
if len(kb_docs) == 0:
    raise ValueError("知识库为空，请检查 KB_FILE 路径: " + KB_FILE)

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': DEVICE}
)

vector_store = FAISS.from_documents(kb_docs, embedding_model)
print("向量库构建完成。")

# ================= 辅助函数 =================
def extract_answer_letter(raw_output: str) -> str:
    import re
    match = re.search(r'\b([A-J])(?:\.|\s|$)', raw_output)
    return match.group(1) if match else ""

def generate_answer(question: str, context: str) -> str:
    prompt = f"""You are a medical expert. Answer the following multiple-choice question using ONLY the provided knowledge.

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
print("\nEvaluating on test set...")
correct = 0
total = len(test_samples)

for idx, sample in enumerate(tqdm(test_samples, desc="Processing")):
    question = sample["question"]
    true_label = sample["label"]
    
    retrieved_docs = vector_store.similarity_search(question, k=TOP_K)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    pred_label = generate_answer(question, context)
    
    if pred_label == true_label:
        correct += 1
    else:
        if idx < 10:
            print(f"\n[Error] Q: {question[:80]}... True: {true_label}, Pred: {pred_label}")

accuracy = correct / total * 100
print(f"\n=== Accuracy: {correct}/{total} = {accuracy:.2f}% ===")