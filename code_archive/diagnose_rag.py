#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 诊断脚本（与 baseline 行为一致）
- 从 test.jsonl 中随机抽取 KNOWLEDGE_SIZE 条作为知识库
- 对后续 TEST_SIZE 条进行 RAG 评估
- 打印每个测试样本的检索文档、原始模型输出、提取结果
"""

import json
import re
import random
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain_community.docstore.document import Document

# ==================== 配置参数（与 rag_baseline.py 对齐）====================
DATA_ROOT = "./data"
TEST_FILE = "./data/Text/test.jsonl"          # 数据源（和 baseline 一致）
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
KNOWLEDGE_SIZE = 200      # 知识库样本数
TEST_SIZE = 100           # 测试样本数
RANDOM_SEED = 42

MAX_TEST_SAMPLES = 10     # 诊断时只打印前10个（避免太长）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 答案提取函数（鲁棒） ====================
def extract_answer_letter(text: str) -> str:
    patterns = [
        r"(?:answer|choice|option|key)(?:\s+is)?\s*([A-J])\b",
        r"\b([A-J])\b(?:\s*[\.\:\)])",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    words = re.findall(r'\b([A-J])\b', text)
    if words:
        return words[0].upper()
    return ""

# ==================== 加载模型 ====================
def load_llm():
    print(f"Loading LLM from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

# ==================== 构建知识库（与 baseline 完全相同） ====================
def build_vector_store_from_samples(samples: List[Dict]):
    """从样本列表构建 FAISS 向量库"""
    docs = []
    for data in samples:
        question = data["question"]
        options = data["options"]
        correct_letter = data["label"]
        correct_text = options[correct_letter]
        doc_text = f"Question: {question}\nCorrect answer: {correct_letter}. {correct_text}"
        docs.append(Document(page_content=doc_text, metadata={"id": data["id"]}))
    
    print(f"Building vector store with {len(docs)} documents...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': DEVICE}
    )
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# ==================== 加载并划分数据 ====================
def load_and_split_data():
    print(f"Loading all samples from {TEST_FILE} ...")
    all_samples = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            all_samples.append(json.loads(line))
    
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)
    
    knowledge_samples = all_samples[:KNOWLEDGE_SIZE]
    test_samples = all_samples[KNOWLEDGE_SIZE:KNOWLEDGE_SIZE+TEST_SIZE]
    print(f"知识库样本数: {len(knowledge_samples)}, 测试样本数: {len(test_samples)}")
    return knowledge_samples, test_samples

# ==================== 生成答案 ====================
def generate_answer(question: str, context: str, tokenizer, model) -> str:
    prompt = f"""You are a medical expert. Answer the following multiple-choice question using ONLY the provided knowledge.

Knowledge:
{context}

Question:
{question}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 Answer: 之后的内容
    if "Answer:" in full_text:
        answer_part = full_text.split("Answer:")[-1].strip()
    else:
        answer_part = full_text[len(prompt):].strip()
    return answer_part

# ==================== 主诊断流程 ====================
def main():
    # 1. 加载并划分数据
    knowledge_samples, test_samples = load_and_split_data()
    if not test_samples:
        print("No test samples.")
        return
    
    # 2. 构建知识库向量库
    vector_store = build_vector_store_from_samples(knowledge_samples)
    
    # 3. 加载 LLM
    tokenizer, model = load_llm()
    
    # 4. 对前 MAX_TEST_SAMPLES 个测试样本进行诊断
    correct = 0
    for idx, sample in enumerate(test_samples[:MAX_TEST_SAMPLES]):
        question = sample["question"]
        true_label = sample["label"]
        
        # 检索
        retrieved_docs = vector_store.similarity_search(question, k=TOP_K)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}")
        print(f"Question: {question[:200]}...")
        print(f"Expected answer: {true_label}")
        print(f"\nRetrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  Doc {i+1}: {doc.page_content[:200]}...")
        
        # 生成答案
        raw_output = generate_answer(question, context, tokenizer, model)
        print(f"\nRaw model output (after extraction): {raw_output}")
        
        extracted = extract_answer_letter(raw_output)
        print(f"Extracted answer: {extracted}")
        
        is_correct = (extracted == true_label)
        if is_correct:
            correct += 1
            print(">>> CORRECT")
        else:
            print(f">>> WRONG (expected {true_label}, got {extracted})")
    
    print(f"\n{'='*60}")
    print(f"Diagnostic RAG accuracy on first {MAX_TEST_SAMPLES} samples: {correct}/{MAX_TEST_SAMPLES} = {correct/MAX_TEST_SAMPLES*100:.2f}%")

if __name__ == "__main__":
    main()