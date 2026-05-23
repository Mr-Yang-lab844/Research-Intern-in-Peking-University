#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot baseline for Chinese medical multiple-choice questions.
No RAG, no external knowledge. Just the model's own parameters.
"""

import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ================= 配置 =================
TEST_FILE = "./data/test.jsonl"      # 中文测试集（已转换的CMExam格式）
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
MAX_SAMPLES = 100                     # 只测前100条，节省时间（想全测可改为None）
RANDOM_SEED = 42                     # 若想随机抽样，可设置seed并打乱
# =======================================

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def extract_answer_letter(text: str) -> str:
    """从生成文本中提取选项字母（A,B,C,D,E）"""
    match = re.search(r'\b([A-E])\b', text)
    return match.group(1) if match else ""

# 加载测试集
print(f"Loading test samples from {TEST_FILE}...")
samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

if MAX_SAMPLES:
    samples = samples[:MAX_SAMPLES]
print(f"Total test samples: {len(samples)}")

# 零样本评测
correct = 0
total = len(samples)

print("\nRunning zero-shot evaluation...")
for idx, sample in enumerate(tqdm(samples, desc="Processing")):
    question = sample["input"]          # 已经包含了问题和选项
    true_label = sample["output"]       # 答案字母
    
    prompt = f"""You are a medical expert. Answer the following multiple-choice question with the single letter of the correct option.

Question:
{question}

Answer (just the letter, e.g., "A"):"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_answer_letter(answer)
    
    if pred == true_label:
        correct += 1
    else:
        # 可选：打印前几个错误案例
        if idx < 10:
            print(f"\n[Error] Q: {question[:80]}... True: {true_label}, Pred: {pred}")

accuracy = correct / total * 100
print(f"\n=== Zero-shot Accuracy: {correct}/{total} = {accuracy:.2f}% ===")