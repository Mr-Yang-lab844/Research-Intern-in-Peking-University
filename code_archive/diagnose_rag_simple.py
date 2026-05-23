#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零样本诊断：直接让模型回答选择题，不检索知识库，打印原始输出和提取结果"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 配置（请根据实际情况修改）====================
MODEL_NAME = "/home/yangchenfeng_intern/work/medical_sae_project/models/Llama-3.1-8B-Instruct"   # 替换为实际模型路径
TEST_FILE = "/home/yangchenfeng_intern/work/medical_sae_project/data/Text/test.jsonl"  # 或使用 MM/test.jsonl
MAX_SAMPLES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_test_samples(path: str):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            q = data.get("question", "")
            true_ans = data.get("label", data.get("answer", data.get("true", ""))).strip()
            samples.append({"question": q, "true_answer": true_ans})
    print(f"Loaded {len(samples)} test samples.")
    return samples

def load_llm():
    print(f"Loading model from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

def generate(prompt: str, tokenizer, model) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full[len(prompt):].strip()
    return response

def main():
    test_samples = load_test_samples(TEST_FILE)[:MAX_SAMPLES]
    if not test_samples:
        print("No test samples.")
        return
    tokenizer, model = load_llm()
    
    correct = 0
    for i, s in enumerate(test_samples):
        q = s["question"]
        true = s["true_answer"]
        print(f"\n{'='*60}\nSample {i+1}")
        print(f"Question: {q[:300]}...")
        print(f"Expected: {true}")
        
        prompt = f"""Answer the following multiple-choice question with only the letter (A, B, C, D, etc.).

Question: {q}

Answer (just the letter):"""
        
        raw = generate(prompt, tokenizer, model)
        print(f"Raw output:\n{raw}")
        extracted = extract_answer_letter(raw)
        print(f"Extracted: '{extracted}'")
        if extracted == true:
            correct += 1
            print(">>> CORRECT")
        else:
            print(f">>> WRONG (expected {true})")
    
    print(f"\nAccuracy: {correct}/{len(test_samples)} = {correct/len(test_samples)*100:.2f}%")

if __name__ == "__main__":
    main()