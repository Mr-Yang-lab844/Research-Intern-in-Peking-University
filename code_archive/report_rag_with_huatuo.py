#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告解读 RAG + SAE (使用华佗知识库)
- 解析自由文本报告
- 从华佗知识库中检索相关段落（关键词匹配）
- Llama 生成解读
- SAE (Layer 20) 计算风险分数
"""

import re
import json
import torch
import numpy as np
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 配置 ==========
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
SAE_PATH = "./models/Llama-Scope/L20R-8x.safetensors"
KNOWLEDGE_FILE = "./knowledge/medical_knowledge.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 加载华佗知识库 ==========
print("Loading Huatuo knowledge base...")
knowledge_entries = []
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # 华佗数据格式：{"questions": "...", "answers": "..."} 或 {"text": "..."}
        if "answers" in data:
            text = data["answers"]
        elif "text" in data:
            text = data["text"]
        else:
            text = list(data.values())[0] if data else ""
        if text:
            knowledge_entries.append(text.strip())
print(f"Loaded {len(knowledge_entries)} knowledge entries.")

# 简单检索：返回包含查询关键词的条目（前3个）
def retrieve_knowledge(query, top_k=3):
    query_lower = query.lower()
    scored = []
    for entry in knowledge_entries:
        # 计算匹配的关键词数量（可改进为TF-IDF，但简单够用）
        score = sum(1 for word in query_lower.split() if word in entry.lower())
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [entry for _, entry in scored[:top_k]]

# ========== 加载模型和SAE ==========
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

print("Loading SAE...")
sae_weights = safetensors.torch.load_file(SAE_PATH)
W_enc = sae_weights['encoder.weight'].to(DEVICE).to(torch.float16)
b_enc = sae_weights['encoder.bias'].to(DEVICE).to(torch.float16)
SAE_TOP_K = 32

def sae_encode(hidden):
    z = hidden @ W_enc.T + b_enc
    topk_vals, topk_idx = torch.topk(z, SAE_TOP_K, dim=-1)
    features = torch.zeros_like(z)
    features.scatter_(-1, topk_idx, topk_vals)
    features = torch.relu(features)
    return features

# ========== 报告解析 ==========
def parse_report(text):
    """提取指标名、数值、单位"""
    # 支持 "血糖：7.2 mmol/L" 或 "血糖 7.2mmol/L"
    pattern = r'([\u4e00-\u9fa5a-zA-Z]+)[：:\s]*([\d\.]+)\s*([a-zA-Z/]+)?'
    matches = re.findall(pattern, text)
    results = []
    for name, value, unit in matches:
        results.append({
            "test": name.strip(),
            "value": float(value),
            "unit": unit.strip() if unit else ""
        })
    return results

# ========== LLM生成解读 ==========
def generate_interpretation(test_name, value, unit, knowledge_text):
    prompt = f"""你是医学专家。根据以下医学知识，对检验结果给出简短解读（1-2句话），并给出建议。

医学知识：
{knowledge_text}

检验项目：{test_name}，结果：{value}{unit}。

解读和建议："""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "解读和建议：" in answer:
        answer = answer.split("解读和建议：")[-1].strip()
    return answer

# ========== 提取隐藏状态并计算风险 ==========
def compute_risk(prompt_text, target_layer=20):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[target_layer][0, -1, :]  # last token
    features = sae_encode(hidden.unsqueeze(0))
    risk = features.abs().sum().item()
    return risk

# ========== 主流程 ==========
if __name__ == "__main__":
    # 测试报告样例（可替换成你自己的）
    reports = [
        "血糖：7.2 mmol/L",
        "总胆固醇：6.1 mmol/L",
        "血红蛋白：110 g/L",
        "血糖：5.1 mmol/L",
    ]

    for idx, report in enumerate(reports):
        print(f"\n{'='*50}\n报告 {idx+1}: {report}")
        parsed = parse_report(report)
        if not parsed:
            print("无法解析")
            continue
        item = parsed[0]
        test = item["test"]
        value = item["value"]
        unit = item["unit"]

        # 构建查询
        query = f"{test} {value}{unit}"
        knowledge_entries_ret = retrieve_knowledge(query, top_k=2)
        knowledge_text = "\n".join(knowledge_entries_ret) if knowledge_entries_ret else "无相关医学知识。"

        print(f"检索到 {len(knowledge_entries_ret)} 条相关知识（片段）")

        # 构造 prompt（用于生成和风险计算）
        prompt = f"检验项目：{test}，结果：{value}{unit}。\n医学知识：{knowledge_text}\n请给出解读和建议："
        
        # 生成解读
        interpretation = generate_interpretation(test, value, unit, knowledge_text)
        print(f"模型解读: {interpretation}")

        # 计算风险
        risk = compute_risk(prompt)
        print(f"SAE 风险分数: {risk:.4f}")