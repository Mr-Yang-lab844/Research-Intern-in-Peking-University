#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成自由文本评估 Excel 文件
用于人工评估：
- 报告解析正确率（指标名、数值、异常判断）
- 就医建议合理性（三级评分）
- 失败案例记录
"""

import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import safetensors.torch
import random
import os

# ================= 配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FREE_TEST_FILE = "./data/free_text_test.jsonl"
OUTPUT_EXCEL = "./evaluation_results.xlsx"

# SAE 配置（与最终演示一致：层15，特征数4）
SAE_LAYER = 15
SAE_TOP_K = 32
FREE_FEATURE_COUNT = 4
CACHE_WEIGHTS = "./cache_supplement/free_weights.npy"   # 使用之前标定的权重

# 随机抽取样本数（可修改）
SAMPLE_SIZE = 50
RANDOM_SEED = 42

# ================= 加载模型 =================
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# ================= 加载 SAE 权重 =================
def load_sae(layer):
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
    w = safetensors.torch.load_file(sae_path)
    return w['encoder.weight'].to(DEVICE).to(torch.float16), w['encoder.bias'].to(DEVICE).to(torch.float16)

W_enc, b_enc = load_sae(SAE_LAYER)

def sae_encode(hidden):
    z = hidden @ W_enc.T + b_enc
    topk = torch.topk(z, SAE_TOP_K, dim=-1)
    f = torch.zeros_like(z)
    f.scatter_(-1, topk.indices, topk.values)
    return torch.relu(f)

def get_free_feature(report_text):
    prompt = f"在某一次体检中，我的{report_text}，正常吗？"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[SAE_LAYER][0, -1, :]
    feat = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    return feat

def generate_answer(report_text):
    prompt = f"在某一次体检中，我的{report_text}，正常吗？"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取“正常吗？”之后的内容
    if "正常吗？" in full_output:
        answer = full_output.split("正常吗？")[-1].strip()
    else:
        answer = full_output.strip()
    # 取第一个句号前的内容（避免过长）
    answer = answer.split('。')[0] + '。'
    return answer

# ================= 加载已标定的特征权重和阈值 =================
if os.path.exists(CACHE_WEIGHTS):
    data = np.load(CACHE_WEIGHTS, allow_pickle=True).item()
    free_top_idx = data['top_idx']
    free_weights = data['top_w']
    FREE_LOW_TH = data['low_th']
    FREE_HIGH_TH = data['high_th']
else:
    # 如果没有缓存，则从开发集标定（这里简化，直接使用之前实验的经验值）
    print("警告：未找到标定权重文件，将使用默认特征索引（前4个）")
    free_top_idx = [0,1,2,3]
    free_weights = np.array([1.0,1.0,1.0,1.0])
    FREE_LOW_TH = 0.4
    FREE_HIGH_TH = 0.6

def get_risk_level(risk):
    if risk < FREE_LOW_TH:
        return "低风险"
    elif risk <= FREE_HIGH_TH:
        return "中风险"
    else:
        return "高风险"

# ================= 加载测试集并抽样 =================
with open(FREE_TEST_FILE, 'r', encoding='utf-8') as f:
    all_samples = [json.loads(line) for line in f]
random.seed(RANDOM_SEED)
samples = random.sample(all_samples, min(SAMPLE_SIZE, len(all_samples)))
print(f"抽取 {len(samples)} 条样本")

# ================= 生成结果 =================
results = []
for i, s in enumerate(samples):
    report = s['report']
    true_label = s['label']  # 1=正确,0=错误
    # 生成模型回答
    answer = generate_answer(report)
    # 计算 SAE 风险分数
    feat = get_free_feature(report)
    risk = np.sum(feat[free_top_idx] * free_weights)
    risk_level = get_risk_level(risk)
    results.append({
        "序号": i+1,
        "原始报告": report,
        "真实标签 (1正确0错误)": true_label,
        "模型回答": answer,
        "SAE风险分数": round(risk, 4),
        "SAE风险等级": risk_level,
        "人工评分-报告解析正确率 (1-5)": "",
        "人工评分-就医建议合理性 (1-3)": "",
        "是否为失败案例 (是/否)": "",
        "备注": ""
    })

# ================= 保存为 Excel =================
df = pd.DataFrame(results)
df.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')
print(f"已生成评估 Excel 文件: {OUTPUT_EXCEL}")
print("请打开文件，逐条对模型回答进行人工评分。")