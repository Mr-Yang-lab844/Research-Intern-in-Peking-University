#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话助手 (Llama-3.1-8B-UltraMedical) - 适配 Gradio 最新版
- 自然对话，不限制提问格式
- 每轮展示模型回答 + SAE风险分数（基于用户输入的最后一个token）
- 使用 messages 格式（role + content）与新版 Chatbot 兼容
"""

import json
import torch
import numpy as np
import safetensors.torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

# ================= 配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-UltraMedical"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAE_TOP_K = 32
FREE_LAYER = 15
FREE_FEATURE_COUNT = 4

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

# ================= SAE 加载 =================
def load_sae(layer):
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
    w = safetensors.torch.load_file(sae_path)
    return w['encoder.weight'].to(DEVICE).to(torch.float16), w['encoder.bias'].to(DEVICE).to(torch.float16)

W_enc, b_enc = load_sae(FREE_LAYER)

def sae_encode(hidden):
    z = hidden @ W_enc.T + b_enc
    topk = torch.topk(z, SAE_TOP_K, dim=-1)
    f = torch.zeros_like(z)
    f.scatter_(-1, topk.indices, topk.values)
    return torch.relu(f)

def get_feature_for_message(message):
    """提取用户消息的最后一个token的隐藏状态，并返回SAE特征向量"""
    inputs = tokenizer(message, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[FREE_LAYER][0, -1, :]
    feat = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    return feat

# ================= SAE 标定 =================
def calibrate_sae():
    cache_file = "./cache/sae_calib_ultramedical.npz"
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data['top_idx'].tolist(), data['top_w'].tolist(), data['low_th'], data['high_th']
    dev_file = "./data/free_text_dev.jsonl"
    with open(dev_file, 'r') as f:
        dev = [json.loads(line) for line in f]
    feats, labels = [], []
    for sample in tqdm(dev, desc="Calibrating SAE"):
        msg = sample['report']
        label = sample['label']  # 1=正确,0=错误
        error_label = 1 - label
        feat = get_feature_for_message(msg)
        feats.append(feat)
        labels.append(error_label)
    X = np.array(feats)
    y = np.array(labels)
    correct = y == 0
    error = y == 1
    mean_c = X[correct].mean(axis=0)
    mean_e = X[error].mean(axis=0)
    std_c = X[correct].std(axis=0) + 1e-8
    std_e = X[error].std(axis=0) + 1e-8
    t = np.abs(mean_e - mean_c) / np.sqrt(std_c**2 + std_e**2)
    t[np.isnan(t)] = 0
    top_idx = np.argsort(t)[-FREE_FEATURE_COUNT:][::-1]
    top_w = t[top_idx]
    risks = np.array([np.sum(feat[top_idx] * top_w) for feat in X])
    low_th = np.percentile(risks, 30)
    high_th = np.percentile(risks, 70)
    os.makedirs("./cache", exist_ok=True)
    np.savez(cache_file, top_idx=top_idx, top_w=top_w, low_th=low_th, high_th=high_th)
    print(f"Calibration done. Low={low_th:.4f}, High={high_th:.4f}")
    return top_idx.tolist(), top_w.tolist(), low_th, high_th

print("Calibrating SAE (first run may take a minute)...")
free_top_idx, free_weights, FREE_LOW_TH, FREE_HIGH_TH = calibrate_sae()
print("Ready.")

# ================= 对话生成（适配新版 Gradio）=================
def generate_response(message, history):
    """
    history: 列表，每个元素为 {"role": "user"|"assistant", "content": ...}
    """
    if not message.strip():
        return "", history

    # 1. 构建完整的 messages 列表
    messages = history + [{"role": "user", "content": message}]

    # 2. 使用 chat template 生成 prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=800, do_sample=False, temperature=0.0)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取 assistant 回复（取最后一个 assistant 块）
    if "assistant" in full:
        response = full.split("assistant")[-1].strip()
    else:
        response = full.strip()

    # 3. 计算 SAE 风险（基于用户当前消息）
    feat = get_feature_for_message(message)
    risk = np.sum(feat[free_top_idx] * free_weights)
    if risk < FREE_LOW_TH:
        risk_level = "低风险 🟢"
    elif risk <= FREE_HIGH_TH:
        risk_level = "中风险 🟡"
    else:
        risk_level = "高风险 🔴"

    response_with_risk = f"{response}\n\n---\n**SAE风险评估**：{risk_level}\n(基于您当前问题的内部状态)"

    # 4. 更新历史（messages 格式）
    new_history = messages + [{"role": "assistant", "content": response_with_risk}]
    return "", new_history

# ================= Gradio 界面 =================
with gr.Blocks(title="医疗问答助手") as demo:
    gr.Markdown("# 🏥 医疗问答助手 (Llama-3.1-8B-UltraMedical)")
    gr.Markdown("输入常见的体检指标或健康问题，模型会尝试给出通俗易懂的解读，并提供SAE风险等级（基于问题内部状态）。请注意：模型对**常规指标**（如血糖、血压、血脂等）解释较为可靠； 对**冷门指标或复杂疾病机理**的回答可能存在错误，仅供参考；本系统不能替代专业医师诊断，如有身体不适请及时就医。")

    chatbot = gr.Chatbot(label="对话记录")   # 无需 type 参数，新版自动适配
    msg = gr.Textbox(label="输入消息", lines=2, placeholder="例如：血糖 6.1 mmol/L 正常吗？")
    send = gr.Button("发送")
    clear = gr.Button("清空对话")

    send.click(generate_response, [msg, chatbot], [msg, chatbot])
    msg.submit(generate_response, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)   # 清空历史

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)