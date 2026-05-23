#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并知识库 + SAE 风险分数（自由文本 + 错误对比）
"""

import json
import random
import re
import torch
import numpy as np
import safetensors.torch
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
SAE_PATH = "./models/Llama-Scope/L20R-8x.safetensors"
TOP_K = 3
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
MAX_NEW_TOKENS_GEN = 120       # 自由文本生成
MAX_NEW_TOKENS_CHOICE = 10
TEST_SAMPLES = 100
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 之前标定的 Top-10 特征索引和权重
TOP_FEATURE_INDICES = [28843, 11033, 4945, 32434, 9495, 30755, 1483, 13022, 18938, 28720]
FEATURE_SCORES = [0.4675, 0.3923, 0.3853, 0.3308, 0.3303, 0.3206, 0.3184, 0.3162, 0.2998, 0.2786]

# ================= 加载知识库 =================
print("加载知识库...")
docs = []
with open(TEXTBOOK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        docs.append(Document(page_content=data["text"], metadata={"source": "textbook"}))
with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        content = f"{data['test_name']}：正常范围 {data['normal_range']} {data['unit']}。{data['description']}"
        docs.append(Document(page_content=content, metadata={"source": "reference"}))
print(f"共加载 {len(docs)} 个文档")

print("构建 FAISS 索引...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
vector_store = FAISS.from_documents(docs, embedding_model)
print("索引完成")

# ================= 加载模型和 SAE =================
print("加载 LLM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

print("加载 SAE...")
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

def get_risk(prompt_text, target_layer=20):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[target_layer][0, -1, :]  # last token
    features = sae_encode(hidden.unsqueeze(0)).squeeze(0).cpu().numpy()
    # 只使用 Top-10 特征加权
    selected = features[TOP_FEATURE_INDICES]
    risk = np.sum(selected * FEATURE_SCORES)
    return risk

def extract_letter(text):
    match = re.search(r'\b([A-E])\b', text)
    return match.group(1) if match else ""

def generate_answer(question, context, max_tokens=MAX_NEW_TOKENS_CHOICE):
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
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return extract_letter(answer)

def generate_free_text(report, context):
    prompt = f"你是医学专家。根据以下医学知识，对检验结果给出简短解读（1-2句话）并给出建议。\n\n知识：{context}\n\n检验报告：{report}\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS_GEN, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "解读：" in answer:
        answer = answer.split("解读：")[-1].strip()
    return answer

# ================= 选择题评估（可选） =================
print("\n加载测试集...")
all_test = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        all_test.append(json.loads(line))
random.seed(RANDOM_SEED)
test_samples = random.sample(all_test, TEST_SAMPLES)

correct = 0
for sample in tqdm(test_samples, desc="选择题评估"):
    q = sample["input"]
    true = sample["output"]
    retrieved = vector_store.similarity_search(q, k=TOP_K)
    ctx = "\n\n".join([d.page_content for d in retrieved])
    pred = generate_answer(q, ctx)
    if pred == true:
        correct += 1
print(f"\n选择题准确率: {correct}/{TEST_SAMPLES} = {correct/TEST_SAMPLES*100:.2f}%")

# ================= 自由文本 + SAE 演示 =================
print("\n" + "="*70)
print("自由文本报告解读 + SAE 风险分数（对比正确与错误解读）")
print("="*70)

demo_reports = [
    ("血糖：7.2 mmol/L", "偏高"),
    ("总胆固醇：6.1 mmol/L", "偏高"),
]

for report, true_status in demo_reports:
    print(f"\n【报告】{report}")
    retrieved = vector_store.similarity_search(report, k=2)
    context = "\n\n".join([d.page_content for d in retrieved])
    
    # 正确解读（基于检索到的知识）
    correct_interpret = generate_free_text(report, context)
    # 计算正确解读的风险分数（基于输入 prompt 的隐藏状态）
    # 注意：风险计算使用同一个 prompt（包含知识），但生成解读后我们取生成 prompt 的最后一个 token
    # 这里我们重新构造 prompt（与生成相同），然后计算风险
    prompt_for_risk = f"你是医学专家。根据以下医学知识，对检验结果给出简短解读（1-2句话）并给出建议。\n\n知识：{context}\n\n检验报告：{report}\n解读："
    risk_correct = get_risk(prompt_for_risk)
    
    # 错误解读：篡改知识（例如将“偏高”说成“正常”）
    wrong_knowledge = context.replace("偏高", "正常").replace("升高", "正常")
    wrong_prompt = f"你是医学专家。根据以下医学知识，对检验结果给出简短解读（1-2句话）并给出建议。\n\n知识：{wrong_knowledge}\n\n检验报告：{report}\n解读："
    risk_wrong = get_risk(wrong_prompt)
    
    print(f"正确解读: {correct_interpret}")
    print(f"正确解读的风险分数: {risk_correct:.4f}")
    print(f"错误解读的风险分数（篡改知识）: {risk_wrong:.4f}")
    print(f"风险变化: {risk_wrong - risk_correct:+.4f}")