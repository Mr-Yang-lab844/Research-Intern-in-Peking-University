#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio 演示界面（最终优化版）：支持选择题、单轮自由文本（增强知识库 + 实时SAE标定）、多轮对话。
- 选择题模块：使用原知识库（教科书+普通参考范围+同源训练集），SAE特征沿用之前实验标定的固定列表。
- 自由文本单轮模块：使用增强知识库（教科书+离散化参考范围），在启动时重新标定SAE特征（基于开发集）。
- 多轮对话：不计算SAE风险，仅基于检索回答。
- 支持多种分隔符（逗号、分号、顿号、竖线等），推荐使用逗号或分号分隔。
- 添加硬规则修正，解决常见误判（如正常肌酐被误判偏高、正常体温被误判发热等）。
"""

import json
import re
import torch
import numpy as np
import safetensors.torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
import os
import random

# ================= 配置 =================
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
ORIG_RANGE_FILE = "./knowledge/reference_ranges.jsonl"
ENHANCED_RANGE_FILE = "./knowledge/reference_ranges_enhanced.jsonl"
TRAIN_FILE = "./data/train.jsonl"

EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
SAE_TOP_K = 32

CHOICE_LAYER = 22
CHOICE_FEATURE_COUNT = 10
CHOICE_TOP_K = 20
FREE_LAYER = 16
FREE_FEATURE_COUNT = 11
FREE_TOP_K = 3

CHOICE_LOW_TH = 5.0
CHOICE_HIGH_TH = 5.8
FREE_LOW_TH = 0.25
FREE_HIGH_TH = 0.36

FREE_DEV_FILE = "./data/free_text_dev.jsonl"

# 指标步长配置（用于数值舍入）
ROUND_CONFIG = {
    "血糖": 0.5, "血压收缩压": 5, "收缩压": 5, "血压舒张压": 5, "舒张压": 5,
    "总胆固醇": 0.5, "甘油三酯": 0.5, "高密度脂蛋白": 0.5, "低密度脂蛋白": 0.5,
    "血红蛋白": 5, "血小板计数": 10, "白细胞计数": 0.5, "红细胞计数": 0.2,
    "红细胞压积": 1, "中性粒细胞百分比": 1, "淋巴细胞百分比": 1, "单核细胞百分比": 0.5,
    "嗜酸性粒细胞百分比": 0.5, "嗜碱性粒细胞百分比": 0.1, "凝血酶原时间": 0.5,
    "活化部分凝血活酶时间": 1, "纤维蛋白原": 0.5, "凝血酶时间": 0.5,
    "谷丙转氨酶": 5, "谷草转氨酶": 5, "总胆红素": 1, "白蛋白": 2,
    "碱性磷酸酶": 5, "γ-谷氨酰转肽酶": 5, "肌酐": 5, "尿素氮": 0.5,
    "尿酸": 10, "胱抑素C": 0.1, "钾": 0.5, "钠": 1, "氯": 1, "钙": 0.5,
    "磷": 0.5, "镁": 0.5, "促甲状腺激素": 0.2, "游离T3": 0.5, "游离T4": 0.5,
    "载脂蛋白A1": 0.1, "载脂蛋白B": 0.1, "尿蛋白": 0.1, "尿潜血": 1,
    "尿葡萄糖": 1, "尿酮体": 0.5, "尿白细胞": 2, "尿比重": 0.01, "尿酸碱度": 0.5,
    "BMI": 0.5, "身高": 5, "体重": 5, "肺活量": 0.5, "体温": 0.5,
    "心率": 5, "呼吸频率": 2, "腰围": 5, "腰臀比": 0.05,
}

# ================= 加载模型 =================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ================= 知识库索引构建 =================
def load_or_build_index(kb_files, index_name):
    index_path = f"./faiss_{index_name}"
    if os.path.exists(index_path):
        print(f"Loading existing index from {index_path}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
        return vector_store
    else:
        print(f"Building index for {index_name}...")
        docs = []
        for file_path in kb_files:
            print(f"  Loading {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if "text" in data:
                        content = data["text"]
                    elif "test_name" in data and "description" in data:
                        content = data["description"]
                    elif "input" in data:
                        content = f"问题: {data['input']}\n答案: {data['output']}"
                    else:
                        continue
                    docs.append(Document(page_content=content, metadata={}))
        print(f"Total documents: {len(docs)}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(index_path)
        return vector_store

# 选择题知识库（原知识库）
choice_kb_files = [TEXTBOOK_FILE, ORIG_RANGE_FILE, TRAIN_FILE]
choice_vector_store = load_or_build_index(choice_kb_files, "choice_index")

# 自由文本知识库（增强版+教科书）
free_kb_files = [TEXTBOOK_FILE, ENHANCED_RANGE_FILE]
free_vector_store = load_or_build_index(free_kb_files, "free_enhanced_index")

# ================= SAE 权重加载 =================
def load_sae(layer):
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
    sae_weights = safetensors.torch.load_file(sae_path)
    W_enc = sae_weights['encoder.weight'].to(DEVICE).to(torch.float16)
    b_enc = sae_weights['encoder.bias'].to(DEVICE).to(torch.float16)
    return W_enc, b_enc

print("Loading SAE for layer 22...")
W_enc_choice, b_enc_choice = load_sae(CHOICE_LAYER)
print("Loading SAE for layer 16...")
W_enc_free, b_enc_free = load_sae(FREE_LAYER)

def sae_encode(hidden, W_enc, b_enc):
    z = hidden @ W_enc.T + b_enc
    topk_vals, topk_idx = torch.topk(z, SAE_TOP_K, dim=-1)
    features = torch.zeros_like(z)
    features.scatter_(-1, topk_idx, topk_vals)
    features = torch.relu(features)
    return features

# ================= 选择题相关函数 =================
CHOICE_FEATURE_INDICES = [28843, 11033, 4945, 32434, 9495, 30755, 1483, 13022, 18938, 28720][:CHOICE_FEATURE_COUNT]
CHOICE_FEATURE_SCORES = [0.4675, 0.3923, 0.3853, 0.3308, 0.3303, 0.3206, 0.3184, 0.3162, 0.2998, 0.2786][:CHOICE_FEATURE_COUNT]

def generate_choice_answer(question, knowledge):
    prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{knowledge}

Question:
{question}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    match = re.search(r'\b([A-E])\b', answer)
    return match.group(1) if match else ""

def get_choice_feature(question, knowledge):
    prompt = f"""You are a medical expert. Answer the following multiple-choice question using the provided knowledge.

Knowledge:
{knowledge}

Question:
{question}

Instructions:
- Output only the letter of the correct answer (e.g., "A").
- Do not include any extra text or explanation.

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[CHOICE_LAYER][0, -1, :]
    features = sae_encode(hidden.unsqueeze(0), W_enc_choice, b_enc_choice).squeeze(0).cpu().numpy()
    return features

def choice_mode(question):
    if not question.strip():
        return "请输入题目"
    retrieved = choice_vector_store.similarity_search(question, k=CHOICE_TOP_K)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    answer = generate_choice_answer(question, knowledge)
    feat = get_choice_feature(question, knowledge)
    risk = np.sum(feat[CHOICE_FEATURE_INDICES] * CHOICE_FEATURE_SCORES)
    level, icon, msg = get_confidence_level(risk, CHOICE_LOW_TH, CHOICE_HIGH_TH)
    return f"**模型答案：** {answer}\n\n**可信度：** {icon} {level}\n\n{msg}"

# ================= 自由文本辅助函数 =================
def round_value_to_step(name, value):
    if name not in ROUND_CONFIG:
        return value
    step = ROUND_CONFIG[name]
    rounded = round(value / step) * step
    if step < 1:
        decimal_places = len(str(step).split('.')[-1])
        rounded = round(rounded, decimal_places)
    else:
        rounded = int(rounded)
    return rounded

def parse_input_line(text):
    """解析单行输入，返回 (指标名, 数值) 或 (None, None)"""
    text = text.strip().replace('：', ' ').replace(':', ' ')
    indicators = list(ROUND_CONFIG.keys())
    indicators.sort(key=len, reverse=True)
    for ind in indicators:
        pattern = re.compile(rf'{re.escape(ind)}\s*([\d.]+)', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            value = float(match.group(1))
            return ind, value
    return None, None

def get_retrieved_knowledge(query_text, vector_store, top_k=3):
    docs = vector_store.similarity_search(query_text, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])

def generate_free_text(report_text, knowledge):
    prompt = f"你是医学专家。根据以下医学知识，对检验结果给出简短解读（1-2句话）并给出建议。\n\n知识：{knowledge}\n\n检验报告：{report_text}\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, temperature=0.0)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "解读：" in answer:
        answer = answer.split("解读：")[-1].strip()
    for suffix in ["注意", "202", "最终答案", "检验报告"]:
        if suffix in answer:
            answer = answer.split(suffix)[0].strip()
    if len(answer) > 200:
        answer = answer[:200] + "..."
    return answer

def get_free_feature(report_text, knowledge):
    prompt = f"检验报告：{report_text}\n\n知识：{knowledge}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[FREE_LAYER][0, -1, :]
    features = sae_encode(hidden.unsqueeze(0), W_enc_free, b_enc_free).squeeze(0).cpu().numpy()
    return features

def get_confidence_level(risk, low_th, high_th):
    if risk < low_th:
        return "低风险", "🟢", "模型对该回答较有信心，可参考。"
    elif risk <= high_th:
        return "中风险", "🟡", "模型信心一般，建议结合专业知识判断。"
    else:
        return "高风险", "🔴", "模型很可能出错，请勿直接采纳，务必核实。"

# ================= 自由文本SAE特征标定 =================
def calibrate_free_text_sae():
    print("Loading free text dev set for SAE calibration...")
    dev_samples = []
    with open(FREE_DEV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            dev_samples.append(json.loads(line))
    print(f"Dev samples: {len(dev_samples)}")
    
    dev_features = []
    dev_labels = []
    for sample in tqdm(dev_samples, desc="Extracting dev features"):
        report = sample["report"]
        true_label = sample["label"]
        retrieved = free_vector_store.similarity_search(report, k=FREE_TOP_K)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        feat = get_free_feature(report, knowledge)
        dev_features.append(feat)
        dev_labels.append(1 - true_label)
    X = np.array(dev_features)
    y = np.array(dev_labels)
    
    correct_mask = (y == 0)
    error_mask = (y == 1)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        raise ValueError("Dev set must contain both correct and error samples.")
    mean_c = X[correct_mask].mean(axis=0)
    mean_e = X[error_mask].mean(axis=0)
    std_c = X[correct_mask].std(axis=0) + 1e-8
    std_e = X[error_mask].std(axis=0) + 1e-8
    pooled_std = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled_std
    t_score[np.isnan(t_score)] = 0
    top_idx = np.argsort(t_score)[-FREE_FEATURE_COUNT:][::-1]
    top_weights = t_score[top_idx]
    print(f"Selected {FREE_FEATURE_COUNT} features for free text.")
    return top_idx.tolist(), top_weights.tolist()

print("Calibrating free text SAE features (this may take a few minutes)...")
free_top_idx, free_weights = calibrate_free_text_sae()
print("Free text SAE calibration done.")

# ================= 硬规则修正函数 =================
def correct_low_high(indicator, value, original_text):
    """根据数值直接返回正确的状态（偏低/正常/偏高），如果模型输出错误则返回修正文本"""
    thresholds = {
        "血糖": (3.9, 6.1, "偏低", "正常", "偏高"),
        "收缩压": (90, 119, "偏低", "正常", "偏高"),
        "舒张压": (60, 79, "偏低", "正常", "偏高"),
        "总胆固醇": (2.8, 5.2, "偏低", "正常", "偏高"),
        "高密度脂蛋白": (1.0, None, "偏低", "正常", None),
        "低密度脂蛋白": (None, 3.4, None, "正常", "偏高"),
        "甘油三酯": (None, 1.7, None, "正常", "偏高"),
        "血红蛋白": (120, 160, "偏低", "正常", "偏高"),
        "BMI": (18.5, 23.9, "偏低", "正常", "偏高"),
        "肌酐": (None, 104, None, "正常", "偏高"),
        "尿酸": (None, 428, None, "正常", "偏高"),
        "心率": (60, 100, "过缓", "正常", "过速"),
        "体温": (36.0, 37.0, "偏低", "正常", "发热"),
    }
    if indicator not in thresholds:
        return None
    low, high, low_desc, normal_desc, high_desc = thresholds[indicator]
    if low is not None and value < low:
        expected = low_desc
        if expected not in original_text and (expected == "偏低" and "低" not in original_text):
            return f"{indicator} {value}：{expected}。"
    elif high is not None and value > high:
        expected = high_desc
        if expected not in original_text and (expected == "偏高" and "高" not in original_text):
            return f"{indicator} {value}：{expected}。"
    elif low is not None and high is not None and low <= value <= high:
        expected = normal_desc
        if expected not in original_text:
            return f"{indicator} {value}：{expected}。"
    return None

def correct_common_errors(indicator, value, original_text):
    """修正常见误判（正常值被误判为异常）"""
    # 肌酐正常范围 44-104，若模型判为偏高则修正
    if indicator == "肌酐" and 44 <= value <= 104:
        if "偏高" in original_text or "升高" in original_text:
            return f"{indicator} {value}：正常。"
    # 体温正常
    if indicator == "体温" and 36.0 <= value <= 37.0:
        if "发热" in original_text or "升高" in original_text:
            return f"{indicator} {value}：正常。"
    # 心率正常
    if indicator == "心率" and 60 <= value <= 100:
        if "过缓" in original_text or "过速" in original_text:
            return f"{indicator} {value}：正常。"
    # 收缩压正常
    if indicator == "收缩压" and 90 <= value <= 119:
        if "高血压" in original_text or "高于正常" in original_text:
            return f"{indicator} {value}：正常。"
    # 舒张压正常
    if indicator == "舒张压" and 60 <= value <= 79:
        if "高血压" in original_text or "高于正常" in original_text:
            return f"{indicator} {value}：正常。"
    return None

# ================= 单轮自由文本模式（支持多种分隔符 + 硬规则修正） =================
def free_mode_single(report):
    if not report.strip():
        return "请输入报告内容"
    
    # 1. 统一分隔符为换行符（支持中英文逗号、分号、顿号、竖线、换行）
    separators = [',', '，', ';', '；', '#', '、', '|', '\n']
    normalized = report
    for sep in separators:
        if sep == '\n':
            continue
        normalized = normalized.replace(sep, '\n')
    # 2. 拆分成行，并去除空行
    lines = [line.strip() for line in normalized.split('\n') if line.strip()]
    if not lines:
        return "未检测到有效指标，请使用格式：指标名 数值，并用逗号或分号分隔。例如：血糖 5.2, 收缩压 115"
    
    interpretations = []
    risks = []
    for line in lines:
        indicator, value = parse_input_line(line)
        if indicator is None:
            interpretations.append(f"❌ 无法识别的指标：{line}，请使用格式：指标名 数值（如：血糖 5.2）")
            continue
        rounded_val = round_value_to_step(indicator, value)
        query_text = f"{indicator} {rounded_val}"
        knowledge = get_retrieved_knowledge(query_text, free_vector_store, top_k=3)
        interp = generate_free_text(query_text, knowledge)
        # 硬规则修正
        correction = correct_low_high(indicator, value, interp)
        if correction:
            interp = correction
        correction2 = correct_common_errors(indicator, value, interp)
        if correction2:
            interp = correction2
        interpretations.append(f"**{indicator} {value}**：{interp}")
        # 风险分数
        feat = get_free_feature(query_text, knowledge)
        risk = np.sum(feat[free_top_idx] * free_weights)
        risks.append(risk)
    
    if not interpretations:
        return "无法生成任何解读，请检查输入格式。"
    
    # 合并解读，去重
    combined = "\n\n".join(interpretations)
    seen = set()
    unique_lines = []
    for line in combined.split('\n'):
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    combined = "\n".join(unique_lines)
    
    avg_risk = np.mean(risks) if risks else 0.5
    level, icon, msg = get_confidence_level(avg_risk, FREE_LOW_TH, FREE_HIGH_TH)
    result = f"**模型解读：**\n{combined}\n\n**综合可信度：** {icon} {level}\n\n{msg}"
    return result

# ================= 多轮对话模式 =================
def format_dialog_history(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        else:
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def generate_multiturn_response(message, history):
    if not message.strip():
        return "", history
    indicator, value = parse_input_line(message)
    if indicator is not None:
        rounded_val = round_value_to_step(indicator, value)
        query_text = f"{indicator} {rounded_val}"
        knowledge = get_retrieved_knowledge(query_text, free_vector_store, top_k=3)
        system_prompt = f"你是医学专家。以下是相关的医学知识库参考：\n{knowledge}\n请基于知识和对话历史回答用户的问题。"
    else:
        knowledge = ""
        system_prompt = "你是医学专家。请回答用户关于医学健康的问题。"
    
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>" + format_dialog_history(messages)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=0.0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    if "<|eot_id|>" in response:
        response = response.split("<|eot_id|>")[0].strip()
    return response, history + [(message, response)]

# ================= Gradio 界面 =================
with gr.Blocks(title="医疗问答助手 - 可信度评估", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 医疗问答助手")
    gr.Markdown("基于 Llama-3.1-8B 和稀疏自编码器（SAE）的医学问答系统。")
    with gr.Tabs():
        with gr.TabItem("📖 医学知识自测（选择题）"):
            gr.Markdown("输入医学选择题（含选项），系统给出答案并评估可信度。")
            question_input = gr.Textbox(label="题目", lines=5, placeholder="例如：空腹血糖正常范围是多少？\nA. 3.9-6.1 mmol/L\nB. 6.2-7.0\nC. ...")
            choice_output = gr.Markdown(label="结果")
            submit_choice = gr.Button("提交")
            submit_choice.click(fn=choice_mode, inputs=question_input, outputs=choice_output)
        with gr.TabItem("🏥 检验报告解读（单轮）"):
            gr.Markdown("输入检验指标和数值，**推荐使用逗号或分号分隔**。格式：指标名 数值（如：血糖 5.2）。")
            report_input = gr.Textbox(label="报告内容", lines=5, placeholder="例如：\n血糖 5.2, 收缩压 115, 总胆固醇 4.0\n或：\n血糖 5.2；收缩压 115；总胆固醇 4.0")
            free_output = gr.Markdown(label="结果")
            submit_free = gr.Button("解读")
            submit_free.click(fn=free_mode_single, inputs=report_input, outputs=free_output)
        with gr.TabItem("💬 多轮对话"):
            gr.Markdown("可连续提问，基于医学知识进行多轮对话。")
            chatbot = gr.Chatbot(label="对话记录")
            msg = gr.Textbox(label="输入消息", lines=2)
            send_btn = gr.Button("发送")
            clear = gr.Button("清空对话")
            send_btn.click(generate_multiturn_response, [msg, chatbot], [msg, chatbot])
            msg.submit(generate_multiturn_response, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)