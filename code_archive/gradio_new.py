#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio 演示界面（HuatuoGPT-o1 版本，无 RAG）
- 仅保留检验报告解读（单轮）和多轮对话
- SAE 层27，特征10
- 模型直接基于自身知识生成解读
- SAE 风险分数基于无 RAG 的开发集标定，阈值使用百分位数（30%, 70%）
- 采用智能清洗：按行分割 + 中文字符比例 >=0.5 保留，彻底去除英文行
- 优化换行显示：使用 <br><br> 分隔各指标解读
- 可选总结：当指标数量 > 5 时自动生成异常统计总结
"""

import json
import re
import torch
import numpy as np
import safetensors.torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

# ================= 配置 =================
MODEL_PATH = "./models/HuatuoGPT-o1-8B"
TOKENIZER_PATH = "./models/Llama-3.1-8B-Instruct"   # 兼容 tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAE_TOP_K = 32
FREE_LAYER = 27
FREE_FEATURE_COUNT = 10
FREE_DEV_FILE = "./data/free_text_dev.jsonl"   # 无 RAG 开发集，格式：{"report": "...", "label": 1/0}（1=正确）

# 指标步长配置（用于数值舍入，保留）
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

# 硬规则阈值（用于修正模型输出，可选）
THRESHOLDS = {
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

# ================= 加载模型 =================
print("Loading HuatuoGPT-o1-8B...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ================= SAE 权重加载 =================
def load_sae(layer):
    sae_path = f"./models/Llama-Scope/L{layer}R-8x.safetensors"
    sae_weights = safetensors.torch.load_file(sae_path)
    W_enc = sae_weights['encoder.weight'].to(DEVICE).to(torch.float16)
    b_enc = sae_weights['encoder.bias'].to(DEVICE).to(torch.float16)
    return W_enc, b_enc

print(f"Loading SAE for layer {FREE_LAYER}...")
W_enc_free, b_enc_free = load_sae(FREE_LAYER)

def sae_encode(hidden, W_enc, b_enc):
    z = hidden @ W_enc.T + b_enc
    topk_vals, topk_idx = torch.topk(z, SAE_TOP_K, dim=-1)
    features = torch.zeros_like(z)
    features.scatter_(-1, topk_idx, topk_vals)
    features = torch.relu(features)
    return features

def get_free_feature_no_rag(report_text):
    """无 RAG：直接使用 prompt '检验报告：...\n\n解读：' 提取最后一个 token 的隐藏状态"""
    prompt = f"检验报告：{report_text}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[FREE_LAYER][0, -1, :]
    features = sae_encode(hidden.unsqueeze(0), W_enc_free, b_enc_free).squeeze(0).cpu().numpy()
    return features

def generate_answer_no_rag(report_text):
    """无 RAG：直接生成解读"""
    prompt = f"检验报告：{report_text}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output

# ================= 后处理清洗函数（按行 + 中文字符比例过滤）=================
def clean_single_output(indicator, value, raw_output):
    """
    智能清洗：优先提取 ## Final Response 或 解读：后的内容，
    然后按行分割，保留中文字符占比 >= 0.5 的行，最后合并并添加标点。
    """
    # 1. 提取候选文本
    if "## Final Response" in raw_output:
        candidate = raw_output.split("## Final Response")[-1].strip()
    else:
        if "解读：" in raw_output:
            candidate = raw_output.split("解读：")[-1].strip()
        else:
            if "## Thinking" in raw_output:
                parts = raw_output.split("## Thinking")
                if len(parts) > 1:
                    after = parts[1]
                    # 找第一个空行或句号作为截断
                    match = re.search(r'\n\s*\n|。', after)
                    if match:
                        candidate = after[match.end():].strip()
                    else:
                        candidate = after.strip()
                else:
                    candidate = raw_output.strip()
            else:
                candidate = raw_output.strip()
    
    # 2. 按换行分割，逐行判断
    lines = candidate.split('\n')
    kept_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 计算中文字符比例
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', line)
        if len(chinese_chars) == 0:
            continue
        total_chars = len(line)
        # 中文字符比例 >= 0.5 则保留
        if len(chinese_chars) / total_chars >= 0.5:
            kept_lines.append(line)
    
    # 3. 如果没有保留任何行，尝试取候选文本的第一个中文句子
    if not kept_lines:
        sentences = re.split(r'[。！？]', candidate)
        for sent in sentences:
            sent = sent.strip()
            if re.search(r'[\u4e00-\u9fff]', sent):
                kept_lines = [sent]
                break
    
    # 4. 合并结果
    result = '。'.join(kept_lines).strip()
    if not result:
        result = "无法生成有效解读。"
    
    # 5. 确保以句号结尾
    if result and result[-1] not in '。！？':
        result += '。'
    
    return f"{indicator} {value}：{result}"

# ================= SAE 特征标定（无 RAG）=================
def calibrate_free_text_sae_no_rag():
    print("Loading free text dev set (no RAG) for SAE calibration...")
    dev_samples = []
    with open(FREE_DEV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            dev_samples.append(json.loads(line))
    print(f"Dev samples: {len(dev_samples)}")
    
    dev_features = []
    dev_labels = []   # 1 表示模型回答错误（不忠实），0 表示正确
    for sample in tqdm(dev_samples, desc="Extracting dev features (no RAG)"):
        report = sample["report"]
        true_label = sample["label"]   # 1=正确, 0=错误
        # 转换为 0=正确,1=错误
        error_label = 1 - true_label
        feat = get_free_feature_no_rag(report)
        dev_features.append(feat)
        dev_labels.append(error_label)
    
    X = np.array(dev_features)
    y = np.array(dev_labels)   # 1=错误
    correct_mask = (y == 0)
    error_mask = (y == 1)
    if correct_mask.sum() == 0 or error_mask.sum() == 0:
        raise ValueError("开发集必须同时包含正确和错误样本")
    mean_c = X[correct_mask].mean(axis=0)
    mean_e = X[error_mask].mean(axis=0)
    std_c = X[correct_mask].std(axis=0) + 1e-8
    std_e = X[error_mask].std(axis=0) + 1e-8
    pooled_std = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled_std
    t_score[np.isnan(t_score)] = 0
    top_idx = np.argsort(t_score)[-FREE_FEATURE_COUNT:][::-1]
    top_weights = t_score[top_idx]
    
    # 计算开发集的风险分数分布（用于百分位数阈值）
    all_risks = []
    for feat in X:
        risk = np.sum(feat[top_idx] * top_weights)
        all_risks.append(risk)
    all_risks = np.array(all_risks)
    low_th = np.percentile(all_risks, 30)   # 30% 分位数
    high_th = np.percentile(all_risks, 70)  # 70% 分位数
    print(f"Calibration done. Selected {FREE_FEATURE_COUNT} features.")
    print(f"Risk percentiles: 30%={low_th:.4f}, 70%={high_th:.4f}")
    
    return top_idx.tolist(), top_weights.tolist(), low_th, high_th

print("Calibrating free text SAE features (no RAG, this may take a few minutes)...")
free_top_idx, free_weights, FREE_LOW_TH, FREE_HIGH_TH = calibrate_free_text_sae_no_rag()
print("Calibration done.")

# ================= 辅助函数 =================
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

def get_confidence_level(risk, low_th, high_th):
    if risk < low_th:
        return "低风险", "🟢", "模型对该回答较有信心，可参考。"
    elif risk <= high_th:
        return "中风险", "🟡", "模型信心一般，建议结合专业知识判断。"
    else:
        return "高风险", "🔴", "模型很可能出错，请勿直接采纳，务必核实。"

# ================= 硬规则修正 =================
def correct_low_high(indicator, value, original_text):
    if indicator not in THRESHOLDS:
        return None
    low, high, low_desc, normal_desc, high_desc = THRESHOLDS[indicator]
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
    if indicator == "肌酐" and 44 <= value <= 104:
        if "偏高" in original_text or "升高" in original_text:
            return f"{indicator} {value}：正常。"
    if indicator == "体温" and 36.0 <= value <= 37.0:
        if "发热" in original_text or "升高" in original_text:
            return f"{indicator} {value}：正常。"
    if indicator == "心率" and 60 <= value <= 100:
        if "过缓" in original_text or "过速" in original_text:
            return f"{indicator} {value}：正常。"
    if indicator == "收缩压" and 90 <= value <= 119:
        if "高血压" in original_text or "高于正常" in original_text:
            return f"{indicator} {value}：正常。"
    if indicator == "舒张压" and 60 <= value <= 79:
        if "高血压" in original_text or "高于正常" in original_text:
            return f"{indicator} {value}：正常。"
    return None

# ================= 单轮自由文本模式（无 RAG）=================
def free_mode_single(report):
    if not report.strip():
        return "请输入报告内容"
    # 统一分隔符
    separators = [',', '，', ';', '；', '#', '、', '|', '\n']
    normalized = report
    for sep in separators:
        if sep == '\n':
            continue
        normalized = normalized.replace(sep, '\n')
    lines = [line.strip() for line in normalized.split('\n') if line.strip()]
    if not lines:
        return "未检测到有效指标，请使用格式：指标名 数值，并用逗号或分号分隔。例如：血糖 5.2, 收缩压 115"

    interpretations = []
    risks = []
    # 用于总结的统计
    status_list = []   # 存储每个指标的状态（"正常"/"偏高"/"偏低"/"未知"）
    for line in lines:
        indicator, value = parse_input_line(line)
        if indicator is None:
            interpretations.append(f"❌ 无法识别的指标：{line}")
            continue
        rounded_val = round_value_to_step(indicator, value)
        query_text = f"{indicator} {rounded_val}"
        # 生成原始输出（无 RAG）
        raw_output = generate_answer_no_rag(query_text)
        # 清洗
        cleaned = clean_single_output(indicator, value, raw_output)
        # 可选的硬规则修正
        correction = correct_low_high(indicator, value, cleaned)
        if correction:
            cleaned = correction
        correction2 = correct_common_errors(indicator, value, cleaned)
        if correction2:
            cleaned = correction2
        interpretations.append(cleaned)
        # 计算 SAE 风险（无 RAG）
        feat = get_free_feature_no_rag(query_text)
        risk = np.sum(feat[free_top_idx] * free_weights)
        risks.append(risk)
        
        # 提取状态用于总结（从清洗后的文本中提取关键词）
        if "正常" in cleaned or "正常。" in cleaned:
            status_list.append("正常")
        elif "偏高" in cleaned or "升高" in cleaned:
            status_list.append("偏高")
        elif "偏低" in cleaned or "降低" in cleaned:
            status_list.append("偏低")
        else:
            status_list.append("未知")

    if not interpretations:
        return "无法生成任何解读，请检查输入格式。"

    # 使用 <br><br> 分隔各指标解读，确保换行可见
    combined = "<br><br>".join(interpretations)
    
    # 可选总结：当指标数量 > 5 时生成简单总结
    if len(interpretations) > 5:
        total = len(status_list)
        normal_cnt = status_list.count("正常")
        high_cnt = status_list.count("偏高")
        low_cnt = status_list.count("偏低")
        unknown_cnt = status_list.count("未知")
        summary = f"<br><br>**📊 汇总**：共 {total} 项指标，其中 {normal_cnt} 项正常，{high_cnt} 项偏高，{low_cnt} 项偏低"
        if unknown_cnt > 0:
            summary += f"，{unknown_cnt} 项无法判断"
        summary += "。"
        combined += summary

    # 去重（简单按行去重，针对 HTML 标签做特殊处理，这里只去重完全相同的行）
    # 由于使用了 <br><br> 分隔，去重可能破坏结构，因此不做全局去重，仅去除连续的重复段落（可选）
    # 简单起见，保留原样，用户反馈中未出现大量重复，不做额外去重
    
    avg_risk = np.mean(risks) if risks else 0.5
    level, icon, msg = get_confidence_level(avg_risk, FREE_LOW_TH, FREE_HIGH_TH)
    result = f"**模型解读：**<br>{combined}<br><br>**综合可信度：** {icon} {level}<br>{msg}"
    return result

# ================= 多轮对话（无 RAG）=================
def generate_multiturn_response(message, history):
    if not message.strip():
        return "", history
    indicator, value = parse_input_line(message)
    if indicator is not None:
        rounded_val = round_value_to_step(indicator, value)
        query_text = f"{indicator} {rounded_val}"
        system_prompt = "你是医学专家。请基于你的医学知识回答用户的问题。"
        user_msg = query_text
    else:
        system_prompt = "你是医学专家。请回答用户关于医学健康的问题。"
        user_msg = message
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": user_msg})
    chat_text = f"System: {system_prompt}\n"
    for msg in messages:
        chat_text += f"User: {msg['content']}\nAssistant: "
    inputs = tokenizer(chat_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=0.0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    response = re.sub(r'<\|.*?\|>', '', response)
    response = re.sub(r'Cutting Knowledge Date:.*?Today Date:.*?\n', '', response)
    return response, history + [(message, response)]

# ================= Gradio 界面 =================
with gr.Blocks(title="医疗报告解读助手 - 可信度评估 (无 RAG)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 医疗报告解读助手 (HuatuoGPT-o1, 无 RAG)")
    gr.Markdown("基于 HuatuoGPT-o1-8B 和稀疏自编码器（SAE）的报告解读系统，直接使用模型自身医学知识生成解读，并提供可信度等级（风险阈值基于开发集百分位数）。")
    with gr.Tabs():
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