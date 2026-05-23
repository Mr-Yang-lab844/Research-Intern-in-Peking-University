#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio 演示界面：医学知识自测（选择题）+ 检验报告解读（自由文本）
- 选择题：层22、特征10、检索 Top‑K=20，输出答案 + 可信度等级
- 自由文本：层16、特征11、检索 Top‑K=3，支持多指标（必须用分隔符）
- 医学内容预检：输入不含医学关键词时返回警告
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

# ----------------------------- 配置 -----------------------------
MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXTBOOK_FILE = "./knowledge/medical_textbook_chunks.jsonl"
REFERENCE_FILE = "./knowledge/reference_ranges.jsonl"
TRAIN_FILE = "./data/train.jsonl"

EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
SAE_TOP_K = 32

CHOICE_LAYER = 22
CHOICE_FEATURE_COUNT = 10
CHOICE_TOP_K = 20            # 统一使用20
FREE_LAYER = 16
FREE_FEATURE_COUNT = 11
FREE_TOP_K = 3

DEV_SET = "./data/free_text_dev.jsonl"

# 阈值（根据统计结果）
CHOICE_LOW_TH = 5.0
CHOICE_HIGH_TH = 5.8
FREE_LOW_TH = 0.25
FREE_HIGH_TH = 0.36

# ----------------------------- 医学关键词预检（请替换为您自己的大词典）-----------------------------
# 此处省略完整的 MEDICAL_KEYWORDS 字典，您应使用之前生成的大词典
# 请从历史代码中复制完整的 MEDICAL_KEYWORDS 并粘贴到下面
MEDICAL_KEYWORDS = {
    # 基础代谢/血糖血脂（新增细分）
    "血糖", "空腹血糖", "餐后血糖", "糖化血红蛋白", "糖耐量", "胰岛素", "C肽", "糖化白蛋白",
    "血糖波动", "胰岛素抵抗", "酮症酸中毒", "低血糖",
    "血压", "收缩压", "舒张压", "脉压差", "平均动脉压", "高血压危象", "低血压",
    "胆固醇", "总胆固醇", "甘油三酯", "高密度脂蛋白", "低密度脂蛋白", "载脂蛋白A", "载脂蛋白B", "同型半胱氨酸", "脂蛋白a",
    
    # 血常规/凝血（新增细分）
    "血红蛋白", "血小板", "白细胞", "红细胞", "血常规", "贫血", "缺铁性贫血", "巨幼细胞性贫血", "溶血性贫血",
    "中性粒细胞", "淋巴细胞", "单核细胞", "嗜酸性粒细胞", "嗜碱性粒细胞", "红细胞压积", "平均红细胞体积",
    "平均红细胞血红蛋白量", "平均红细胞血红蛋白浓度", "血小板压积", "血小板分布宽度", "凝血酶原时间",
    "活化部分凝血活酶时间", "纤维蛋白原", "D-二聚体", "出血时间", "凝血时间", "凝血因子", "血栓弹力图",
    
    # 肝肾功能/电解质（新增细分）
    "谷丙转氨酶", "谷草转氨酶", "肌酐", "尿素氮", "尿酸", "胆红素", "直接胆红素", "间接胆红素",
    "白蛋白", "球蛋白", "白球比", "碱性磷酸酶", "谷氨酰转肽酶", "乳酸脱氢酶", "血肌酐", "尿肌酐",
    "肝功能", "肾功能", "肝衰竭", "肾衰竭", "尿毒症", "肝纤维化", "肝硬化",
    "电解质", "钾", "钠", "氯", "钙", "磷", "镁", "碳酸氢根", "渗透压", "低血钾", "高血钾", "低血钙",
    
    # 甲状腺/内分泌激素（新增细分）
    "甲状腺", "T3", "T4", "TSH", "游离T3", "游离T4", "促甲状腺激素", "甲状腺球蛋白", "甲状腺抗体",
    "皮质醇", "促肾上腺皮质激素", "肾上腺素", "去甲肾上腺素", "多巴胺", "性激素", "孕酮", "睾酮", "雌激素",
    "泌乳素", "生长激素", "甲状旁腺激素", "维生素D", "叶酸", "维生素B12", "促性腺激素", "黄体生成素", "卵泡刺激素",
    "内分泌失调", "甲状腺功能亢进", "甲状腺功能减退", "甲状腺结节",
    
    # 尿常规/粪便（新增细分）
    "尿常规", "尿蛋白", "尿糖", "尿酮体", "尿潜血", "尿白细胞", "尿胆红素", "尿胆原", "尿比重",
    "尿酸碱度", "尿管型", "尿结晶", "尿微量白蛋白", "尿沉渣", "24小时尿蛋白定量",
    "粪便常规", "粪便隐血", "粪便白细胞", "寄生虫", "粪便培养", "粪便虫卵检查", "黑便", "柏油样便",
    
    # 肿瘤标志物（新增细分）
    "AFP", "CEA", "CA199", "CA125", "CA153", "PSA", "游离PSA", "SCC", "CYFRA21-1", "NSE",
    "CA724", "CA242", "铁蛋白", "β2微球蛋白", "肿瘤标志物筛查", "癌前病变",
    
    # 感染/炎症指标（新增细分）
    "C反应蛋白", "CRP", "降钙素原", "PCT", "血沉", "ESR", "白细胞介素", "IL-6", "IL-10", "肿瘤坏死因子",
    "抗体", "抗原", "核酸检测", "PCR", "ELISA", "酶联免疫吸附试验",
    "新冠", "甲流", "乙流", "支原体", "衣原体", "结核", "乙肝", "丙肝", "艾滋", "梅毒", "淋病", "衣原体",
    "布鲁氏菌病", "恙虫病", "登革热", "手足口病", "带状疱疹", "水痘", "麻疹", "百日咳", "猩红热", "流行性腮腺炎",
    "细菌培养", "药敏试验", "耐药性", "抗生素", "抗病毒药物",
    
    # 体格/营养指标（新增细分）
    "BMI", "体重指数", "腰围", "臀围", "腰臀比", "体脂率", "肌肉量", "基础代谢", "身高", "体重",
    "体成分分析", "营养不良", "肥胖", "消瘦", "水肿", "皮下脂肪厚度","体温"
    
    # 心肺功能（新增细分）
    "肺活量", "FEV1", "FVC", "血氧饱和度", "SpO2", "心率", "体温", "呼吸频率", "心功能", "肺功能",
    "动脉血气", "氧分压", "二氧化碳分压", "肺动脉压", "心输出量", "射血分数", "潮气量", "残气量",
    "哮喘", "慢阻肺", "COPD", "肺炎", "肺结核", "肺栓塞", "气胸", "胸腔积液", "心力衰竭", "心肌缺血",
    
    # 影像学/器械检查（新增细分）
    "心电图", "ECG", "动态心电图", "Holter", "B超", "彩超", "CT", "增强CT", "MRI", "增强MRI", "X光", "DR", "CR",
    "超声", "胃镜", "肠镜", "胶囊内镜", "支气管镜", "膀胱镜", "宫腔镜", "腹腔镜", "关节镜", "椎间孔镜",
    "病理检查", "活检", "穿刺", "造影", "冠脉造影", "PET-CT", "PET-MRI", "骨密度", "双能X线",
    "脑电图", "EEG", "动态脑电图", "肌电图", "EMG", "诱发电位", "眼底镜", "裂隙灯", "角膜地形图", "眼压计",
    "经颅多普勒", "TCD", "超声心动图", "心脏彩超", "骨扫描", "心肌灌注显像", "肾动态显像",
    
    # 常见疾病（新增细分+罕见病相关）
    "糖尿病", "1型糖尿病", "2型糖尿病", "糖尿病并发症", "高血压", "原发性高血压", "继发性高血压",
    "高血脂", "高尿酸", "痛风", "肝炎", "甲肝", "乙肝", "丙肝", "脂肪肝", "肝硬化", "肝癌",
    "肾炎", "肾小球肾炎", "肾病综合征", "肾衰竭", "肺炎", "支气管炎", "慢阻肺", "哮喘",
    "胃炎", "慢性胃炎", "胃溃疡", "十二指肠溃疡", "肠炎", "溃疡性结肠炎", "克罗恩病",
    "冠心病", "心梗", "心肌梗死", "脑梗", "脑梗死", "脑出血", "中风", "脑卒",
    "癌症", "肿瘤", "肺癌", "胃癌", "肠癌", "肝癌", "乳腺癌", "宫颈癌", "卵巢癌", "前列腺癌",
    "甲亢", "甲减", "白血病", "淋巴瘤", "抑郁症", "焦虑症", "精神分裂症", "强迫症",
    "类风湿关节炎", "RA", "强直性脊柱炎", "AS", "系统性红斑狼疮", "SLE", "干燥综合征", "SS",
    "帕金森病", "PD", "阿尔茨海默病", "AD", "多发性硬化", "MS", "癫痫", "脑瘫",
    "白癜风", "痤疮", "玫瑰痤疮", "脂溢性皮炎", "神经性皮炎", "带状疱疹", "毛囊炎", "湿疹", "银屑病", "荨麻疹",
    "股骨头坏死", "滑膜炎", "腱鞘炎", "腰椎间盘突出症", "颈椎病", "骨折", "脱位", "骨质疏松",
    "多囊卵巢综合征", "PCOS", "子宫内膜异位症", "卵巢囊肿", "宫颈癌", "卵巢癌", "月经不调",
    "手足口病", "川崎病", "百日咳", "麻疹", "水痘", "小儿肺炎", "发育迟缓", "佝偻病",
    "结膜炎", "角膜炎", "葡萄膜炎", "视网膜病变", "视神经炎", "白内障", "青光眼", "近视", "远视", "散光",
    "鼻炎", "鼻窦炎", "咽炎", "扁桃体炎", "中耳炎", "喉炎", "喉癌", "鼻息肉",
    "龋齿", "牙周炎", "口腔溃疡", "牙周脓肿", "智齿冠周炎", "牙髓炎", "根尖周炎",
    "罕见病", "渐冻症", "肌萎缩侧索硬化", "ALS", "血友病", "地中海贫血", "苯丙酮尿症",
    
    # 全身/局部症状（新增细分）
    "头痛", "偏头痛", "头晕", "眩晕", "恶心", "呕吐", "腹痛", "胃痛", "腹痛", "腹泻", "便秘",
    "咳嗽", "咳痰", "干咳", "湿咳", "发热", "寒战", "低热", "高热", "弛张热", "稽留热",
    "乏力", "消瘦", "水肿", "凹陷性水肿", "心悸", "胸闷", "胸痛", "压榨性胸痛",
    "关节痛", "肌肉痛", "皮疹", "瘙痒", "黄疸", "出血", "瘀斑", "紫癜",
    "麻木", "抽搐", "惊厥", "震颤", "晕厥", "昏迷", "嗜睡", "昏睡", "意识模糊",
    "呼吸困难", "喘息", "咯血", "呕血", "黑便", "鲜血便", "里急后重",
    "尿频", "尿急", "尿痛", "排尿困难", "遗尿", "尿失禁", "血尿",
    "闭经", "崩漏", "痛经", "白带异常", "外阴瘙痒",
    "耳鸣", "耳痛", "听力下降", "鼻塞", "流涕", "咽痛", "声音嘶哑",
    "视力模糊", "视力下降", "视物变形", "失明", "口干", "口苦", "口臭",
    "牙龈出血", "牙痛", "口腔溃疡", "咽喉异物感",
    
    # 医疗核心术语（新增细分）
    "症状", "诊断", "鉴别诊断", "治疗", "药物治疗", "手术治疗", "放疗", "化疗", "靶向治疗", "免疫治疗", "透析", "血液透析", "腹膜透析",
    "检验", "检查", "指标", "参考范围", "正常值", "异常", "阳性", "阴性", "临界值", "定性", "定量",
    "康复", "护理", "体检", "健康体检", "疫苗", "接种", "预防接种", "急救", "清创", "缝合", "换药", "消毒",
    "病历", "医嘱", "处方", "会诊", "转诊", "住院", "出院", "门诊", "急诊", "病房", "ICU", "CCU", "手术室",
    "麻醉", "全身麻醉", "局部麻醉", "硬膜外麻醉", "腰麻", "静脉麻醉", "术后护理", "并发症", "后遗症", "预后", "复发", "缓解", "治愈",
    
    # 妇科/儿科/男科专属（新增细分）
    "月经", "痛经", "备孕", "怀孕", "妊娠", "产检", "胎心", "胎动", "羊水", "孕酮", "hcg", "人绒毛膜促性腺激素",
    "新生儿", "婴幼儿", "儿童", "发育迟缓", "佝偻病", "矮小症", "性早熟", "遗尿症",
    "遗精", "勃起功能障碍", "早泄", "前列腺炎", "前列腺增生", "睾丸炎", "附睾炎", "精索静脉曲张",
    
    # 口腔/眼科/耳鼻喉（新增细分）
    "龋齿", "牙周炎", "口腔溃疡", "牙周脓肿", "智齿冠周炎", "牙髓炎", "根尖周炎", "口腔CT", "口腔全景片",
    "眼压", "近视", "远视", "散光", "白内障", "青光眼", "结膜炎", "角膜炎", "葡萄膜炎", "视网膜脱离", "视神经萎缩",
    "鼻炎", "鼻窦炎", "过敏性鼻炎", "鼻息肉", "咽炎", "扁桃体炎", "喉炎", "喉癌", "中耳炎", "内耳炎", "听力测试", "前庭功能检查",
    
    # 中医相关（新增）
    "中医", "脉象", "舌苔", "舌质", "阴虚", "阳虚", "气虚", "血虚", "痰湿", "湿热", "血瘀", "气滞",
    "经络", "穴位", "足三里", "关元", "气海", "中脘", "太冲", "合谷", "针灸", "艾灸", "拔罐", "刮痧", "中药", "方剂",
    
    # 康复相关（新增）
    "康复训练", "物理治疗", "运动治疗", "言语治疗", "作业治疗", "认知训练", "平衡训练", "步态训练",
    "针灸", "按摩", "推拿", "理疗", "热敷", "冷敷", "电疗", "磁疗", "超声波治疗", "康复评估",
    
    # 急救相关（新增）
    "心肺复苏", "CPR", "除颤", "AED", "止血", "包扎", "固定", "搬运", "吸氧", "气管插管", "静脉输液",
    "急救药物", "肾上腺素", "阿托品", "多巴胺", "硝酸甘油", "止血带", "急救箱",
    
    # 药物相关（新增类别+具体药名）
    "降压药", "氨氯地平", "硝苯地平", "缬沙坦", "氯沙坦", "美托洛尔", "卡托普利",
    "降糖药", "二甲双胍", "格列齐特", "格列美脲", "胰岛素", "阿卡波糖", "西格列汀",
    "抗生素", "阿莫西林", "头孢克肟", "头孢地尼", "阿奇霉素", "罗红霉素", "左氧氟沙星", "莫西沙星",
    "退烧药", "布洛芬", "对乙酰氨基酚", "阿司匹林",
    "抗炎药", "双氯芬酸钠", "塞来昔布", "泼尼松", "地塞米松",
    "化疗药", "紫杉醇", "顺铂", "奥沙利铂", "甲氨蝶呤", "环磷酰胺",
    "止咳化痰药", "氨溴索", "沙丁胺醇", "布地奈德", "异丙托溴铵",
    "胃药", "奥美拉唑", "兰索拉唑", "泮托拉唑", "雷贝拉唑", "多潘立酮", "莫沙必利",
    "泻药", "乳果糖", "聚乙二醇", "开塞露",
    
    # 实验室相关（新增）
    "培养基", "菌落计数", "药敏试验", "核酸扩增", "PCR", "ELISA", "酶联免疫吸附试验",
    "流式细胞术", "基因检测", "染色体核型分析", "生化检测", "免疫检测", "微生物检测", "细胞培养",
    "离心机", "显微镜", "分光光度计", "电泳仪", "培养箱"
}
def contains_medical_keyword(text):
    if not text:
        return False
    text_lower = text.lower()
    for kw in MEDICAL_KEYWORDS:
        if kw in text or kw.lower() in text_lower:
            return True
    return False

# ----------------------------- 加载模型 -----------------------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ----------------------------- 知识库索引 -----------------------------
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
                    elif "test_name" in data:
                        content = f"{data['test_name']}：正常范围 {data['normal_range']} {data['unit']}。{data['description']}"
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

choice_kb_files = [TEXTBOOK_FILE, REFERENCE_FILE, TRAIN_FILE]
choice_vector_store = load_or_build_index(choice_kb_files, "choice_index")
free_kb_files = [TEXTBOOK_FILE, REFERENCE_FILE]
free_vector_store = load_or_build_index(free_kb_files, "free_index")

# ----------------------------- SAE 权重 -----------------------------
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

# ----------------------------- 特征标定 -----------------------------
def calibrate_choice_features():
    with open("./data/test.jsonl", "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f]
    random.seed(42)
    samples = random.sample(all_samples, 100)
    dev_samples = samples[:50]
    dev_features = []
    dev_labels = []
    for s in tqdm(dev_samples, desc="Calibrating choice"):
        q = s["input"]
        true = s["output"]
        retrieved = choice_vector_store.similarity_search(q, k=CHOICE_TOP_K)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        pred = generate_choice_answer(q, knowledge)
        is_correct = (pred == true)
        feat = get_choice_feature(q, knowledge)
        dev_features.append(feat)
        dev_labels.append(0 if is_correct else 1)
    X = np.array(dev_features)
    y = np.array(dev_labels)
    correct_mask = (y == 0)
    error_mask = (y == 1)
    mean_c = X[correct_mask].mean(axis=0)
    mean_e = X[error_mask].mean(axis=0)
    std_c = X[correct_mask].std(axis=0) + 1e-8
    std_e = X[error_mask].std(axis=0) + 1e-8
    pooled = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled
    t_score[np.isnan(t_score)] = 0
    top_idx = np.argsort(t_score)[-CHOICE_FEATURE_COUNT:][::-1]
    top_weights = t_score[top_idx]
    return top_idx, top_weights

def calibrate_free_features():
    dev_samples = []
    with open(DEV_SET, "r", encoding="utf-8") as f:
        for line in f:
            dev_samples.append(json.loads(line))
    dev_features = []
    dev_labels = []
    for s in tqdm(dev_samples, desc="Calibrating free"):
        report = s["report"]
        true_label = s["label"]
        retrieved = free_vector_store.similarity_search(report, k=FREE_TOP_K)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        feat = get_free_feature(report, knowledge)
        dev_features.append(feat)
        dev_labels.append(0 if true_label == 1 else 1)
    X = np.array(dev_features)
    y = np.array(dev_labels)
    correct_mask = (y == 0)
    error_mask = (y == 1)
    mean_c = X[correct_mask].mean(axis=0)
    mean_e = X[error_mask].mean(axis=0)
    std_c = X[correct_mask].std(axis=0) + 1e-8
    std_e = X[error_mask].std(axis=0) + 1e-8
    pooled = np.sqrt(std_c**2 + std_e**2)
    t_score = np.abs(mean_e - mean_c) / pooled
    t_score[np.isnan(t_score)] = 0
    top_idx = np.argsort(t_score)[-FREE_FEATURE_COUNT:][::-1]
    top_weights = t_score[top_idx]
    return top_idx, top_weights

# ----------------------------- 特征提取 -----------------------------
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

def get_free_feature(report, knowledge):
    prompt = f"检验报告：{report}\n\n知识：{knowledge}\n\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[FREE_LAYER][0, -1, :]
    features = sae_encode(hidden.unsqueeze(0), W_enc_free, b_enc_free).squeeze(0).cpu().numpy()
    return features

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

# 稳定版自由文本生成函数（简单 prompt + 后处理）
def generate_free_text(report, knowledge):
    prompt = f"你是医学专家。根据以下医学知识，对检验结果给出简短解读（1-2句话）并给出建议。\n\n知识：{knowledge}\n\n检验报告：{report}\n解读："
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "解读：" in answer:
        answer = answer.split("解读：")[-1].strip()
    # 后处理：删除常见幻觉前缀
    if "检验报告：" in answer:
        answer = answer.split("检验报告：")[0].strip()
    # 限制长度，只保留前100字
    if len(answer) > 100:
        answer = answer[:100] + "..."
    return answer

# ----------------------------- Gradio 界面逻辑 -----------------------------
print("Calibrating choice features (this may take a minute)...")
choice_top_idx, choice_weights = calibrate_choice_features()
print("Choice features ready.")
print("Calibrating free text features...")
free_top_idx, free_weights = calibrate_free_features()
print("Free features ready.")

def get_confidence_level(risk, low_th, high_th):
    if risk < low_th:
        return "低风险", "🟢", "模型对该回答较有信心，可参考。"
    elif risk <= high_th:
        return "中风险", "🟡", "模型信心一般，建议结合专业知识判断。"
    else:
        return "高风险", "🔴", "模型很可能出错，请勿直接采纳，务必核实。"

def choice_mode(question):
    if not question.strip():
        return "请输入题目"
    if not contains_medical_keyword(question):
        return "⚠️ 您输入的内容未检测到明确的医学信息。本系统仅处理医学相关问题，请提供正确的医学选择题（包含医学术语）。"
    retrieved = choice_vector_store.similarity_search(question, k=CHOICE_TOP_K)
    knowledge = "\n\n".join([doc.page_content for doc in retrieved])
    answer = generate_choice_answer(question, knowledge)
    feat = get_choice_feature(question, knowledge)
    risk = np.sum(feat[choice_top_idx] * choice_weights)
    level, icon, msg = get_confidence_level(risk, CHOICE_LOW_TH, CHOICE_HIGH_TH)
    return f"**模型答案：** {answer}\n\n**可信度：** {icon} {level}\n\n{msg}"

def free_mode(report):
    if not report.strip():
        return "请输入报告内容"
    if not contains_medical_keyword(report):
        return "⚠️ 您输入的内容未检测到明确的医学信息。本系统仅处理医学检验报告或医疗相关问题，请提供正确的医学内容（如血糖、血压等指标）。"
    
    # 多指标拆分（禁止空格，只使用显式分隔符）
    separators = [',', '，', ';', '；', '#', '、', '|', '\n']
    normalized = report
    for sep in separators:
        if sep == '\n':
            continue
        normalized = normalized.replace(sep, '\n')
    lines = [line.strip() for line in normalized.split('\n') if line.strip()]
    if not lines:
        lines = [report]
    
    interpretations = []
    for idx, item in enumerate(lines):
        retrieved = free_vector_store.similarity_search(item, k=FREE_TOP_K)
        knowledge = "\n\n".join([doc.page_content for doc in retrieved])
        interp = generate_free_text(item, knowledge)
        # 添加指标序号和原文
        interpretations.append(f"**{idx+1}. {item}**\n{interp}")
    
    combined_interpretation = "\n\n".join(interpretations)
    
    # 风险分数仍基于原始完整报告
    retrieved_full = free_vector_store.similarity_search(report, k=FREE_TOP_K)
    knowledge_full = "\n\n".join([doc.page_content for doc in retrieved_full])
    feat = get_free_feature(report, knowledge_full)
    risk = np.sum(feat[free_top_idx] * free_weights)
    level, icon, msg = get_confidence_level(risk, FREE_LOW_TH, FREE_HIGH_TH)
    
    result = f"**模型解读：**\n{combined_interpretation}\n\n**可信度：** {icon} {level}\n\n{msg}"
    return result

# ----------------------------- 构建 Gradio 界面 -----------------------------
with gr.Blocks(title="医疗问答助手 - 可信度评估") as demo:
    gr.Markdown("# 医疗问答助手")
    gr.Markdown("基于 Llama-3.1-8B 和稀疏自编码器（SAE）的医学问答系统，提供答案并给出可信度等级。")
    with gr.Tabs():
        with gr.TabItem("📖 医学知识自测（选择题）"):
            gr.Markdown("输入医学选择题（含选项），系统给出答案并评估可信度。**请确保题目包含医学术语。**")
            question_input = gr.Textbox(label="题目", placeholder="例如：空腹血糖正常范围是多少？\nA. 3.9-6.1 mmol/L\nB. 6.2-7.0\nC. ...", lines=5)
            choice_output = gr.Markdown(label="结果")
            submit_choice = gr.Button("提交")
            submit_choice.click(fn=choice_mode, inputs=question_input, outputs=choice_output)
        with gr.TabItem("🏥 检验报告解读（自由文本）"):
            gr.Markdown("输入检验报告中的指标和数值。**多个指标请使用分隔符：换行、逗号（中英文）、分号（中英文）、井号、顿号、竖线。** 禁止使用空格分隔。")
            report_input = gr.Textbox(label="报告内容", placeholder="例如：血糖：7.2 mmol/L\n总胆固醇：6.1 mmol/L", lines=4)
            free_output = gr.Markdown(label="结果")
            submit_free = gr.Button("解读")
            submit_free.click(fn=free_mode, inputs=report_input, outputs=free_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)