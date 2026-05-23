# 医学大模型 SAE 忠实性校验系统

基于 **Llama-3.1-8B-Instruct** 和 **稀疏自编码器 (Sparse Autoencoder, SAE)** 的医学问答忠实性检测系统。  
本仓库包含从数据准备、RAG 知识库构建、SAE 特征标定到最终演示的完整代码。

> **2026年5月更新**：新增基于 **HuatuoGPT-o1-8B** 和 **Llama-3.1-8B-UltraMedical** 的实验，并提供了**完全基于多轮对话的最终演示**（`gradio_ultramedical.py`），无需结构化输入，模型自动理解医学问题并给出风险等级。所有自由文本实验已统一使用 **196 条固定划分样本**（开发/测试各98条），结果稳定可复现。
> 目前核心代码存放在github上面的code_core文件夹，归档代码存放在code_archive文件夹
---

## 项目原理

1. **RAG 检索**：使用 FAISS 向量库 + 中文嵌入模型 (`BAAI/bge-base-zh-v1.5`) 从医学知识库中检索相关段落。
2. **LLM 生成**：基座模型根据检索到的知识生成答案（选择题字母）或解读（自由文本）。
3. **SAE 特征提取**：在模型生成过程中，捕获特定层最后一个 token 的隐藏状态，通过预训练 TopK SAE 编码为稀疏特征。
4. **特征标定与风险分数**：利用开发集（正确/错误样本）计算每个 SAE 特征的区分度（t‑statistic），选出 Top‑K 特征并加权求和，得到风险分数。分数越高表示答案越可能不忠实。最终根据统计分位数将风险分为低/中/高三档。

---

## 环境配置

### 硬件要求
- GPU: NVIDIA RTX 4090 (24GB+ 显存) 或同等算力
- 内存: 32GB+
- 硬盘: 100GB+ (存放模型、知识库、实验数据)

### 软件依赖

```bash
# 创建 conda 环境
conda create -n sae_env python=3.10 -y
conda activate sae_env

# 安装核心库
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.44.0
pip install accelerate datasets langchain langchain-community faiss-gpu sentence-transformers
pip install safetensors gradio scikit-learn
pip install huggingface_hub
```

> 若安装 `faiss-gpu` 失败，可改用 `faiss-cpu`（速度稍慢）。

### 模型与 SAE 权重

1. **基座模型**：`Llama-3.1-8B-Instruct`  
   从 ModelScope 下载（免申请）：
   ```bash
   modelscope download --model AI-ModelScope/Llama-3.1-8B-Instruct --local_dir ./models/Llama-3.1-8B-Instruct
   ```

2. **医疗微调模型**：
   - `HuatuoGPT-o1-8B`：基于 Llama-3.1-8B 进行医疗 CoT+PPO 微调，具备“先思考后回答”能力。  
     下载命令：
     ```bash
     huggingface-cli download FreedomIntelligence/HuatuoGPT-o1-8B --local-dir ./models/HuatuoGPT-o1-8B
     ```
   - `Llama-3.1-8B-UltraMedical`：清华大学微调的高性能医疗模型，知识丰富，适合多轮对话。  
     下载命令：
     ```bash
     huggingface-cli download TsinghuaC3I/Llama-3.1-8B-UltraMedical --local-dir ./models/Llama-3.1-8B-UltraMedical
     ```

3. **SAE 权重**：Llama Scope 的 TopK SAE（层0‑31，残差流，32K 特征）  
   下载示例（层22）：
   ```bash
   wget https://huggingface.co/OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x/resolve/main/Llama3_1-8B-Base-L22R-8x/L22R-8x.safetensors -O ./models/Llama-Scope/L22R-8x.safetensors
   ```
   完整列表见 `models/Llama-Scope/`。

---

## 知识库构建

项目使用以下知识库：
- **教科书知识库**：`./knowledge/medical_textbook_chunks.jsonl`（约 3.2 万条，从 MedQA 教科书切分）。
- **参考范围知识库**：`./knowledge/reference_ranges.jsonl`（约 50 条手工整理的常见检验指标参考范围）。
- **增强参考范围**：`./knowledge/reference_ranges_enhanced.jsonl`（离散化数值，用于自由文本实验）。
- **选择题额外知识库**：`./data/train.jsonl`（CMExam 训练集，5.2 万条选择题）。

首次运行任意脚本时，会自动构建 FAISS 索引并缓存。

---

## 实验脚本说明

所有 Python 脚本位于 `code/` 目录下。早期实验脚本已移至 `code/archive/`，不影响核心使用。

### 🔥 最终演示脚本

| 文件名 | 描述 |
|--------|------|
| `gradio_ultramedical.py` | **最终多轮对话演示**（基于 Llama-3.1-8B-UltraMedical）。用户输入任何医学问题（自然语言），模型回答并附带 SAE 风险等级（基于用户输入的最后一个 token）。无需特定格式，支持连续对话。 |

运行最终演示：
```bash
conda activate sae_env
cd ~/work/medical_sae_project
python code/gradio_ultramedical.py
# 浏览器打开 http://localhost:7860
```

---

### 📊 核心实验脚本（用于生成报告数据）

#### 原始 Llama-3.1-8B-Instruct 相关

| 文件名 | 实验内容 | 输出 |
|--------|----------|------|
| `quick_test_choice_all_layers.py` | 选择题全层扫描（0‑31层），Top‑10 特征加权和 | 各层 AUC 表格 |
| `choice_ablation_features.py` | 选择题（层20/22）特征数量消融（1‑50） | AUC vs 特征数量 |
| `choice_ablation_other.py` | 选择题检索数量 (1,3,5,7) 与随机种子消融 | AUC 矩阵 |
| `quick_test_all_layers_enhanced.py` | 自由文本全层扫描（增强知识库，98条，已归档但保留脚本） | 各层 AUC 表格 |
| `quick_test_all_layers_llama_rag.py` | **自由文本 RAG 全层扫描（196条固定划分）** | 各层 AUC 表格 |
| `free_text_ablation_advanced.py` | 自由文本 RAG 特征数量消融（层13/16，196条） | AUC vs 特征数量 |
| `free_text_ablation_topk.py` | **自由文本 RAG 检索数量消融**（默认层16、特征25；可通过修改代码中的 `LAYER` 和 `TOP_FEATURES` 参数用于层13等其他层） | AUC vs Top‑K |
| `analyze_risk_distribution.py` | 统计风险分数分布（均值、分位数等） | 终端输出 |
| `choice_accuracy_boost.py` | 选择题准确率优化（不同检索数量，加入同源训练集） | 准确率 |

#### HuatuoGPT-o1-8B 相关实验

| 文件名 | 实验内容 | 输出 |
|--------|----------|------|
| `quick_test_all_layers_huatuo_free.py` | 自由文本 RAG 全层扫描（196条固定划分） | 各层 AUC 表格 |
| `quick_test_all_layers_huatuo_norag_free.py` | 自由文本无 RAG 全层扫描（自然语言 prompt，196条） | 各层 AUC 表格 |
| `quick_test_choice_all_layers_huatuo.py` | 选择题 RAG 全层扫描（100条） | 各层 AUC 表格 |
| `free_text_ablation_huatuo_layers.py` | 自由文本 RAG 特征数量消融（层3、6、27，196条） | AUC vs 特征数量 |
| `free_text_ablation_topk_seed_huatuo.py` | 自由文本 RAG 检索数量消融（层27，固定特征10） | AUC vs Top‑K |
| `free_text_layer14_ablation_norag.py` | 自由文本无 RAG 特征数量消融（层14，196条） | AUC vs 特征数量 |
| `choice_ablation_features_huatuo_multilayer.py` | 选择题特征数量消融（层20,22,24,25,27） | AUC vs 特征数量 |
| `choice_ablation_topk_seed_huatuo_200.py` | 选择题检索与随机种子消融（层24，200样本） | AUC 矩阵 |

#### Llama-3.1-8B-UltraMedical 相关实验

| 文件名 | 实验内容 | 输出 |
|--------|----------|------|
| `ultra_free_full_layer_scan.py` | 自由文本无 RAG + 有 RAG 全层扫描（196条） | 各层 AUC 表格 |
| `quick_test_free_ultramedical_norag.py` | 自由文本无 RAG 全层扫描（196条） | 各层 AUC 表格 |
| `quick_test_choice_ultramedical_rag.py` | **选择题 RAG 全层扫描（100样本）** | 各层 AUC 表格 |
| `free_text_ablation_ultramedical_features.py` | 自由文本无 RAG 特征数量消融（层15，196条） | AUC vs 特征数量 |
| `free_text_ablation_ultra_norag_layer15.py` | 同（层15无RAG） | — |
| `free_text_ablation_ultra_rag_layer10.py` | 自由文本 RAG 特征数量消融（层10，196条） | AUC vs 特征数量 |
| `free_text_ablation_ultra_rag_layer10_topk.py` | 自由文本 RAG 检索数量消融（层10，固定特征15） | AUC vs Top‑K |
| `choice_ablation_features_ultramedical.py` | 选择题特征数量消融（层18、27） | AUC vs 特征数量 |
| `choice_ablation_topk_seed_ultra.py` | 选择题检索与随机种子消融（层27，200样本） | AUC 矩阵 |

#### 补充实验

| 文件名 | 实验内容 | 输出 |
|--------|----------|------|
| `supplementary_experiments.py` | SAE拦截效率、工具调用命中率、随机特征对比、不同嵌入模型对比 | 终端输出 |

---

### 🧪 辅助/历史脚本

早期实验、临时调试脚本及过渡演示已移至 `code/archive/`，不影响主流程。以下是各脚本的原始用途说明，可供参考：

| 脚本名称 | 用途 |
|----------|------|
| `rag_baseline.py` | 早期 RAG 基线（英文 MedXpertQA） |
| `rag_baseline_chinese.py` | 中文 RAG 基线（CMExam） |
| `zero_shot_chinese.py` | 零样本中文选择题对照 |
| `diagnose_rag_simple.py` | 简单诊断 RAG 检索效果 |
| `diagnose_rag.py` | 详细诊断 RAG 检索问题 |
| `diagnose_choice.py` | 诊断选择题答案提取 |
| `rag_huatuo.py` | 使用华佗知识库的早期 RAG 实验 |
| `report_rag_with_huatuo.py` | 华佗知识库自由文本解读原型 |
| `merge_all_knowledge.py` | 合并多个知识库文件的工具脚本 |
| `test_combined_kb.py` | 测试合并知识库的选择题准确率 |
| `test_combined_with_sae.py` | 测试合并知识库 + SAE 风险计算 |
| `test_textbook_rag.py` | 单独测试教科书知识库 RAG |
| `rag_full_knowledge_sae.py` | 第一版 RAG+SAE（全量知识库） |
| `rag_with_sae.py` | 早期 RAG+SAE 集成实验 |
| `rag_sae_attemp_with_olddata.py` | 选择题最终评估（旧版数据） |
| `rag_sae_analyze.py` | 分析 SAE 特征与风险分数关系 |
| `sae_feature_attribution.py` | 选择题特征筛选（加权和版本） |
| `run_rag_sae_pipeline.py` | 自由文本完整 pipeline（含逻辑回归） |
| `free_text_ablation.py` | 自由文本特征数量消融（49+49 样本） |
| `free_text_ablation_topk_seed.py` | 自由文本检索与随机种子消融（错误方法） |
| `free_text_ablation_topk_seed_enhanced.py` | 早期自由文本检索数量消融（层13，特征21），功能已被 `free_text_ablation_topk.py` 通过修改参数覆盖 |
| `quick_test_all_layers_huatuo_free_rag_norag.py` | HuatuoGPT‑o1 同时跑 RAG 和无 RAG 的整合脚本（已被拆分） |
| `choice_ablation_topk_seed_huatuo.py` | HuatuoGPT‑o1 选择题检索与种子消融（100 样本旧版） |
| `quick_test_choice_ultramedical_norag.py` | UltraMedical 选择题无 RAG 全层扫描（未使用） |
| `gradio_demo.py` | 早期 Gradio 演示（选择题 + 自由文本） |
| `gradio_demo_with_multiturn.py` | 早期 Gradio 多轮对话演示 |
| `gradio_new.py` | 中间过渡 Gradio 版本 |
| `generate_evaluation_excel.py` | 生成 Excel 供人工评估（模型未输出，废弃） |

> 这些脚本已不再使用，保留仅作为历史参考。如需运行，请手动移出 `archive/` 目录并注意依赖和环境要求。

---

## 数据文件说明

| 路径 | 内容 |
|------|------|
| `./data/train.jsonl` | CMExam 训练集（选择题格式，5.2 万条） |
| `./data/test.jsonl` | CMExam 测试集（选择题格式，6606 条） |
| `./data/free_text_dev.jsonl` | 自由文本开发集（98 条，含正确/错误标签） |
| `./data/free_text_test.jsonl` | 自由文本测试集（98 条） |
| `./knowledge/medical_textbook_chunks.jsonl` | 教科书知识块（3.2 万条） |
| `./knowledge/reference_ranges.jsonl` | 检验指标参考范围（约 50 条） |
| `./knowledge/reference_ranges_enhanced.jsonl` | 增强参考范围（离散化数值） |
| `./models/Llama-3.1-8B-Instruct/` | 基座模型 |
| `./models/HuatuoGPT-o1-8B/` | 医疗推理模型 |
| `./models/Llama-3.1-8B-UltraMedical/` | 清华医疗模型 |
| `./models/Llama-Scope/` | SAE 权重文件 |

---

## 如何复现主要实验结果

### 原始 Llama 模型

#### 1. 选择题最佳 AUC（层22，特征10，Top‑K=7）
```bash
python code/quick_test_choice_all_layers.py   # 输出各层 AUC
python code/choice_ablation_features.py       # 修改 LAYER = 22 运行
python code/choice_ablation_other.py          # 修改 LAYER=22, FEATURE_COUNT=10 运行
```

#### 2. 自由文本 RAG 全层扫描（196条）
```bash
python code/quick_test_all_layers_llama_rag.py
```

#### 3. 自由文本特征数量消融（层13/16）
```bash
python code/free_text_ablation_advanced.py   # 修改 LAYERS = [13,16]
```

#### 4. 自由文本检索数量消融（默认层16，特征25；如需测试层13，修改脚本中的 `LAYER=13` 和 `TOP_FEATURES=21`）
```bash
python code/free_text_ablation_topk.py
```

### HuatuoGPT-o1 模型

#### 1. 自由文本无 RAG 全层扫描
```bash
python code/quick_test_all_layers_huatuo_norag_free.py
```

#### 2. 自由文本 RAG 全层扫描
```bash
python code/quick_test_all_layers_huatuo_free.py
```

#### 3. 自由文本特征数量消融（层27）
```bash
python code/free_text_ablation_huatuo_layers.py   # 修改 LAYERS = [27]
```

#### 4. 选择题检索与种子消融（200样本）
```bash
python code/choice_ablation_topk_seed_huatuo_200.py
```

### Llama-3.1-8B-UltraMedical 模型

#### 1. 自由文本无 RAG 全层扫描
```bash
python code/quick_test_free_ultramedical_norag.py
```

#### 2. 自由文本 RAG 全层扫描（无 RAG + 有 RAG 一起）
```bash
python code/ultra_free_full_layer_scan.py
```

#### 3. 自由文本特征数量消融（无 RAG 层15）
```bash
python code/free_text_ablation_ultramedical_features.py   # 修改 LAYER = 15
```

#### 4. 自由文本 RAG 特征数量消融（层10）
```bash
python code/free_text_ablation_ultra_rag_layer10.py
```

#### 5. 选择题检索与种子消融（200样本）
```bash
python code/choice_ablation_topk_seed_ultra.py
```

#### 6. 选择题 RAG 全层扫描（100样本）
```bash
python code/quick_test_choice_ultramedical_rag.py
```

---

## 常见问题

**Q: 运行 `gradio_ultramedical.py` 时提示 `No module named 'gradio'`**  
A: 执行 `pip install gradio`。

**Q: 第一次运行脚本时卡在 `Loading weights...`**  
A: 首次运行会下载嵌入模型 `BAAI/bge-base-zh-v1.5`（约 1.3 GB），请耐心等待。后续会缓存。

**Q: 如何指定 GPU？**  
A: 在运行前设置 `export CUDA_VISIBLE_DEVICES=0`（替换为可用卡号）。

**Q: 最终演示支持哪些输入格式？**  
A: 支持自然语言，例如“血糖 6.1 mmol/L 正常吗？”或“我的血压 145/95 需要担心吗？”无需固定格式。

**Q: 为什么演示中有时会输出英文？**  
A: 模型偶尔会输出英文，但可以通过后续对话要求中文回答。实验表明，明确要求“请用中文回答”即可纠正。

---

## 引用

本工作基于以下开源成果：

- Llama-3.1-8B-Instruct (Meta)
- HuatuoGPT-o1-8B (FreedomIntelligence)
- Llama-3.1-8B-UltraMedical (TsinghuaC3I)
- Llama Scope SAE (OpenMOSS-Team)
- CMExam 数据集
- MedQA 教科书
- LangChain, FAISS, HuggingFace Transformers

---

## 许可证

本项目仅供学术研究使用。基座模型和 SAE 权重遵循其各自的许可证。
```
