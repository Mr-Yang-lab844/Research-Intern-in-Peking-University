```markdown
# 面向事实性幻觉检测的医学报告解读助手

基于 **Llama-3.1-8B-Instruct** 和 **稀疏自编码器 (Sparse Autoencoder, SAE)** 的医学问答事实性幻觉检测系统。  
本仓库包含从数据准备、RAG 知识库构建、SAE 特征标定到最终演示的完整代码。

> **2026年5月更新**：新增基于 **HuatuoGPT-o1-8B** 和 **Llama-3.1-8B-UltraMedical** 的实验，并提供了**完全基于多轮对话的最终演示**（`gradio_ultramedical.py`），无需结构化输入，模型自动理解医学问题并给出风险等级。所有自由文本实验已统一使用 **196 条固定划分样本**（开发/测试各98条），结果稳定可复现。

---

## 项目原理

1. **RAG 检索**：使用 FAISS 向量库 + 中文嵌入模型 (`BAAI/bge-base-zh-v1.5`) 从医学知识库中检索相关段落。
2. **LLM 生成**：基座模型根据检索到的知识生成答案（选择题字母）或解读（自由文本）。
3. **SAE 特征提取**：在模型生成过程中，捕获特定层最后一个 token 的隐藏状态，通过预训练 TopK SAE 编码为稀疏特征。
4. **特征标定与风险分数**：利用开发集（正确/错误样本）计算每个 SAE 特征的区分度（t‑statistic），选出 Top‑K 特征并加权求和，得到风险分数。分数越高表示答案越可能产生事实性幻觉。最终根据统计分位数将风险分为低/中/高三档。

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

在本仓库存储时，核心脚本在 `code_core` 文件夹，早期实验脚本已归档在 `code_archive` 文件夹。  
建议下载运行时将所有 Python 脚本放于 `code/` 目录下。早期实验脚本可以移至 `code/archive/`，不影响核心使用。

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
| `quick_test_choice_all_layers.py` | 选择题全层扫描（0‑31层，200条），Top‑10 特征加权和 | 各层 AUC 表格 |
| `choice_ablation_features.py` | 选择题（层20/22）特征数量消融（1‑50） | AUC vs 特征数量 |
| `choice_ablation_other.py` | 选择题检索数量 (1,3,5,7) 与随机种子消融 | AUC 矩阵 |
| `quick_test_all_layers_enhanced.py` | 自由文本全层扫描（增强知识库，98条，已归档但保留脚本） | 各层 AUC 表格 |
| `quick_test_all_layers_llama_rag.py` | 自由文本 RAG 全层扫描（196条固定划分） | 各层 AUC 表格 |
| `free_text_ablation_advanced.py` | 自由文本 RAG 特征数量消融（层13/16，196条） | AUC vs 特征数量 |
| `free_text_ablation_topk.py` | 自由文本 RAG 检索数量消融（默认层16、特征25；可通过修改代码中的 `LAYER` 和 `TOP_FEATURES` 参数用于层13等其他层） | AUC vs Top‑K |
| `analyze_risk_distribution.py` | 统计风险分数分布（均值、分位数等） | 终端输出 |
| `choice_accuracy_boost.py` | 选择题准确率优化（不同检索数量，加入同源训练集） | 准确率 |

#### HuatuoGPT-o1-8B 相关实验

| 文件名 | 实验内容 | 输出 |
|--------|----------|------|
| `quick_test_all_layers_huatuo_free.py` | 自由文本 RAG 全层扫描（196条固定划分） | 各层 AUC 表格 |
| `quick_test_all_layers_huatuo_norag_free.py` | 自由文本无 RAG 全层扫描（自然语言 prompt，196条） | 各层 AUC 表格 |
| `quick_test_choice_all_layers_huatuo.py` | 选择题 RAG 全层扫描（200条） | 各层 AUC 表格 |
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
| `quick_test_choice_ultramedical_rag.py` | 选择题 RAG 全层扫描（200样本） | 各层 AUC 表格 |
| `free_text_ablation_ultramedical_features.py` | 自由文本无 RAG 特征数量消融（层15，196条） | AUC vs 特征数量 |
| `free_text_ablation_ultra_norag_layer15.py` | 同（层15无RAG） | — |
| `free_text_ablation_ultra_rag_layer10.py` | 自由文本 RAG 特征数量消融（层10，196条） | AUC vs 特征数量 |
| `free_text_ablation_ultra_rag_layer10_topk.py` | 自由文本 RAG 检索数量消融（层10，固定特征15） | AUC vs Top‑K |
| `choice_ablation_features_ultramedical.py` | 选择题特征数量消融（层18、27） | AUC vs 特征数量 |
| `choice_ablation_topk_seed_ultra.py` | 选择题检索与随机种子消融（层27，200样本） | AUC 矩阵 |

### 🧪 补充实验

| 文件名 | 实验内容 | 输出 |
|--------|----------|------|
| `supplementary_experiments.py` | **完整补充实验集**（含阈值分析、SAE拦截效率、工具调用命中率、随机特征对比、不同嵌入模型对比、自由文本原始隐藏状态基线） | 终端输出，生成缓存文件用于其他脚本 |
| `supplementary_experiments(1).py` | **基线对比与多 token 特征探索**（选择题与自由文本的原始隐藏状态、SAE 无标定 L1、随机特征基线，以及各 token 位置 AUC、多 token 组合、衰减系数、选择性位置最大值、新思路探索） | 终端输出，完整复现报告中的基线表格和多 token 融合结果 |

**补充实验运行说明**：
- 运行 `supplementary_experiments.py` 前需先完成主实验，确保缓存文件（如 `free_weights.npy`、`free_test_feats.npy`）已生成。
- `supplementary_experiments(1).py` 可独立运行，但依赖相同的缓存文件；它会计算并打印选择题和自由文本的所有基线 AUC（与报告表格一致），并执行多 token 特征探索实验。

> **注意**：文件名中的括号为中文全角字符，在 Linux 系统中可能需要转义或用引号括起来。建议使用 `python "supplementary_experiments(1).py"` 运行。

---

### 🧪 辅助/历史脚本

早期实验、临时调试脚本及过渡演示已移至 `code/archive/`，不影响主流程。详细列表见仓库。

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
