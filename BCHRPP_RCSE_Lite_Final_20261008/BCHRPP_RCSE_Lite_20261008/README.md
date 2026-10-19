# BCHR++ RCSE-Lite: 简洁可靠的多模态情感证据推理（可运行工程）

**默认入口 `bchrpp/full_model.py` 为新版本 RCSE-Lite。** 保留原始 `legacy_model.py`、此前多机制 `enhanced_v1.py` 作为独立对照，而不是把两者的模块继续叠加到新模型。当前工程支持 **CMU-MOSI 与 CH-SIMSv2 的预提取特征回归**；另有合成数据上的分类接口单测。**本工程没有真实数据集的重新训练结果，不能声称已达到 SOTA 或精度提升。**

## 1. 单一创新主线

* **可靠性折扣的互补性（Reliability-discounted Complementarity）**：一条共享模态质量探针监督每个模态的预测质量。跨模态不一致只有在源模态与比较对象都相对可靠时，才可提高其证据优先级。
* **统一证据分配（Coupled Evidence Allocation）**：仅计算一次联合路由分布，将其同步用于直接模态汇聚和稀疏推理节点的权重/读出，避免重复学习独立的贡献、参与和权威权重。
* **结构化残差修正（Residual Structured Correction）**：保留轻量直接预测路径，稀疏关系仅对直接预测做有界校正；取消第二套完整预测头、额外 Transformer 和多阶段仲裁。

这里只声称**对所提供 BCHR++ 工程的架构重设计**。可靠性门控、互补性建模、注意力及残差预测分别存在既有工作；是否在国际论文意义上具有充分新颖性，仍需系统查新、理论论证与真实实验支持。

## 2. 主要文件

| 文件 | 内容 |
| --- | --- |
| `bchrpp/full_model.py` | RCSE-Lite 完整模型（新默认） |
| `losses/bchrpp_loss.py` | `task + unimodal + calibration` 三项损失 |
| `bchrpp/enhanced_v1.py` | 上一轮 SOTA 综合优化版，用于对照 |
| `bchrpp/legacy_model.py` | 原始 BCHR++ 模型，用于对照与历史权重读取 |
| `losses/enhanced_v1_loss.py` | 旧优化版独立损失，防止复用新损失导致不兼容 |
| `scripts/train_mosi.py` | MOSI 回归训练与测试入口 |
| `scripts/train_simsv2.py` | CH-SIMSv2 回归训练与测试入口 |
| `scripts/run_ablations.py` | 多随机种子、消融及原版对照 |
| `scripts/eval_acc7.py` | 根据 checkpoint 元信息自动选择**匹配的**模型版本 |
| `scripts/profile_models.py` | 模型参数量和同平台 CPU 前向计时 |
| `tests/test_smoke.py` | 遮罩、单模态、前向反向、分类、权重加载等检查 |
| `docs/METHOD_AND_PROTOCOL.md` | 设计动机、精确公式、原创性边界、实验方案 |
| `docs/TEST_REPORT.md` | 已完成的本地测试与环境说明 |

## 3. 安装与立即自测

Python 3.10+，PyTorch 2.4+。依赖见 `requirements.txt`。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/train.py --dataset MOSI --config configs/bchrpp_base.yaml --smoke-test
python scripts/profile_models.py --dims 128
```

## 4. 用自己的预提取特征训练

指定的 `.pkl` 需包含 `train/valid/test` 分割；每条数据包括文本/音频/视觉特征、回归标签等字段；具体要求见原数据加载器，**不包含数据下载**。

**CMU-MOSI**：

```bash
python scripts/train_mosi.py \
  --data-path /path/to/MOSI/Processed/unaligned_50.pkl \
  --checkpoint checkpoints/mosi_rcse_lite_best.pt \
  --epochs 60 --batch-size 24 --hidden-dim 128 \
  --queries 4 --topk 4 --lr 0.0003 --seed 222 --amp
```

**CH-SIMSv2**：

```bash
python scripts/train_simsv2.py \
  --data-path /path/to/SIMSv2/Processed/unaligned.pkl \
  --checkpoint checkpoints/simsv2_rcse_lite_best.pt \
  --epochs 60 --batch-size 16 --hidden-dim 128 \
  --queries 4 --topk 4 --lr 0.0003 --seed 222 --amp
```

默认采用验证集 MAE 选择 checkpoint（不使用测试集调参）。自动推断三种输入特征维度；`--amp` 仅在 CUDA 生效。新模型保存 `version=bchrpp_rcse_lite_v1`。

## 5. 消融与多随机种子

```bash
python scripts/run_ablations.py \
  --dataset mosi \
  --data-path /path/to/MOSI/Processed/unaligned_50.pkl \
  --output-dir results/mosi_rcse \
  --seeds 222,128,2026 \
  --modes lite,no_reliability,no_complementarity,no_temporal,dense_graph,no_refinement,no_quality_aux
```

三套网络的历史工程对照：

```bash
python scripts/run_ablations.py \
  --dataset mosi --data-path /path/to/MOSI/Processed/unaligned_50.pkl \
  --output-dir results/model_comparison --seeds 222,128,2026 \
  --modes legacy,enhanced_v1,lite
```

旧优化模型使用其独立 loss 和构造参数；为严格复现历史运行，还应显式对齐其原始学习率、辅助损失权重、topk、特征版本及训练轮数。参数更少、synthetic smoke 成功，并不意味着真实精度更好。

## 6. Checkpoint 的归属（很重要）

* `checkpoints/mosi_best.pt` 是**原始模型的历史权重**，并非 RCSE-Lite 的已训练权重。
* RCSE-Lite、新旧优化版、原始版之间**不能直接互载**。独立模型定义与 `version` 防止错误加载。
* 可通过 `python scripts/eval_acc7.py --checkpoint ... --data-path ...` 检验相应 MOSI checkpoint。
* MOSI、MOSEI、CH-SIMS、CH-SIMSv2、IEMOCAP、MELD、CFI-V2 与 UDIVA **不可直接视为已打通相同训练接口**；当前只核验了 MOSI 与 CH-SIMSv2 的具体回归管线。

## 7. 研究诚信与命名

新模块是**学习到的关联网络**，不是可辨识的因果超图；局部深度卷积不是完整 Mamba。参考 CICA、GraphMamba、KGCF-GA、MSBF、CASDF、CAMS-AMSS 等工作的动机，但没有直接复制其特征提取器、公开训练权重或将其原论文结果标为本模型结果。新方法能否在标准评估下实现高精度，必须通过真实数据、相同设置、多随机种子和缺失模态鲁棒性实验证实。
