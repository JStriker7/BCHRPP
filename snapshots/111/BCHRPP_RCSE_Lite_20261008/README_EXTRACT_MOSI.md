# 在 MOSI 上跑通 RCSE-Lite（离线特征 → 训练 → 评测）

本目录新增了三个**新文件**，用于把 `extract/` 的离线特征提取与 `BCHRPP_RCSE_Lite` 的
RCSE-Lite 模型串起来。**没有修改任何既有文件**：

| 新文件 | 作用 |
| --- | --- |
| `datasets/mosi_extract.py` | 读取 `extract/extract_mosi_features.py` 输出的 `.pt`，校验格式并转换为 RCSE-Lite 所需的 `[B,T,D]` + mask 输入 |
| `scripts/train_mosi_extract.py` | 用 `bchrpp.full_model.BCHRPP` + `losses.bchrpp_loss.compute_loss` 训练，用 `metrics.evaluation.mosi_metrics` 输出 **acc7 / mae / mse / corr / acc2 / f1** |
| `scripts/smoke_mosi_extract.py` | 合成数据端到端冒烟测试（CPU、无需真实 MOSI/GPU），用于先验证桥接可用 |
| `run_mosi_extract.sh` | 端到端一键脚本（检查布局 → 提取特征 → 校验 → 训练+评测） |

## 1. 为什么需要这个桥接

- 项目自带的 `scripts/train_mosi.py → train_regression.py` 读的是 `.pkl`
  （`unaligned_50.pkl`，字段 `text/audio/vision/regression_labels`）。
- 但 `extract/extract_mosi_features.py` 产出的是 `.pt`，且结构不同：
  `train/valid/test` 各是一个样本列表，每个样本是词级对齐的
  `text(1024)/audio(512)/visual(768)` 特征 + `*_valid` 掩码 + 标量 `label`。

所以直接用 `train_mosi.py` 会因格式不符失败；`datasets/mosi_extract.py` 就是两者之间的适配层。

> **注意**：只用 `extract/extract_mosi_features.py`（格式 `bhgm_sent_crc_mosi_v1`，
> 维度 text=1024 / audio=512 / visual=768）。**不要**混入 `en_extract.py` 或
> `extract_mosi_reference_av.py` 的特征——它们的文本是“在线 DeBERTa”占位（维度 1），
> 与 RCSE-Lite 的固定线性投影不兼容。

## 2. 服务器依赖

```bash
python -m pip install -r requirements.txt   # numpy / PyYAML / torch
# 提取脚本额外需要（按实际版本安装）：
python -m pip install transformers librosa opencv-python pillow
# 若 --clap-model 指向 .pt 权重（默认 630k-audioset-best.pt），还需 laion-clap：
python -m pip install laion-clap
```

GPU 上请安装与显卡匹配的 CUDA 版 PyTorch（`torch>=2.4`）。

## 3. 运行步骤

最省事的方式是编辑 `run_mosi_extract.sh` 顶部的路径后 `bash run_mosi_extract.sh`。
下面给出等价的手工命令（`DATA_ROOT`/`RUN_ROOT`/`$MODELS/...` 需替换为服务器真实路径）。

### 3.0 先做一次合成数据冒烟测试（强烈建议，CPU 即可，几秒钟）

```bash
python scripts/smoke_mosi_extract.py
```

看到 `SMOKE OK; test= {...'acc7'... 'mae'...}` 即说明桥接代码（数据适配 + 模型 + 损失 + 指标）
在服务器上可用，之后再跑真实提取与训练。

### 3.1 先只读检查数据布局（不加载任何模型）

```bash
python extract/extract_mosi_features.py --data-dir "$DATA_ROOT" --check-layout
```

确认输出的 `audio_found`、`visual_found`、`word_timestamps_found` 覆盖正常。

### 3.2 离线提取特征（原样 extract 代码）

```bash
python extract/extract_mosi_features.py \
  --data-dir "$DATA_ROOT" \
  --output "$RUN_ROOT/features/mosi_deberta_clip_clap.pt" \
  --cache-dir "$RUN_ROOT/cache/mosi_offline" \
  --deberta-model "$MODELS/deberta-v3-large" \
  --clip-model "$MODELS/clip-vit-large-patch14" \
  --clap-model "$MODELS/630k-audioset-best.pt" \
  --clap-bert-model "$MODELS/bert-base-uncased" \
  --clap-roberta-model "$MODELS/roberta-base" \
  --clap-roberta-tokenizer "$MODELS/roberta-large" \
  --clap-bart-model "$MODELS/bart-base"
```

建议首次先 `--max-samples 1` 做一次冒烟（不能用于训练，只截取标签前几行）。

### 3.3 校验特征文件

```bash
python scripts/train_mosi_extract.py --features "$RUN_ROOT/features/mosi_deberta_clip_clap.pt" --check-features
```

### 3.4 训练 + 评测（输出 acc7 / mae / …）

```bash
python scripts/train_mosi_extract.py \
  --features "$RUN_ROOT/features/mosi_deberta_clip_clap.pt" \
  --checkpoint "$RUN_ROOT/runs/mosi_extract_rcse_lite_best.pt" \
  --epochs 60 --batch-size 24 --hidden-dim 128 \
  --queries 4 --topk 4 --lr 0.0003 --seed 222 --amp
```

训练用**验证集 MAE** 选最佳 checkpoint（`--patience` 默认 15 轮早停），结束后对**测试集只评一次**。
结果打印在 stdout，同时写入 `checkpoint` 同名的 `.json` 文件。

### 3.5 用已保存权重只评测试集（可选）

```bash
python scripts/train_mosi_extract.py \
  --features "$RUN_ROOT/features/mosi_deberta_clip_clap.pt" \
  --checkpoint "$RUN_ROOT/runs/mosi_extract_rcse_lite_best.pt" --evaluate
```

## 4. 结果字段

`metrics.evaluation.mosi_metrics` 返回（`result.json` 里 `test` / `validation` 同结构）：

- `mae`：连续预测值上的平均绝对误差（**未截断**）
- `mse`：均方误差
- `corr`：Pearson 相关系数
- `acc2` / `f1`：二分类（正/负，含 0 口径）准确率 / 加权 F1
- `acc7`：预测先裁剪到 `[-3,3]` 再四舍五入到 7 档后与标签一致的占比

## 5. 可选参数（消融 / 调参）

`scripts/train_mosi_extract.py` 支持与 `train_regression.py` 一致的消融开关：
`--no-reliability`、`--no-complementarity`、`--no-temporal`、`--no-sparse-graph`、
`--no-refinement`、`--label-reweight`，以及 `--aux-weight` / `--confidence-weight` 等。
多随机种子可复跑不同 `--seed` 取平均。
