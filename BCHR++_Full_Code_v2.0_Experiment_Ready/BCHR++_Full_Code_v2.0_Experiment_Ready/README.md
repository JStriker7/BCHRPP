# BCHR++ 的 MOSI 实验说明

本项目使用老师提供的 BCHRPP 模型框架运行 MOSI 实验。特征提取只使用外层同级目录中的原版 `extract/extract_mosi_features.py`，不修改其中的预处理、对齐或特征提取方法。该脚本离线生成词级的 DeBERTa-v3-large 文本特征（1024 维）、LAION-CLAP 音频特征（512 维）和 CLIP ViT-L/14 视觉特征（768 维）；图像输入分辨率以实际加载的 CLIP processor 配置为准。训练适配层先将三种特征投影到老师模型的隐藏维度，再输出一个连续情感分数。

必须区分“跑通老师给的代码”和“完整复现方法文档”：当前老师代码是简化框架，并未实现文档 3.1–3.14 节的全部机制。例如，反事实贡献和认知竞争模块尚未参与模型前向计算，名为 `CausalHypergraphTransformer` 的模块实际包装的是标准 Transformer Encoder。因此，本项目得到的指标只能称为**老师所给代码框架的 MOSI 实验结果**，不能称为完整 BCHR++ 方法的复现结果。

## 新服务器的目录隔离

把 `extract/` 原样复制到一个新的实验目录；老师的模型代码也复制到该目录，不覆盖服务器上其他项目。MOSI 原始数据、预训练模型、提取缓存、特征文件、训练权重和日志应分别放置。提取脚本自带的 `/mnt/data8tb/...` 等默认路径即使碰巧可用，也应在新实验中用命令行参数逐项显式指定，以便记录资源来源。

这台服务器已确认 `/mnt/data2t` 可写且空间充足，建议把新环境、包缓存、HF 缓存、临时文件、代码和实验输出放在 `/mnt/data2t/wjy/bchrpp_mosi_20261014`。`/` 仅剩约 98 GB，不要将大型模型或环境安装到 `~/SYX`、Conda base 或已有的 `muxlink` 环境。已有 MOSI 数据及模型位于其他用户 `wxr` 的路径，只能在确认有权用于本实验后只读使用；不要在那些目录中写入缓存、特征或权重。

先在服务器执行以下**只读检查**，不需要安装 PyTorch：

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
DATA_ROOT=/mnt/data8tb/wxr/MMML/data/MOSI
head -n 3 "$DATA_ROOT/label.csv"
for d in "$DATA_ROOT/wav" "$DATA_ROOT/processed/audio" "$DATA_ROOT/Raw" "$DATA_ROOT/processed/frames"; do
  if [ -d "$d" ]; then
    echo "存在: $d"
    find -L "$d" -maxdepth 4 -type f \( -iname '*.wav' -o -iname '*.mp4' -o -iname '*.npy' \) -print -quit
  else
    echo "不存在: $d"
  fi
done
for p in \
  /mnt/data8tb/wxr/bhgm_sent_crc_mosi_fixed/model/deberta-v3-large \
  /mnt/data26t/model/clip-vit-large-patch14 \
  /mnt/data26t/model/bert-base-uncased \
  /mnt/data26t/model/roberta-base \
  /mnt/data26t/model/roberta-large \
  /mnt/data26t/model/bart-base; do
  if [ -d "$p" ]; then
    echo "模型目录: $p"
    find -L "$p" -maxdepth 4 -type f \( -name config.json -o -name preprocessor_config.json -o -name spm.model -o -name vocab.txt -o -name vocab.json -o -name merges.txt -o -name model.safetensors -o -name pytorch_model.bin \) -printf '%p %s bytes\n' | head -n 20
  else
    echo "缺少模型目录: $p"
  fi
done
stat -c '%n %s bytes' /mnt/data8tb/wxr/MMML/checkpoints/630k-audioset-best.pt
```

文件夹可读不代表模型完整；尤其应确认 DeBERTa/CLIP 的 `config.json`、权重和 tokenizer/processor 文件在可直接加载的目录中，而不只是嵌套的 HF 缓存快照。还要检查 MOSI 的音频目录，`Raw` 存在不能证明 WAV 已迁移。GPU 型号确认后再选择匹配的 PyTorch 版本；安装依赖后才能运行下面的 `--check-layout`（脚本在参数解析前会导入 NumPy 和 PyTorch）。

下载或安装之前，先确认服务器的可用数据盘、Python、GPU 驱动、Conda 环境和 PyTorch CUDA 版本。离线 `.pt` 格式的 CLAP 提取分支还依赖 `laion_clap`，并要求本地 BERT、RoBERTa、BART 资源。不要把 `en_extract.py` 或参考 AV 流程的特征混进来；它们的特征维度和数据约定不同。

下面的命令中，`DATA_ROOT` 是 MOSI 数据根目录，`MODELS` 是预训练模型目录，`RUN_ROOT` 是本次实验的独立输出目录。应先按新服务器的实际路径设置这三个变量。第一段命令在包含 `extract/` 的目录下运行；训练命令在内层 `BCHR++_Full_Code_v2.0_Experiment_Ready/` 目录下运行。

## 先检查数据布局

```bash
python extract/extract_mosi_features.py \
  --data-dir "$DATA_ROOT" --check-layout
```

检查输出中的 `audio_found`、`visual_found`、`word_timestamps_found`，并核对 `label.csv` 的样本划分与音视频文件覆盖情况。原脚本在缺少有效词级时间戳时使用均匀时间区间，这是原方法已有的回退行为；须记录时间戳覆盖数量，不能把均匀对齐描述为真实词级强制对齐。如果某个样本的整种模态缺失，训练适配层会拒绝该特征文件，防止无声地影响结果。

## 原样提取离线三模态特征

确认数据完整、模型文件和依赖齐备后，使用一套全新的缓存及输出路径。下面的 `$MODELS/...` 只是示意；如果模型分散在不同目录，直接将每个参数分别写为上述实际路径，不需要复制到同一个模型目录：

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

首次加载前建议将 `--max-samples 1` 与独立的 `--output`、`--cache-dir` 配合做一次小规模冒烟检查；这一步不能用于训练，因为只截取标签文件的前几行，可能缺失验证/测试划分。若更改模型或提取参数，重新使用新的缓存和输出目录，不复用旧缓存。原脚本遇到样本提取错误会最终报错；不要拿不完整的数据进行训练。

## 训练和评测

进入内层模型代码目录后运行：

```bash
python -m unittest discover -s tests -v
python -m scripts.train --features "$RUN_ROOT/features/mosi_deberta_clip_clap.pt" --check-features
python -m scripts.train --dataset MOSI --config configs/bchrpp_mosi_100ep.yaml \
  --features "$RUN_ROOT/features/mosi_deberta_clip_clap.pt" \
  --run-dir "$RUN_ROOT/runs/mosi_seed2026"
python -m scripts.train --features "$RUN_ROOT/features/mosi_deberta_clip_clap.pt" \
  --evaluate "$RUN_ROOT/runs/mosi_seed2026/best.pt"
```

`configs/bchrpp_mosi_100ep.yaml` 设置最多训练 100 轮、连续 10 轮验证集 MAE 未严格降低则早停，因此实际训练轮数可能小于 100；不在测试集上调参。服务器如果仍使用之前上传的旧压缩包，需要单独上传新配置，或从已有配置在自己的实验目录生成等效配置，并把 `--config` 改为其实际路径。`--run-dir` 必须指向不存在的新目录，以免覆盖先前实验。训练仅凭验证集 MAE 选取最佳轮次，然后对测试集评测一次。结果保存在该目录的 `best.pt` 和 `result.json`。MOSI 的 Acc-7 会先将连续预测值限制在 [-3, 3] 再四舍五入；MAE 则基于未截断的连续预测值计算。报告还包括 Corr、两种 Acc-2/F1 统计口径、各划分样本数、提取模型路径和对齐方式计数。

原提取脚本在 `.pt` 文件中存放 NumPy 数组，因此读取时使用 `torch.load(..., weights_only=False)`。只读取自己确认来源可信的特征文件。
