# MOSI 在线文本冻结/解冻成对实验

以原整帧 `.pt` 中的词、标签、CLAP 音频和 CLIP 视觉为输入。
在线 DeBERTa-v3-large 使用原提取器的 fast tokenizer、512 token 上限、
每个词的子词均值。第一组冻结 DeBERTa（保持 eval 和 FP32），第二组
解冻整个 DeBERTa；融合网络及其学习率相同，解冻编码器学习率 2e-5。
启动训练前会在三个 split 各 3 个样本上验证冻结输出与缓存词特征
最大绝对误差不超过 0.001，失败立即停止；不改音频或视觉特征。

从本仓库的 `experiment/mosi-text-unfreeze-public` 分支检出代码，
进入 `BCHRPP_RCSE_Lite_Final_20261008/BCHRPP_RCSE_Lite_20261008`。
自行设置已有实验输出目录；数据、模型及 Python 环境不会由脚本下载：

```bash
export MOSI_RUN_DIR=/path/to/private/experiment-root
# 可选：MOSI_PYTHON=/path/to/python；MOSI_FEATURES=/path/to/features.pt
# 可选：CUDA_VISIBLE_DEVICES=0
mkdir -p "$MOSI_RUN_DIR/results"
set -o pipefail
bash scripts/run_mosi_text_online.sh 2>&1 | tee "$MOSI_RUN_DIR/results/text_online.log"
```

输出分别在 `paired_text_online_frozen_gpu0` 与
`paired_text_online_unfrozen_gpu0`，按相同种子比较；先确认冻结版与
原整帧基线的验证指标是否接近，再解释解冻收益。Checkpoint 包含
DeBERTa 参数，体积和运行时间明显高于原训练；默认使用 GPU 0。
中断时只复用已完整写出 JSON 的 seed，勿覆盖未完成 checkpoint。
