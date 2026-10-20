# MOSI 在线 DeBERTa：去模态对照

本包从公开的在线解冻文本版本 `f3f7a1d` 独立制作。特征仍为原整帧
`mosi_deberta_clip_clap.pt`；DeBERTa-v3-large 全层解冻，文本学习率
`2e-5`，BCHR++ 融合学习率 `3e-4`，损失及融合结构不变。
只使用有效位掩码移除音频或视觉；辅助损失读取同一有效位，不为缺失模态
计算单模态/置信度损失。四组为 `full`、`no_audio`、`no_vision`、
`text_only`，每组独立运行 `222/128/2026`，总计 12 次训练。

将 ZIP 上传到 wjy 的 `/home/wjy/SYX`。在该服务器上：

```bash
cd /home/wjy/SYX
ZIP=mosi_modality_ablation_wjy_20261020.zip
DST=BCHRPP_RCSE_Lite_20261020_modality
test ! -e "$DST" || { echo "目录已存在：$DST；请先核对，不能覆盖"; exit 1; }
unzip -tq "$ZIP" && mkdir "$DST" && cd "$DST" && unzip -q "../$ZIP"
bash scripts/run_mosi_modality_next.sh
```

默认使用物理 GPU 0、`/mnt/data2t/wjy/bchrpp_mosi_20261014/env/bin/python`
及原整帧特征。路径可通过 `MOSI_RUN_DIR`、`MOSI_PYTHON`、
`MOSI_FEATURES`、`MOSI_TEXT_MODEL`、`MOSI_OUTPUT_ROOT`、`MOSI_LOG`
环境变量覆盖；模型必须与特征元数据一致，不会下载权重。

脚本按输入及 GPU 预检、完整单测、冻结文本缓存一致性和小样本融合前向、
12 次训练的顺序执行。需要核对 `1284/229/686` 划分、样本/标签指纹
`6296d258a3255962bce41154e11bc9e41498e6c2a8e201b725c678a9f432c84e`
及三模态词级形状。每组结果在
`$MOSI_RUN_DIR/results/mosi_modality_20261020_gpu0/<组名>/`，包含
checkpoint、逐种子 JSON、`results.csv`、`summary.csv`；总体对照为上级
`comparison.csv`，日志为同级 `.log`。再次执行只复用身份匹配的完整结果，
不覆盖不完整 checkpoint；发生中断先保留现场，不要删除已有权重。

使用 `metricsTop.py:regression/MOSI`，只按未四舍五入的验证集 MAE
选择每个 checkpoint，测试结果不得用来决定下一次训练配置。对论文
49.56% Acc-7 只作固定测试口径参考，三种子均值和论文汇总方式未必相同。
本机仅完成 CPU 单元测试；正式 GPU 训练须在 wjy 运行后报告原始 CSV。
