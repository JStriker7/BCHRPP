# BCHR++ 工程移交：RCSE-Lite（2026-10-08）

当前默认 `bchrpp/full_model.py` 对应 RCSE-Lite，而不是此前的 enhanced_v1。
原始 checkpoint `checkpoints/mosi_best.pt` **仅属于** `legacy_model.py`。
此前的增强模型、增强损失单独保存在 `bchrpp/enhanced_v1.py` 与 `losses/enhanced_v1_loss.py`。
阅读 `README.md` 的真实训练命令，阅读 `docs/METHOD_AND_PROTOCOL.md` 的机制公式，阅读 `docs/TEST_REPORT.md` 的验证范围。
目前所有端到端训练测试均在合成小样本执行；高精度应以原始真实数据重训、多seed、统一实验设置证实。
