# RCSE-Lite 工程核验记录（2026-10-08）

## 测试环境

* Python 3.13，PyTorch 2.10.0+cpu，CPU（无 CUDA）。
* 单线程 CPU，`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；时间统计只用作本次同环境对比。
* 模型比较均采用：batch=4、T=50、输入维度 768/5/20、queries=4；每模型前向 warm-up 4 次、记录40次 eval/no_grad 平均。

## 1. 单元测试

`python -m unittest discover -s tests -v`：15/15 通过；包括 masked softmax、全缺失拒绝、模态屏蔽保持权重和预测不变、前后向传播、分类输出、重载、旧模型检查点加载、metrics。

## 2. 合成输入的完整数据流程

已使用本地小型**合成** PKL（18/9/9 splits，各样本 7 时步）核验：

* `train_mosi.py` 与 `train_simsv2.py`：完成两轮训练，验证、最佳 checkpoint 存储及 test 阶段均可执行。
* `eval_acc7.py`：可读回新的 `bchrpp_rcse_lite_v1` checkpoint。
* `run_ablations.py`：`lite/no_complementarity/no_refinement` 三模式各一随机种子独立生成CSV。
* `legacy`、`enhanced_v1` 模型的单轮训练和独立 checkpoint 读回同样可执行。

这些合成数据数值**绝不可作为真实基准性能、精度优势或 SOTA 结论**。

## 3. 参数量与 CPU 前向计时

| hidden dimension | Legacy | Enhanced v1 | RCSE-Lite |
| --- | ---: | ---: | ---: |
| d=128 参数 | 1,338,371 | 592,010 | **202,758** |
| d=128 CPU平均毫秒/批 | 5.548 | 6.299 | **2.991** |
| d=256 参数 | 3,033,091 | 2,150,666 | **602,118** |
| d=256 CPU平均毫秒/批 | 9.329 | 11.218 | **5.214** |

这些性能数据依赖 CPU、PyTorch 版本、线程、d/Q/TopK 等，不能直接迁移到 GPU 或其他机器，也不代表训练吞吐。旧版与新版权重不可混用。

## 4. 尚未完成

* 没有真实 MOSI/MOSEI/CH-SIMS/CH-SIMSv2 多 seed 实验结果。
* 没有具备统计显著性的精度提升、泛化增益、抗缺失能力证据。
* 没有验证显存、GPU 延迟、训练收敛、论文投稿 SOTA 口径。
* 后续如在同一协议下 Lite 精度较低，应保留消融并调整质量探针或图路径，而不是仅凭参数量宣称更优。
