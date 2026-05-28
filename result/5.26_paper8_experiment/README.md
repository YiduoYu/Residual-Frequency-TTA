# 5.26 Paper8 Experiment Result Index

这个目录是本地结果的人工整理索引，用来明确标记：这批结果来自 `script/5.26/run_paper8_spectral.sh` 这套 5.26 paper8 实验流程。

原始结果没有被移动，仍然保留在 `Spectral-master/results/summary/`、`Spectral-master/results/BASELINE/`、`Spectral-master/results/SIMPLE/`、`Spectral-master/results/COSA_SPECTRAL/` 等原位置。本目录只复制了最需要查看和引用的核心文件，方便判断来源、写报告和复查。

## 判断依据

- 控制脚本：`script/5.26/run_paper8_spectral.sh`
- GPU 来源目录：`/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/`
- 本地目标目录：`Spectral-master/results/`
- 主运行 ID：`paper8_8datasets_20260527_020659`
- reboot 后续跑 ID：`paper8_resume_after_reboot_20260528_120157`
- 实验设置：8 个数据集，6 个模型，seeds `0 1 2`，horizons `96 192 336 720`
- 阶段逻辑：`train -> baseline -> cosa -> spectral -> collect`
- 续跑设置：`SKIP_EXISTING=True`，`CONTINUE_ON_ERROR=True`，`COLLECT_REQUIRE_COMPLETE=False`

注意：文件名里出现 `20260527` / `20260528` 是因为实验在 5.26 夜间启动并跨天运行、GPU reboot 后又继续 resume；它仍然对应 `script/5.26/` 下的这套实验脚本。

## 核心结果文件

### final_tables

- `final_tables/paper8_full_comparison.csv`
  - 最完整的逐 seed / model / dataset / horizon 结果表。
  - 1909 行，包含表头；实际结果记录 1908 条。
  - family 计数：`BASELINE=513`，`SIMPLE=930`，`COSA_SPECTRAL=465`。

- `final_tables/paper8_full_comparison_averaged.csv`
  - 用于报告的平均结果表。
  - 685 行，包含表头；实际平均记录 684 条。
  - 这个表适合直接用来比较 baseline、COSA simple、COSA PaaS 和 spectral 类结果。

### failure_logs

- `failure_logs/paper8_8datasets_20260527_020659_failures.tsv`
  - 主运行失败记录。
  - 295 行，包含表头；实际失败 294 条。
  - stage 计数：`train=63`，`baseline=63`，`cosa=126`，`spectral=42`。

- `failure_logs/paper8_resume_after_reboot_20260528_120157_failures.tsv`
  - reboot 后续跑失败记录。
  - 64 行，包含表头；实际失败 63 条。
  - 这些主要是因为前面 train 没有生成 checkpoint，后续 spectral 阶段自然跳过/失败。

### run_logs

- `run_logs/paper8_8datasets_20260527_020659.log`
  - 主运行日志。

- `run_logs/paper8_resume_after_reboot_20260528_120157.log`
  - GPU reboot 后从已有结果继续跑 `spectral collect` 的日志。

### CHECKSUMS.sha256

- 核心 CSV、failure TSV、run log 的 SHA-256 校验值。
- 以后如果担心结果被覆盖或改动，可以用它核对文件身份。

## 原始结果位置

这些目录仍在原地，后续脚本继续查已有结果时可以正常使用：

- `Spectral-master/results/summary/`
- `Spectral-master/results/BASELINE/`
- `Spectral-master/results/PAPER8_TRAIN/`
- `Spectral-master/results/SIMPLE/`
- `Spectral-master/results/COSA_SPECTRAL/`

## 如何使用

- 写报告或做最终表：优先看 `final_tables/paper8_full_comparison_averaged.csv`。
- 查单个 seed 或单个组合：看 `final_tables/paper8_full_comparison.csv`。
- 解释为什么某些组合缺失：看 `failure_logs/`。
- 追溯实验过程：看 `run_logs/`。

这个目录不包含 train checkpoint。checkpoint 很大，仍然建议留在 GPU 的 `Spectral-master/checkpoints/` 下，除非确实需要专门备份。
