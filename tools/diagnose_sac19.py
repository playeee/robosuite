#!/usr/bin/env python3
"""
SAC_19 诊断脚本（薄包装）。

该脚本原为 SAC_19 专用诊断工具，现已重构为通用分析工具
analyze_training.py 的薄包装。保留此文件以兼容旧调用方式。

分析 SAC_19 训练运行的 TensorBoard 日志与 test_rollouts/ 数据。

使用方式：
    /home/playeee/miniconda3/envs/robosuite/bin/python /home/playeee/projects/robosuite/tools/diagnose_sac19.py

底层实现见：tools/analyze_training.py
"""

import os
import sys
import subprocess

PROJECT_ROOT = "/home/playeee/projects/robosuite"
ANALYZE_SCRIPT = os.path.join(PROJECT_ROOT, "tools/analyze_training.py")
PYTHON_BIN = "/home/playeee/miniconda3/envs/robosuite/bin/python"

# SAC_19 默认配置
DEFAULT_LOGDIR = os.path.join(PROJECT_ROOT, "logs/sac_lift_so101_realistic/")
DEFAULT_RUN = "SAC_19"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tools/sac19_analysis/")


def main():
    """调用 analyze_training.py 分析 SAC_19。"""
    cmd = [
        PYTHON_BIN, ANALYZE_SCRIPT,
        "--logdir", DEFAULT_LOGDIR,
        "--run", DEFAULT_RUN,
        "--output-dir", DEFAULT_OUTPUT_DIR,
    ]
    print("[diagnose_sac19.py] 调用通用分析工具:")
    print("  " + " ".join(cmd))
    print("-" * 80)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
