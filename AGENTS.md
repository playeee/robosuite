# AGENTS.md

## 环境规范

本项目使用 conda 环境 `robosuite`。运行脚本、安装依赖都必须指定该环境：

```bash
conda run -n robosuite pip install <package>
conda run -n robosuite python <script>.py
```

不要用 `base` 环境或其他环境。
