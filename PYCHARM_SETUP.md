# PyCharm 项目配置指南

## 🎯 目标
配置PyCharm使用当前的conda环境来运行表格数据可视化问答AI Agent项目。

## 📋 环境信息
- **Python路径**: `/home/leipeng/miniconda3/bin/python`
- **Conda环境**: `base`
- **项目路径**: `/home/leipeng/桌面/cursor`

## 🚀 配置步骤

### 1. 打开项目
1. 启动PyCharm
2. 选择 `File` → `Open`
3. 浏览到项目目录：`/home/leipeng/桌面/cursor`
4. 点击 `OK` 打开项目

### 2. 配置Python解释器
1. 打开 `File` → `Settings` (或 `PyCharm` → `Preferences` on macOS)
2. 在左侧面板选择 `Project: cursor` → `Python Interpreter`
3. 点击右上角的齿轮图标 ⚙️
4. 选择 `Add...`

### 3. 添加Conda环境
1. 在弹出窗口中选择 `Conda Environment`
2. 选择 `Existing environment`
3. 在 `Interpreter` 字段中输入或浏览到：
   ```
   /home/leipeng/miniconda3/bin/python
   ```
4. 点击 `OK`

### 4. 验证环境
在PyCharm的Python Console中运行以下代码验证环境：

```python
# 检查Python版本
import sys
print(f"Python版本: {sys.version}")

# 检查关键包
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import openai
    import transformers
    import fastapi
    print("✅ 所有核心依赖包已正确安装")
except ImportError as e:
    print(f"❌ 缺少依赖包: {e}")
```

### 5. 配置运行配置
1. 点击右上角的运行配置下拉菜单
2. 选择 `Edit Configurations...`
3. 点击 `+` 添加新配置
4. 选择 `Python`
5. 配置如下：
   - **Name**: `Table Agent`
   - **Script path**: `/home/leipeng/桌面/cursor/run.py`
   - **Python interpreter**: 选择刚才配置的conda环境
   - **Working directory**: `/home/leipeng/桌面/cursor`

### 6. 安装PyCharm插件（可选）
推荐安装以下插件：
- **Python**: 默认已安装
- **Conda**: 用于conda环境管理
- **Database Tools**: 用于数据库连接
- **Markdown**: 用于README文件预览

## 🔧 高级配置

### 配置代码检查
1. 进入 `Settings` → `Editor` → `Inspections`
2. 确保Python相关的检查都已启用
3. 可以配置PEP8代码风格检查

### 配置Git集成
1. 进入 `Settings` → `Version Control` → `Git`
2. 确保Git路径正确
3. 项目会自动检测Git仓库

### 配置终端
1. 进入 `Settings` → `Tools` → `Terminal`
2. 设置Shell路径为：`/usr/bin/bash`
3. 可以配置conda环境自动激活

## 🧪 测试配置

### 运行项目
1. 在PyCharm中打开 `run.py`
2. 右键选择 `Run 'run.py'`
3. 或者使用快捷键 `Ctrl+Shift+F10`

### 检查输出
应该看到类似输出：
```
🚀 启动表格数据可视化问答AI Agent...
📊 加载数据提取器...
🔍 加载语义解析器...
📝 加载同义词映射器...
🤖 加载LLM集成...
📈 加载可视化引擎...
✅ 所有组件加载完成！
🌐 服务器启动在 http://localhost:8000
```

## 🐛 常见问题解决

### 问题1: 找不到Python解释器
**解决方案**:
- 确保conda环境路径正确
- 手动输入完整路径：`/home/leipeng/miniconda3/bin/python`

### 问题2: 包导入错误
**解决方案**:
- 检查PyCharm是否使用了正确的Python解释器
- 在Terminal中运行：`conda list` 确认包已安装

### 问题3: 项目无法运行
**解决方案**:
- 检查工作目录设置
- 确保所有依赖文件都在项目根目录

## 📁 项目结构
```
/home/leipeng/桌面/cursor/
├── core/                    # 核心模块
│   ├── semantic_parser.py   # 语义解析
│   ├── synonym_mapper.py    # 同义词映射
│   ├── data_extractor.py    # 数据提取
│   ├── rag_engine.py        # RAG引擎
│   ├── llm_integration.py   # LLM集成
│   ├── visualization_engine.py # 可视化引擎
│   └── unit_recognizer.py   # 单位识别
├── templates/               # Web模板
├── data/                   # 数据目录
├── run.py                  # 主启动文件
├── app.py                  # Flask应用
├── table_agent.py          # AI Agent主类
├── config.py               # 配置文件
└── requirements.txt         # 依赖列表
```

## 🎉 完成！
配置完成后，您就可以在PyCharm中：
- 编辑和调试代码
- 运行AI Agent
- 使用集成的终端
- 享受智能代码补全和错误检查

## 📞 需要帮助？
如果遇到问题，可以：
1. 检查PyCharm的Event Log
2. 查看Python Console的输出
3. 确认所有文件路径正确
4. 验证conda环境激活状态





