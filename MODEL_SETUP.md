# 本地模型配置指南

## 📝 问题说明
系统默认使用在线的sentence-transformers模型，但由于网络限制可能无法下载。当前系统已自动使用TF-IDF作为替代方案，功能正常但效果可能不如使用专业模型。

## 🎯 解决方案

### 方案1：使用国内镜像下载（推荐）

#### 1. 安装huggingface-hub
```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ huggingface-hub
```

#### 2. 运行下载脚本
```bash
python3 download_model.py
```

#### 3. 重启应用
模型下载完成后，系统会自动使用本地模型。

### 方案2：手动下载模型

#### 1. 访问国内镜像站
https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2

#### 2. 下载所有文件到本地
下载以下文件到 `./models/all-MiniLM-L6-v2/` 目录：
- config.json
- pytorch_model.bin
- tokenizer_config.json
- vocab.txt
- tokenizer.json
- special_tokens_map.json
- modules.json
- sentence_bert_config.json
- config_sentence_transformers.json

#### 3. 确认目录结构
```
cursor/
├── models/
│   └── all-MiniLM-L6-v2/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       └── ... (其他文件)
├── core/
├── app.py
└── run.py
```

#### 4. 重启应用
```bash
python3 run.py
```

### 方案3：继续使用TF-IDF（当前方案）

如果暂时无法下载模型，可以继续使用TF-IDF方案：
- ✅ 优点：无需下载，即开即用
- ⚠️ 缺点：语义理解能力较弱
- 📊 适用场景：关键词匹配、简单查询

## 🔍 验证配置

运行应用后，查看日志：
```bash
# 如果看到这个，说明使用本地模型成功
成功加载本地嵌入模型: ./models/all-MiniLM-L6-v2

# 如果看到这个，说明使用TF-IDF
使用TF-IDF作为替代方案
```

## 💡 性能对比

| 特性 | Sentence-Transformers | TF-IDF |
|------|----------------------|--------|
| 语义理解 | ✅ 强 | ⚠️ 弱 |
| 精确匹配 | ✅ 好 | ✅ 好 |
| 速度 | ⚠️ 中等 | ✅ 快 |
| 内存占用 | ⚠️ 较高 | ✅ 低 |
| 网络依赖 | ⚠️ 首次需要 | ✅ 无 |

## 📌 注意事项

1. 模型文件约250MB，下载需要一定时间
2. 下载完成后无需再次下载，可离线使用
3. 如果下载失败，系统会自动使用TF-IDF，不影响基本功能
4. 建议网络环境好的时候下载模型以获得最佳体验




