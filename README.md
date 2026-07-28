# 表格数据可视化问答AI Agent

一个专门针对表格数据的智能可视化问答系统，能够理解自然语言查询，自动分析数据，并生成专业的可视化图表。

## 🌟 核心特性

### 智能语义解析
- **实体识别**: 自动识别查询中的关键实体（如"二氧化碳"、"温度"等）
- **条件提取**: 智能解析查询条件（如"大于100"、"包含CO2"等）
- **意图识别**: 理解用户的分析意图（趋势分析、对比分析、分布分析等）
- **单位识别**: 智能识别和转换各种单位（温度、压力、浓度等）⭐

### 同义词映射系统
- **智能列匹配**: 将自然语言实体映射到表格列名
- **多语言支持**: 支持中英文同义词映射
- **自定义扩展**: 支持添加领域特定的同义词

### 混合RAG策略
- **关键词检索**: 基于TF-IDF的精确匹配
- **向量检索**: 基于语义相似度的深度匹配
- **混合融合**: 结合两种检索策略获得最佳结果

### 专业可视化
- **自动图表推荐**: 根据数据特征和查询需求推荐合适的图表类型
- **多种图表类型**: 支持直方图、散点图、柱状图、饼图、热力图等
- **交互式图表**: 支持Plotly交互式可视化

### 智能分析
- **统计分析**: 自动计算描述性统计和相关性分析
- **趋势识别**: 识别数据中的趋势和模式
- **异常检测**: 自动检测数据中的异常值
- **专业报告**: 生成结构化的分析报告

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 8GB+ RAM (推荐)
- 2GB+ 磁盘空间

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd table-agent
```

2. **安装依赖**
```bash
pip install -r requirements.txt

# 下载 spaCy 语言模型（用于语义解析）
python -m spacy download en_core_web_sm
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑.env文件，设置您的API密钥
```

4. **启动应用**
```bash
python run.py
```

5. **访问Web界面**
打开浏览器访问: http://localhost:8000

## 📊 使用指南

### 1. 上传数据
- 支持CSV、XLSX、XLS、JSON格式
- 拖拽上传或点击选择文件
- 系统会自动分析数据结构

### 2. 智能问答
输入自然语言查询，例如：
- "分析温度分布"
- "比较不同材料的性能"
- "显示二氧化碳浓度大于100的数据"
- "绘制温度和压力的散点图"

### 3. 可视化生成
- 系统会根据查询自动生成合适的图表
- 支持手动选择图表类型和参数
- 提供交互式图表查看

### 4. 分析报告
- 自动生成专业的分析报告
- 包含统计摘要、趋势分析、异常检测
- 支持导出为JSON格式

## 🔧 配置说明

### DeepSeek 集成 ⭐
本项目使用 DeepSeek 作为大语言模型后端，提供强大的中英文理解和生成能力：

- **模型**: deepseek-v4-pro
- **API**: https://api.deepseek.com/v1 (OpenAI 兼容接口)
- **优势**:
  - 优秀的中英双语理解能力
  - 专业的数据分析能力
  - 稳定的 API 服务
  - 支持任意客体分子的分离分析

### 环境变量配置
```bash
# DeepSeek API configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-v4-pro

# 向量数据库配置
VECTOR_DB_PATH=./data/vector_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 数据处理配置
MAX_ROWS=10000
SUPPORTED_FORMATS=.csv,.xlsx,.xls,.json
```

### 同义词映射配置
系统内置了常用的同义词映射，您也可以添加自定义映射：

```python
# 添加自定义同义词
table_agent.add_custom_synonym_mapping(
    key="温度",
    synonyms=["temperature", "temp", "T", "温度"]
)
```

## 📈 支持的图表类型

| 图表类型 | 适用场景 | 示例用途 |
|---------|---------|---------|
| 直方图 | 分布分析 | 分析数值变量的分布特征 |
| 散点图 | 相关性分析 | 探索两个变量之间的关系 |
| 柱状图 | 分类对比 | 比较不同类别的数值 |
| 饼图 | 比例分析 | 显示分类变量的占比 |
| 热力图 | 相关性矩阵 | 展示多变量间的相关关系 |
| 线图 | 趋势分析 | 显示时间序列数据的变化 |

## 🛠️ API接口

### 数据上传
```http
POST /api/upload
Content-Type: multipart/form-data

file: <表格文件>
```

### 智能问答
```http
POST /api/query
Content-Type: application/x-www-form-urlencoded

query: 分析温度分布
```

### 可视化生成
```http
POST /api/visualize
Content-Type: application/x-www-form-urlencoded

chart_type: histogram
x_column: temperature
y_column: pressure
```

### 获取数据预览
```http
GET /api/data-preview?max_rows=10
```

## 🔍 高级功能

### 1. 自定义同义词映射
```python
# 添加领域特定的同义词
table_agent.add_custom_synonym_mapping(
    key="反应温度",
    synonyms=["reaction_temperature", "react_temp", "T_reaction"]
)
```

### 2. 批量分析
```python
# 批量处理多个查询
queries = [
    "分析温度分布",
    "比较不同材料的性能",
    "显示异常值"
]

for query in queries:
    result = table_agent.process_query(query)
    print(f"查询: {query}")
    print(f"回答: {result['response']['answer']}")
```

### 3. 导出分析报告
```python
# 导出详细的分析报告
result = table_agent.export_analysis_report(
    query="分析温度分布",
    output_path="temperature_analysis.json"
)
```

## 🐛 故障排除

### 常见问题

1. **依赖安装失败**
```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

2. **OpenAI API调用失败**
- 检查API密钥是否正确
- 确认网络连接正常
- 检查API配额是否充足

3. **内存不足**
- 减少MAX_ROWS配置
- 使用更小的数据集进行测试
- 增加系统内存

4. **可视化显示异常**
- 检查matplotlib后端配置
- 确认字体设置正确
- 查看日志文件获取详细错误信息

### 日志查看
```bash
# 查看应用日志
tail -f logs/table_agent.log

# 查看错误日志
grep "ERROR" logs/table_agent.log
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [DeepSeek](https://deepseek.com/) - 提供强大的语言模型
- [Hugging Face](https://huggingface.co/) - 提供预训练模型
- [Plotly](https://plotly.com/) - 提供交互式可视化
- [FastAPI](https://fastapi.tiangolo.com/) - 提供现代Web框架

## 📞 支持

如果您遇到问题或有建议，请：
- 创建Issue
- 发送邮件至: [your-email@example.com]
- 查看文档: [项目文档链接]

---

**让数据分析变得简单而智能！** 🚀
