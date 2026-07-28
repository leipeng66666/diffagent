# 新功能说明

## ✅ 已完成的功能

### 1. GraphRAG方法
- ✅ 创建了 `core/graphrag_engine.py` - GraphRAG引擎
- ✅ 支持基于知识图谱的数据分析
- ✅ 自动构建实体和关系图
- ✅ 兼容SimpleDataFrame

### 2. 方法选择开关
- ✅ 在UI中添加了GraphRAG开关
- ✅ 支持通过API切换方法（`/api/set-method`）
- ✅ 在查询时可以通过参数选择方法

### 3. 画图功能
- ✅ 创建了 `core/code_generator.py` - 代码生成器
- ✅ 自动检测画图意图（关键词：画、绘制、图表等）
- ✅ LLM生成绘图代码并自动执行
- ✅ 在聊天框中直接显示生成的图片

### 4. 查询意图识别
- ✅ 自动区分分析/推理请求和画图请求
- ✅ 分析/推理请求使用原方法或GraphRAG
- ✅ 画图请求使用代码生成方式

## 📁 新增文件

1. `core/graphrag_engine.py` - GraphRAG引擎
2. `core/code_generator.py` - 代码生成和执行器
3. `test_new_features.py` - 功能测试脚本
4. `quick_test.py` - 快速测试脚本

## 🔧 修改的文件

1. `table_agent.py` - 集成GraphRAG和代码生成器
2. `app.py` - 添加方法切换API
3. `templates/index.html` - 添加GraphRAG开关和画图显示

## 🚀 使用方法

### 1. 使用GraphRAG方法
- 在Web界面中打开"使用 GraphRAG 方法"开关
- 或者通过API: `POST /api/set-method` with `use_graphrag=true`

### 2. 画图功能
- 在聊天框中输入画图请求，例如：
  - "画一个温度分布的直方图"
  - "绘制散点图显示温度和压力的关系"
  - "生成材料性能的柱状图"
- 系统会自动生成代码并执行，在聊天框中显示图片

### 3. 分析/推理功能
- 输入分析请求，例如：
  - "分析温度分布"
  - "比较不同材料的性能"
- 系统会根据开关选择使用原方法或GraphRAG方法

## ⚠️ 注意事项

1. GraphRAG构建知识图谱需要一些时间（特别是大数据集）
2. 画图功能需要LLM生成代码，可能需要10-30秒
3. 确保已安装networkx: `pip install networkx`

## 🧪 测试

运行测试脚本：
```bash
python quick_test.py
```

或完整测试：
```bash
python test_new_features.py
```

## 📝 技术细节

### GraphRAG实现
- 使用NetworkX构建知识图谱
- 提取实体（分类列、统计信息）
- 建立关系（共现、相关性）
- 图遍历查找相关数据

### 代码生成实现
- 使用LLM根据用户需求生成Python代码
- 自动执行代码生成图片
- 转换为base64格式在聊天框显示

## 🎉 功能状态

- ✅ GraphRAG模块：完成
- ✅ 方法选择开关：完成
- ✅ 画图功能：完成
- ✅ 意图识别：完成
- ✅ SimpleDataFrame兼容：完成

所有功能已实现并修复了主要兼容性问题，可以进行测试使用！


