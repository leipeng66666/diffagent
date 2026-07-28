#!/bin/bash
# 表格数据可视化问答AI Agent - 启动脚本

echo "🚀 启动表格数据可视化问答AI Agent"
echo "=================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 错误: 需要Python 3.8或更高版本，当前版本: $python_version"
    exit 1
fi

echo "✅ Python版本检查通过: $python_version"

# 检查依赖
echo "📦 检查依赖包..."
missing_packages=()

# 检查关键依赖
packages=("pandas" "numpy" "matplotlib" "seaborn" "plotly" "openai" "fastapi" "uvicorn")
for package in "${packages[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "❌ 缺少以下依赖包: ${missing_packages[*]}"
    echo "📥 正在安装依赖..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败，请手动运行: pip3 install -r requirements.txt"
        exit 1
    fi
    echo "✅ 依赖安装完成"
else
    echo "✅ 所有依赖包已安装"
fi

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p data logs uploads reports static templates
echo "✅ 目录创建完成"

# 检查配置文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到.env配置文件"
    if [ -f "env_example.txt" ]; then
        echo "📋 复制示例配置文件..."
        cp env_example.txt .env
        echo "✅ 已创建.env文件，请编辑并设置您的API密钥"
    fi
fi

# 检查API密钥
if [ -f ".env" ]; then
    if ! grep -q "OPENAI_API_KEY=your_openai_api_key_here" .env; then
        echo "✅ API密钥已配置"
    else
        echo "⚠️  请在.env文件中设置您的OpenAI API密钥"
    fi
fi

# 测试通义千问集成
echo "🧪 测试阿里云通义千问集成..."
python3 quick_test.py
if [ $? -eq 0 ]; then
    echo "✅ 通义千问集成测试通过"
else
    echo "⚠️ 通义千问集成测试失败，但继续启动应用"
fi

# 启动应用
echo "🚀 启动Web服务器..."
echo "📍 访问地址: http://localhost:8000"
echo "🛑 按Ctrl+C停止服务器"
echo ""

python3 run.py
