#!/usr/bin/env python3
"""
下载sentence-transformers模型到本地（使用国内镜像）
"""

import os
import sys

def download_model_with_mirror():
    """使用国内镜像下载模型"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    local_dir = "./models/all-MiniLM-L6-v2"
    
    print("=" * 60)
    print("📥 下载Sentence-Transformers模型")
    print("=" * 60)
    print(f"模型名称: {model_name}")
    print(f"保存路径: {local_dir}")
    print(f"镜像源: https://hf-mirror.com")
    print("=" * 60)
    
    # 创建目录
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # 首先检查是否已安装huggingface_hub
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("\n⚠️ 需要安装huggingface-hub")
            print("正在安装...")
            os.system("pip install -i https://mirrors.aliyun.com/pypi/simple/ huggingface-hub")
            from huggingface_hub import snapshot_download
        
        # 设置镜像源
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        print("\n🚀 开始下载模型（使用国内镜像）...")
        print("⏳ 模型大小约250MB，请耐心等待...")
        
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True  # 支持断点续传
        )
        
        print("\n" + "=" * 60)
        print("✅ 模型下载成功!")
        print("=" * 60)
        print(f"📁 模型路径: {os.path.abspath(local_dir)}")
        print("\n💡 下一步:")
        print("   1. 重启应用: python3 run.py")
        print("   2. 系统将自动使用本地模型")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 下载失败: {e}")
        print("=" * 60)
        print("\n📝 备选方案：")
        print("1. 手动下载模型文件：")
        print("   访问: https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2")
        print("   下载所有文件到:", os.path.abspath(local_dir))
        print("\n2. 继续使用TF-IDF（已修复，可正常使用）")
        print("=" * 60)
        
        return False

if __name__ == "__main__":
    success = download_model_with_mirror()
    sys.exit(0 if success else 1)

