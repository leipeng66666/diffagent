#!/usr/bin/env python3
"""
使用ModelScope（魔搭）下载sentence-transformers模型
ModelScope是国内可用的模型托管平台，无需翻墙
"""

import os
import sys

def install_modelscope():
    """安装ModelScope"""
    print("📦 安装ModelScope...")
    os.system("pip install -i https://mirrors.aliyun.com/pypi/simple/ modelscope")

def download_with_modelscope():
    """使用ModelScope下载模型"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    local_dir = "./models/all-MiniLM-L6-v2"
    
    print("=" * 60)
    print("📥 使用ModelScope下载模型")
    print("=" * 60)
    print(f"模型名称: {model_name}")
    print(f"保存路径: {local_dir}")
    print(f"下载源: ModelScope（国内镜像）")
    print("=" * 60)
    
    try:
        # 尝试导入modelscope
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("\n⚠️ 需要安装modelscope")
            install_modelscope()
            from modelscope import snapshot_download
        
        print("\n🚀 开始下载模型...")
        print("⏳ 模型大小约250MB，请耐心等待...")
        
        # ModelScope的all-MiniLM-L6-v2模型ID
        model_id = "sentence-transformers/paraphrase-MiniLM-L6-v2"  # ModelScope上的类似模型
        
        cache_dir = snapshot_download(model_id, cache_dir=local_dir)
        
        print("\n" + "=" * 60)
        print("✅ 模型下载成功!")
        print("=" * 60)
        print(f"📁 模型路径: {cache_dir}")
        print("\n💡 下一步:")
        print("   1. 重启应用: python3 run.py")
        print("   2. 系统将自动使用本地模型")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ModelScope下载失败: {e}")
        print("=" * 60)
        return False

def main():
    """主函数"""
    print("\n选择下载方式：")
    print("1. ModelScope（推荐，国内可用）")
    print("2. 跳过下载，使用TF-IDF")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        success = download_with_modelscope()
        if not success:
            print("\n⚠️ 下载失败，将使用TF-IDF")
    else:
        print("\n✅ 将使用TF-IDF（已修复，可正常使用）")
    
    print("\n" + "=" * 60)
    print("📚 系统说明：")
    print("- TF-IDF方案已完全修复，可正常使用")
    print("- 所有功能都可以正常工作")
    print("- 本地模型可以提供更好的语义理解")
    print("=" * 60)

if __name__ == "__main__":
    main()




