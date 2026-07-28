#!/usr/bin/env python3
"""
Table Data Visualization Q&A AI Agent - Startup Script
"""
import os
import sys
import uvicorn
from loguru import logger
from config import settings

def setup_logging():
    """Setup logging"""
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        settings.LOG_FILE,
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    )
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
    )

def check_dependencies():
    """Check dependencies"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'openai', 'transformers', 'sentence-transformers', 'spacy',
        'chromadb', 'faiss', 'langchain', 'fastapi', 'uvicorn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing dependencies: {', '.join(missing_packages)}")
        logger.info("Please run: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data", "logs", "uploads", "reports", "static", "templates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    """Main function"""
    logger.info("Starting Table Data Visualization Q&A AI Agent")
    
    # Setup logging
    setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check API key
    if not settings.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set, some features may not work")
        logger.info("Please set OPENAI_API_KEY in .env file")
    
    # Start server
    logger.info("Starting web server...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
