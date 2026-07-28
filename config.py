"""
Configuration file - Table Data Visualization Q&A AI Agent
"""
import os
from typing import Dict, List, Any
from pydantic.v1 import BaseSettings

class Settings(BaseSettings):
    """Application configuration"""
    
    # API configuration
    OPENAI_API_KEY: str = ""  # Set via .env file — do NOT hardcode keys
    OPENAI_BASE_URL: str = "https://api.deepseek.com/v1"
    OPENAI_MODEL: str = "deepseek-v4-pro"
    
    # Vector database configuration
    VECTOR_DB_PATH: str = "./data/vector_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Local model path (use if exists)
    LOCAL_MODEL_PATH: str = "./models/all-MiniLM-L6-v2/sentence-transformers/paraphrase-MiniLM-L6-v2"
    
    # Data processing configuration
    MAX_ROWS: int = 10000
    SUPPORTED_FORMATS: List[str] = [".csv", ".xlsx", ".xls", ".json"]
    
    # Visualization configuration
    DEFAULT_PLOT_STYLE: str = "seaborn-v0_8"
    MAX_PLOT_POINTS: int = 1000
    
    # Retrieval configuration
    TOP_K_RESULTS: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/table_agent.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Synonym mapping configuration
SYNONYM_MAPPING = {
    "carbon_dioxide": ["CO2", "carbon dioxide", "CO₂"],
    "temperature": ["temperature", "temp", "T"],
    "pressure": ["pressure", "press", "P"],
    "time": ["time", "t", "Time"],
    "material": ["material", "materials", "Material"],
    "concentration": ["concentration", "conc", "Concentration"],
    "mass": ["mass", "weight", "Mass"],
    "volume": ["volume", "vol", "Volume"],
    "density": ["density", "dens", "Density"],
    "energy": ["energy", "Energy", "E"],
    "power": ["power", "Power", "P"],
    "efficiency": ["efficiency", "eff", "Efficiency"],
    "velocity": ["velocity", "speed", "Velocity"],
    "acceleration": ["acceleration", "accel", "Acceleration"],
    "force": ["force", "Force", "F"],
    "distance": ["distance", "dist", "Distance"],
    "area": ["area", "Area", "A"],
    "length": ["length", "Length", "L"],
    "width": ["width", "Width", "W"],
    "height": ["height", "Height", "H"],
    "depth": ["depth", "Depth", "D"]
}

# Chart type configuration
CHART_TYPES = {
    "distribution": ["histogram", "density", "box", "violin"],
    "trend": ["line", "scatter", "area"],
    "comparison": ["bar", "column", "grouped_bar"],
    "correlation": ["scatter", "heatmap", "correlation"],
    "categorical": ["pie", "donut", "treemap"],
    "geographic": ["map", "choropleth", "scatter_mapbox"]
}

# Prompt templates
PROMPT_TEMPLATES = {
    "data_analysis": """
You are a professional data analyst. Please answer the question based on the following table data:

Data context:
{context}

User question: {question}

Please provide:
1. Data overview and key statistics
2. Trend analysis and pattern recognition
3. Multi-dimensional comparison and correlation analysis
4. Professional conclusions and recommendations

Please answer professionally and accurately.
""",
    
    "visualization": """
Please generate an appropriate visualization chart for the following data:

Data description: {data_description}
Chart type: {chart_type}
Data: {data}

Please generate a clear, attractive, and informative chart.
"""
}

settings = Settings()
