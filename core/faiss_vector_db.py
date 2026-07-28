
import numpy as np
import faiss
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FAISSVectorDB:
    """FAISS-based vector database implementation"""
    
    def __init__(self, db_path: str = "./data/vector_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.metadata = []
        self.dimension = None
        
    def create_index(self, dimension: int, index_type: str = "flat"):
        """Create FAISS index"""
        self.dimension = dimension
        
        if index_type == "flat":
            # Exact search, slower but accurate results
            self.index = faiss.IndexFlatIP(dimension)  # Inner product search
        elif index_type == "ivf":
            # Approximate search, faster but may have precision loss
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Created FAISS index, dimension: {dimension}, type: {index_type}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add vectors to database"""
        if self.index is None:
            raise ValueError("Please create index first")
        
        # Ensure vectors are float32 type
        vectors = vectors.astype(np.float32)
        
        # Add to index
        self.index.add(vectors)
        
        # Save metadata
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(vectors)} vectors")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search similar vectors"""
        if self.index is None:
            raise ValueError("Please create index first")
        
        # Ensure query vector is float32 type
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    "id": int(idx),
                    "score": float(score),
                    "metadata": self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def save(self, filename: str = "faiss_index"):
        """Save index and metadata"""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        index_path = self.db_path / f"{filename}.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = self.db_path / f"{filename}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved index to: {index_path}")
        logger.info(f"Saved metadata to: {metadata_path}")
    
    def load(self, filename: str = "faiss_index"):
        """Load index and metadata"""
        index_path = self.db_path / f"{filename}.index"
        metadata_path = self.db_path / f"{filename}_metadata.json"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        
        logger.info(f"Loaded index: {index_path}")
        logger.info(f"Loaded metadata: {metadata_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.index is None:
            return {"status": "Not initialized"}
        
        return {
            "status": "Initialized",
            "vector_count": self.index.ntotal,
            "vector_dimension": self.dimension,
            "metadata_count": len(self.metadata)
        }
