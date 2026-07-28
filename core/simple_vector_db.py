
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SimpleVectorDB:
    """Simple vector database implementation (based on numpy and sklearn)"""
    
    def __init__(self, db_path: str = "./data/vector_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.vectors = None
        self.metadata = []
        self.dimension = None
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add vectors to database"""
        if self.vectors is None:
            self.vectors = vectors
            self.dimension = vectors.shape[1]
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        
        self.metadata.extend(metadata)
        logger.info(f"Added {len(vectors)} vectors, total: {len(self.vectors)}")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search similar vectors"""
        if self.vectors is None:
            return []
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector.reshape(1, -1), self.vectors)[0]
        
        # Get top-k results
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            result = {
                "id": int(idx),
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx]
            }
            results.append(result)
        
        return results
    
    def save(self, filename: str = "simple_index"):
        """Save vectors and metadata"""
        if self.vectors is None:
            raise ValueError("No vectors to save")
        
        # Save vectors
        vectors_path = self.db_path / f"{filename}_vectors.npy"
        np.save(vectors_path, self.vectors)
        
        # Save metadata
        metadata_path = self.db_path / f"{filename}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved vectors to: {vectors_path}")
        logger.info(f"Saved metadata to: {metadata_path}")
    
    def load(self, filename: str = "simple_index"):
        """Load vectors and metadata"""
        vectors_path = self.db_path / f"{filename}_vectors.npy"
        metadata_path = self.db_path / f"{filename}_metadata.json"
        
        if not vectors_path.exists():
            raise FileNotFoundError(f"Vector file not found: {vectors_path}")
        
        # Load vectors
        self.vectors = np.load(vectors_path)
        self.dimension = self.vectors.shape[1]
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        
        logger.info(f"Loaded vectors: {vectors_path}")
        logger.info(f"Loaded metadata: {metadata_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.vectors is None:
            return {"status": "Not initialized"}
        
        return {
            "status": "Initialized",
            "vector_count": len(self.vectors),
            "vector_dimension": self.dimension,
            "metadata_count": len(self.metadata)
        }
