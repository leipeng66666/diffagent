"""
Hybrid RAG Engine - Combining keyword retrieval and vector retrieval
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from loguru import logger
# Vector database import
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not installed, using FAISS as alternative")

try:
    from .faiss_vector_db import FAISSVectorDB
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from .simple_vector_db import SimpleVectorDB
    SIMPLE_AVAILABLE = True
except ImportError:
    SIMPLE_AVAILABLE = False
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

class HybridRAGEngine:
    """Hybrid RAG Engine"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", local_model_path: str = None):
        """Initialize RAG engine"""
        self.embedding_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=384, stop_words='english')  # Fixed dimensions
        self.tfidf_corpus = []  # Store all documents for fitting
        
        # Try to load embedding model
        try:
            # Prefer local model
            if local_model_path and os.path.exists(local_model_path):
                logger.info(f"Attempting to load local model: {local_model_path}")
                self.embedding_model = SentenceTransformer(local_model_path)
                logger.info(f"Successfully loaded local embedding model: {local_model_path}")
            else:
                logger.info(f"Attempting to load online model: {embedding_model}")
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Successfully loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.warning(f"Unable to load embedding model: {e}")
            logger.info("Using TF-IDF as alternative")
            self.embedding_model = None
        
        # Initialize vector database
        if CHROMADB_AVAILABLE:
            self.chroma_client = chromadb.Client()
        else:
            self.chroma_client = None
        self.collection = None
        
        # Cache
        self.embeddings_cache = {}
        self.tfidf_cache = {}
    
    def index_data(self, df: pd.DataFrame, table_name: str = "default") -> None:
        """Build index for data"""
        logger.info(f"Starting to build index for table '{table_name}'")
        
        # Collect all document texts for TF-IDF fitting
        all_docs = []
        for idx, row in df.iterrows():
            doc_text = self._create_document_text(row, df.columns)
            all_docs.append(doc_text)
            self.tfidf_corpus.append(doc_text)
        
        # If using TF-IDF, fit all documents first
        if self.embedding_model is None:
            logger.info("Using TF-IDF, fitting all documents...")
            self.tfidf_vectorizer.fit(self.tfidf_corpus)
        
        if CHROMADB_AVAILABLE and self.chroma_client is not None:
            # Use ChromaDB
            try:
                self.collection = self.chroma_client.get_collection(table_name)
                self.chroma_client.delete_collection(table_name)
            except:
                pass
            
            self.collection = self.chroma_client.create_collection(
                name=table_name,
                metadata={"description": f"Table data for {table_name}"},
                embedding_function=None
            )
            
            for idx, row in df.iterrows():
                doc_text = all_docs[idx]
                if self.embedding_model is not None:
                    embedding = self.embedding_model.encode(doc_text).tolist()
                else:
                    embedding = self._get_tfidf_embedding(doc_text)
                
                if len(embedding) != 384:
                    if len(embedding) > 384:
                        embedding = embedding[:384]
                    else:
                        embedding.extend([0.0] * (384 - len(embedding)))
                
                self.collection.add(
                    documents=[doc_text],
                    embeddings=[embedding],
                    metadatas=[{
                        "row_index": str(idx),
                        "table_name": table_name,
                        "columns": ",".join(df.columns)
                    }],
                    ids=[f"{table_name}_{idx}"]
                )
        else:
            # Use FAISS or simple in-memory storage
            logger.info("ChromaDB not available, using in-memory document store")
            self.collection = {
                "documents": all_docs,
                "ids": [f"{table_name}_{idx}" for idx in range(len(all_docs))],
                "metadatas": [{
                    "row_index": str(idx),
                    "table_name": table_name,
                    "columns": ",".join(df.columns)
                } for idx in range(len(all_docs))]
            }
        
        # Build TF-IDF index
        try:
            self._build_tfidf_index(df)
        except Exception as e:
            logger.warning(f"TF-IDF index building failed: {e}")
        
        logger.info(f"Index building complete, total {len(df)} rows of data")
    
    def _create_document_text(self, row: pd.Series, columns: List[str]) -> str:
        """Create document text"""
        doc_parts = []
        
        for col in columns:
            value = row[col]
            if pd.notna(value):
                # Format text
                if isinstance(value, (int, float)):
                    doc_parts.append(f"{col}: {value}")
                else:
                    doc_parts.append(f"{col}: {str(value)}")
        
        return " | ".join(doc_parts)
    
    def _build_tfidf_index(self, df: pd.DataFrame) -> None:
        """Build TF-IDF index"""
        # Create text representation for each row
        documents = []
        for idx, row in df.iterrows():
            doc_text = self._create_document_text(row, df.columns)
            documents.append(doc_text)
        
        # Train TF-IDF vectorizer
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        self.tfidf_cache["vectorizer"] = self.tfidf_vectorizer
        self.tfidf_cache["matrix"] = self.tfidf_matrix
    
    def search(self, query: str, top_k: int = 10, 
               search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """Search relevant data"""
        logger.info(f"Search query: {query}")
        
        if search_type == "vector":
            results = self._vector_search(query, top_k)
        elif search_type == "keyword":
            results = self._keyword_search(query, top_k)
        elif search_type == "hybrid":
            results = self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
        
        logger.info(f"Found {len(results)} relevant results")
        return results
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Vector search"""
        if not self.collection:
            return []
        
        # If collection is a plain dict (no ChromaDB), fall back to keyword search
        if isinstance(self.collection, dict):
            return self._keyword_search(query, top_k)
        
        # Generate query embedding
        if self.embedding_model is not None:
            query_embedding = self.embedding_model.encode(query).tolist()
        else:
            # Use TF-IDF as alternative
            query_embedding = self._get_tfidf_embedding(query)
        
        # Ensure query embedding vector dimension is consistent
        if len(query_embedding) != 384:
            if len(query_embedding) > 384:
                query_embedding = query_embedding[:384]
            else:
                query_embedding.extend([0.0] * (384 - len(query_embedding)))
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            formatted_results.append({
                "content": doc,
                "metadata": metadata,
                "score": 1 - distance,  # Convert to similarity score
                "search_type": "vector"
            })
        
        return formatted_results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Keyword search"""
        if "vectorizer" not in self.tfidf_cache:
            return []
        
        # Vectorize query
        query_vector = self.tfidf_cache["vectorizer"].transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.tfidf_cache["matrix"]).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        formatted_results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return results with similarity
                formatted_results.append({
                    "content": f"Row {idx}",
                    "metadata": {"row_index": idx},
                    "score": similarities[idx],
                    "search_type": "keyword"
                })
        
        return formatted_results
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Hybrid search"""
        # Perform both searches
        vector_results = self._vector_search(query, top_k)
        keyword_results = self._keyword_search(query, top_k)
        
        # Merge results
        all_results = vector_results + keyword_results
        
        # Deduplicate (based on row index)
        seen_indices = set()
        unique_results = []
        
        for result in all_results:
            row_idx = result["metadata"].get("row_index")
            if row_idx not in seen_indices:
                seen_indices.add(row_idx)
                unique_results.append(result)
        
        # Sort by score
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        
        return unique_results[:top_k]
    
    def get_context_for_llm(self, search_results: List[Dict[str, Any]], 
                          max_context_length: int = 4000) -> str:
        """Prepare context for LLM"""
        context_parts = []
        current_length = 0
        
        for result in search_results:
            content = result["content"]
            score = result["score"]
            search_type = result["search_type"]
            
            # Format context segment
            context_part = f"[{search_type.upper()}] (similarity: {score:.3f})\n{content}\n"
            
            # Check length limit
            if current_length + len(context_part) > max_context_length:
                break
            
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "\n".join(context_parts)
    
    def analyze_data_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data patterns"""
        patterns = {
            "correlations": {},
            "trends": {},
            "anomalies": {},
            "clusters": {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Calculate correlation
            corr_matrix = df[numeric_cols].corr()
            patterns["correlations"] = corr_matrix
        
        # Analyze trends
        for col in numeric_cols:
            if len(df[col].dropna()) > 1:
                try:
                    # Calculate linear trend
                    x = np.arange(len(df))
                    y = df[col].dropna()
                    if len(y) > 1:
                        slope = np.polyfit(x[:len(y)], y, 1)[0]
                        patterns["trends"][col] = {
                            "slope": slope,
                            "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                        }
                except:
                    pass
        
        # Detect outliers
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                patterns["anomalies"][col] = {
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100,
                    "outlier_values": outliers[col].tolist()
                }
        
        return patterns
    
    def generate_insights(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Generate data insights"""
        insights = {
            "summary": {},
            "patterns": {},
            "recommendations": []
        }
        
        # Basic statistical summary
        insights["summary"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "missing_values": sum(sum(1 for val in df[col] if val is None or val == '' or val == 'nan') for col in df.columns)
        }
        
        # Analyze patterns
        patterns = self.analyze_data_patterns(df)
        insights["patterns"] = patterns
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df, patterns)
        insights["recommendations"] = recommendations
        
        return insights
    
    def _generate_recommendations(self, df: pd.DataFrame, patterns: Dict[str, Any]) -> List[str]:
        """Generate analysis recommendations"""
        recommendations = []
        
        # Recommendations based on missing values
        missing_data = {col: sum(1 for val in df[col] if val is None or val == '' or val == 'nan') for col in df.columns}
        high_missing_cols = [col for col, count in missing_data.items() if count > len(df) * 0.1]
        if len(high_missing_cols) > 0:
            recommendations.append(f"The following columns have many missing values, recommend checking data quality: {high_missing_cols}")
        
        # Recommendations based on correlations
        if "correlations" in patterns:
            corr_matrix = patterns["correlations"]
            high_corr_pairs = []
            for col1 in corr_matrix:
                for col2 in corr_matrix[col1]:
                    if col1 != col2 and abs(corr_matrix[col1][col2]) > 0.7:
                        high_corr_pairs.append((col1, col2, corr_matrix[col1][col2]))
            
            if high_corr_pairs:
                recommendations.append(f"Found strongly correlated variable pairs: {high_corr_pairs[:3]}")
        
        # Recommendations based on outliers
        if "anomalies" in patterns:
            anomaly_cols = [col for col, info in patterns["anomalies"].items() 
                          if info["percentage"] > 5]
            if anomaly_cols:
                recommendations.append(f"The following columns have many outliers, recommend further analysis: {anomaly_cols}")
        
        return recommendations
    
    def _get_tfidf_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using TF-IDF"""
        try:
            # Use fitted TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.transform([text])
            embedding = tfidf_matrix.toarray()[0].tolist()
            
            # Ensure dimension is 384
            if len(embedding) > 384:
                embedding = embedding[:384]
            elif len(embedding) < 384:
                embedding.extend([0.0] * (384 - len(embedding)))
            
            return embedding
        except Exception as e:
            logger.warning(f"TF-IDF embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * 384
