"""
Synonym Mapping Module - Map natural language entities to table column names
"""
import re
from typing import Dict, List, Tuple, Optional
from loguru import logger
from config import SYNONYM_MAPPING

class SynonymMapper:
    """Synonym Mapper"""
    
    def __init__(self, custom_mapping: Optional[Dict] = None):
        """Initialize synonym mapper"""
        self.mapping = SYNONYM_MAPPING.copy()
        if custom_mapping:
            self.mapping.update(custom_mapping)
        
        # Build reverse mapping
        self.reverse_mapping = self._build_reverse_mapping()
        
        # Build fuzzy matching patterns
        self.fuzzy_patterns = self._build_fuzzy_patterns()
    
    def _build_reverse_mapping(self) -> Dict[str, str]:
        """Build reverse mapping table"""
        reverse = {}
        for key, synonyms in self.mapping.items():
            for synonym in synonyms:
                reverse[synonym.lower()] = key
        return reverse
    
    def _build_fuzzy_patterns(self) -> Dict[str, List[str]]:
        """Build fuzzy matching patterns"""
        patterns = {}
        for key, synonyms in self.mapping.items():
            patterns[key] = []
            for synonym in synonyms:
                # Add various variants
                patterns[key].extend([
                    synonym.lower(),
                    synonym.upper(),
                    synonym.capitalize(),
                    synonym.replace(" ", "_"),
                    synonym.replace(" ", "-"),
                    synonym.replace("_", " "),
                    synonym.replace("-", " ")
                ])
        return patterns
    
    def map_entity_to_columns(self, entity: str, available_columns: List[str]) -> List[Tuple[str, float]]:
        """Map entity to table column names"""
        logger.info(f"Mapping entity '{entity}' to column names")
        
        results = []
        
        # 1. Exact match
        exact_matches = self._exact_match(entity, available_columns)
        results.extend(exact_matches)
        
        # 2. Synonym match
        synonym_matches = self._synonym_match(entity, available_columns)
        results.extend(synonym_matches)
        
        # 3. Fuzzy match
        fuzzy_matches = self._fuzzy_match(entity, available_columns)
        results.extend(fuzzy_matches)
        
        # 4. Semantic similarity match
        semantic_matches = self._semantic_match(entity, available_columns)
        results.extend(semantic_matches)
        
        # Deduplicate and sort
        results = self._deduplicate_and_sort(results)
        
        logger.info(f"Mapping result: {results}")
        return results
    
    def _exact_match(self, entity: str, columns: List[str]) -> List[Tuple[str, float]]:
        """Exact match"""
        matches = []
        entity_lower = entity.lower()
        
        for column in columns:
            column_lower = column.lower()
            if entity_lower == column_lower:
                matches.append((column, 1.0))
            elif entity_lower in column_lower or column_lower in entity_lower:
                matches.append((column, 0.9))
        
        return matches
    
    def _synonym_match(self, entity: str, columns: List[str]) -> List[Tuple[str, float]]:
        """Synonym match"""
        matches = []
        entity_lower = entity.lower()
        
        # Find synonyms for the entity
        entity_synonyms = []
        for key, synonyms in self.mapping.items():
            if entity_lower in [s.lower() for s in synonyms]:
                entity_synonyms.extend(synonyms)
        
        # Find synonyms for column names
        for column in columns:
            column_lower = column.lower()
            
            # Direct match synonyms
            for synonym in entity_synonyms:
                if synonym.lower() == column_lower:
                    matches.append((column, 0.95))
                    break
            
            # Check if column name contains synonym
            for synonym in entity_synonyms:
                if synonym.lower() in column_lower or column_lower in synonym.lower():
                    matches.append((column, 0.8))
                    break
        
        return matches
    
    def _fuzzy_match(self, entity: str, columns: List[str]) -> List[Tuple[str, float]]:
        """Fuzzy match"""
        matches = []
        entity_lower = entity.lower()
        
        for column in columns:
            column_lower = column.lower()
            
            # Calculate edit distance similarity
            similarity = self._calculate_similarity(entity_lower, column_lower)
            if similarity > 0.6:
                matches.append((column, similarity * 0.7))
            
            # Check partial match
            if self._partial_match(entity_lower, column_lower):
                matches.append((column, 0.6))
        
        return matches
    
    def _semantic_match(self, entity: str, columns: List[str]) -> List[Tuple[str, float]]:
        """Semantic similarity match"""
        matches = []
        
        # Here can integrate more advanced semantic similarity computation
        # For example using sentence-transformers or word2vec
        
        for column in columns:
            # Simple semantic matching rules
            semantic_score = self._calculate_semantic_similarity(entity, column)
            if semantic_score > 0.5:
                matches.append((column, semantic_score * 0.6))
        
        return matches
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (edit distance)"""
        if not str1 or not str2:
            return 0.0
        
        # Simplified edit distance calculation
        len1, len2 = len(str1), len(str2)
        if len1 == 0:
            return 0.0 if len2 > 0 else 1.0
        if len2 == 0:
            return 0.0
        
        # Calculate longest common subsequence
        lcs = self._longest_common_subsequence(str1, str2)
        return lcs / max(len1, len2)
    
    def _longest_common_subsequence(self, str1: str, str2: str) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _partial_match(self, str1: str, str2: str) -> bool:
        """Check partial match"""
        # Check if one string contains the main part of another
        min_len = min(len(str1), len(str2))
        if min_len < 3:
            return False
        
        # Check if there's a long enough common substring
        for i in range(len(str1) - min_len + 1):
            for j in range(len(str2) - min_len + 1):
                if str1[i:i+min_len] == str2[j:j+min_len]:
                    return True
        
        return False
    
    def _calculate_semantic_similarity(self, entity: str, column: str) -> float:
        """Calculate semantic similarity"""
        # Simple semantic similarity calculation
        # In practice, more advanced NLP models can be integrated here
        
        entity_words = set(entity.lower().split())
        column_words = set(column.lower().split())
        
        if not entity_words or not column_words:
            return 0.0
        
        intersection = entity_words.intersection(column_words)
        union = entity_words.union(column_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _deduplicate_and_sort(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Deduplicate and sort"""
        # Deduplicate by column name, keeping highest score
        unique_results = {}
        for column, score in results:
            if column not in unique_results or score > unique_results[column]:
                unique_results[column] = score
        
        # Sort by score
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def get_synonyms(self, entity: str) -> List[str]:
        """Get synonyms for entity"""
        entity_lower = entity.lower()
        
        # Direct lookup
        for key, synonyms in self.mapping.items():
            if entity_lower in [s.lower() for s in synonyms]:
                return synonyms
        
        # Reverse lookup
        if entity_lower in self.reverse_mapping:
            key = self.reverse_mapping[entity_lower]
            return self.mapping.get(key, [])
        
        return []
    
    def add_synonym_mapping(self, key: str, synonyms: List[str]):
        """Add synonym mapping"""
        self.mapping[key] = synonyms
        # Update reverse mapping
        for synonym in synonyms:
            self.reverse_mapping[synonym.lower()] = key
    
    def suggest_column_mappings(self, entities: List[str], columns: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Suggest column mappings for multiple entities"""
        suggestions = {}
        
        for entity in entities:
            mappings = self.map_entity_to_columns(entity, columns)
            if mappings:
                suggestions[entity] = mappings
        
        return suggestions



