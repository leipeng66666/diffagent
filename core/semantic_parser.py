"""
Semantic Parsing Module - Identify key entities and conditions in queries
"""
import re
import spacy
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from loguru import logger
from core.unit_recognizer import UnitRecognizer

@dataclass
class QueryEntity:
    """Query Entity"""
    text: str
    label: str
    confidence: float
    start: int
    end: int
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "text": self.text,
            "label": self.label,
            "confidence": self.confidence,
            "start": self.start,
            "end": self.end
        }

@dataclass
class QueryCondition:
    """Query Condition"""
    entity: str
    operator: str  # >, <, =, >=, <=, contains, between
    value: Any
    confidence: float
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "entity": self.entity,
            "operator": self.operator,
            "value": self.value,
            "confidence": self.confidence
        }

class SemanticParser:
    """Semantic Parser"""
    
    def __init__(self, model_name: str = "zh_core_web_sm"):
        """Initialize semantic parser"""
        self.nlp = None
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Using English model: en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using simple text processing")
                self.nlp = None
        
        # Initialize unit recognizer
        self.unit_recognizer = UnitRecognizer()
        
        # Predefined patterns
        self.patterns = {
            "numeric_condition": [
                r"(\w+)\s*[>大于]\s*(\d+(?:\.\d+)?)",
                r"(\w+)\s*[<小于]\s*(\d+(?:\.\d+)?)",
                r"(\w+)\s*[=等于]\s*(\d+(?:\.\d+)?)",
                r"(\w+)\s*[>=大于等于]\s*(\d+(?:\.\d+)?)",
                r"(\w+)\s*[<=小于等于]\s*(\d+(?:\.\d+)?)",
                r"(\w+)\s*在\s*(\d+(?:\.\d+)?)\s*到\s*(\d+(?:\.\d+)?)\s*之间"
            ],
            "text_condition": [
                r"(\w+)\s*包含\s*([^\s]+)",
                r"(\w+)\s*是\s*([^\s]+)",
                r"(\w+)\s*等于\s*([^\s]+)"
            ],
            "time_condition": [
                r"(\w+)\s*在\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s*到\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
                r"(\w+)\s*大于\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
                r"(\w+)\s*小于\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})"
            ]
        }
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query"""
        logger.info(f"Starting to parse query: {query}")
        
        # Use spaCy for basic NLP processing (if available)
        if self.nlp is not None:
            doc = self.nlp(query)
            entities = self._extract_entities(doc)
        else:
            # Use simple text processing
            entities = self._extract_entities_simple(query)
        
        # Extract conditions
        conditions = self._extract_conditions(query)
        
        # Identify intent
        intent = self._identify_intent(query)
        
        # Identify visualization needs
        visualization_needs = self._identify_visualization_needs(query)
        
        # Identify unit information
        units = self.unit_recognizer.extract_units_from_text(query)
        
        # Enhancement: Detect comparison and separation queries
        comparison_info = self._detect_comparison_and_separation(query, entities)
        
        result = {
            "query": query,
            "entities": entities,
            "conditions": conditions,
            "intent": intent,
            "visualization_needs": visualization_needs,
            "units": units,
            "comparison_info": comparison_info,
            "confidence": self._calculate_confidence(entities, conditions)
        }
        
        logger.info(f"Parse result: {result}")
        return result
    
    def _extract_entities(self, doc) -> List[QueryEntity]:
        """Extract named entities"""
        entities = []
        
        for ent in doc.ents:
            entity = QueryEntity(
                text=ent.text,
                label=ent.label_,
                confidence=0.8,  # spaCy doesn't provide confidence, using default value
                start=ent.start_char,
                end=ent.end_char
            )
            entities.append(entity)
        
        # Manually extract possible entities (based on keywords)
        manual_entities = self._extract_manual_entities(doc)
        entities.extend(manual_entities)
        
        return entities
    
    def _extract_entities_simple(self, query: str) -> List[QueryEntity]:
        """Simple entity extraction (without spaCy)"""
        entities = []
        
        # Simple entity recognition based on keywords
        keywords = [
            "温度", "压力", "浓度", "质量", "体积", "密度", "速度", "时间",
            "温度", "压力", "浓度", "质量", "体积", "密度", "速度", "时间",
            "temperature", "pressure", "concentration", "mass", "volume", "density", "speed", "time"
        ]
        
        for keyword in keywords:
            if keyword in query:
                start = query.find(keyword)
                entity = QueryEntity(
                    text=keyword,
                    label="KEYWORD",
                    confidence=0.7,
                    start=start,
                    end=start + len(keyword)
                )
                entities.append(entity)
        
        return entities
    
    def _extract_manual_entities(self, doc) -> List[QueryEntity]:
        """Manually extract entities"""
        entities = []
        
        # Find numeric values
        for token in doc:
            if token.like_num:
                entity = QueryEntity(
                    text=token.text,
                    label="NUMERIC",
                    confidence=0.9,
                    start=token.idx,
                    end=token.idx + len(token.text)
                )
                entities.append(entity)
        
        # Find possible column names
        for token in doc:
            if token.is_alpha and len(token.text) > 2:
                # Check if it might be a column name
                if self._is_potential_column_name(token.text):
                    entity = QueryEntity(
                        text=token.text,
                        label="COLUMN",
                        confidence=0.6,
                        start=token.idx,
                        end=token.idx + len(token.text)
                    )
                    entities.append(entity)
        
        return entities
    
    def _is_potential_column_name(self, text: str) -> bool:
        """Determine if it might be a column name"""
        # Simple heuristic rules
        column_indicators = ["column", "field", "attribute", "variable", "parameter", "列", "字段", "属性", "变量", "参数"]
        return any(indicator in text.lower() for indicator in column_indicators)
    
    def _extract_conditions(self, query: str) -> List[QueryCondition]:
        """Extract query conditions"""
        conditions = []
        
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        condition = self._create_condition(pattern_type, groups, match)
                        if condition:
                            conditions.append(condition)
        
        return conditions
    
    def _create_condition(self, pattern_type: str, groups: Tuple, match) -> QueryCondition:
        """Create query condition object"""
        if pattern_type == "numeric_condition":
            if len(groups) == 2:
                entity, value = groups
                operator = self._extract_operator(match.group(0))
                return QueryCondition(
                    entity=entity,
                    operator=operator,
                    value=float(value),
                    confidence=0.8
                )
            elif len(groups) == 3:
                entity, min_val, max_val = groups
                return QueryCondition(
                    entity=entity,
                    operator="between",
                    value=(float(min_val), float(max_val)),
                    confidence=0.8
                )
        
        elif pattern_type == "text_condition":
            entity, value = groups
            operator = "contains" if "包含" in match.group(0) else "equals"
            return QueryCondition(
                entity=entity,
                operator=operator,
                value=value,
                confidence=0.7
            )
        
        elif pattern_type == "time_condition":
            if len(groups) == 2:
                entity, date = groups
                operator = self._extract_operator(match.group(0))
                return QueryCondition(
                    entity=entity,
                    operator=operator,
                    value=date,
                    confidence=0.9
                )
            elif len(groups) == 3:
                entity, start_date, end_date = groups
                return QueryCondition(
                    entity=entity,
                    operator="between",
                    value=(start_date, end_date),
                    confidence=0.9
                )
        
        return None
    
    def _extract_operator(self, text: str) -> str:
        """Extract operator from text"""
        if ">" in text or "大于" in text:
            return ">"
        elif "<" in text or "小于" in text:
            return "<"
        elif ">=" in text or "大于等于" in text:
            return ">="
        elif "<=" in text or "小于等于" in text:
            return "<="
        elif "=" in text or "等于" in text:
            return "="
        else:
            return "contains"
    
    def _identify_intent(self, query: str) -> str:
        """Identify query intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["分析", "分析", "统计", "总结"]):
            return "analysis"
        elif any(word in query_lower for word in ["比较", "对比", "差异"]):
            return "comparison"
        elif any(word in query_lower for word in ["趋势", "变化", "发展"]):
            return "trend"
        elif any(word in query_lower for word in ["分布", "频率", "比例"]):
            return "distribution"
        elif any(word in query_lower for word in ["相关性", "关联", "关系"]):
            return "correlation"
        else:
            return "general"
    
    def _identify_visualization_needs(self, query: str) -> List[str]:
        """Identify visualization needs"""
        query_lower = query.lower()
        needs = []
        
        if any(word in query_lower for word in ["图", "图表", "可视化", "画"]):
            if any(word in query_lower for word in ["柱状图", "条形图", "bar"]):
                needs.append("bar_chart")
            if any(word in query_lower for word in ["线图", "趋势图", "line"]):
                needs.append("line_chart")
            if any(word in query_lower for word in ["散点图", "scatter"]):
                needs.append("scatter_plot")
            if any(word in query_lower for word in ["饼图", "pie"]):
                needs.append("pie_chart")
            if any(word in query_lower for word in ["直方图", "分布图", "histogram"]):
                needs.append("histogram")
            if any(word in query_lower for word in ["热力图", "heatmap"]):
                needs.append("heatmap")
        
        return needs
    
    def _calculate_confidence(self, entities: List[QueryEntity], conditions: List[QueryCondition]) -> float:
        """Calculate parsing confidence"""
        if not entities and not conditions:
            return 0.0
        
        entity_conf = sum(e.confidence for e in entities) / len(entities) if entities else 0
        condition_conf = sum(c.confidence for c in conditions) / len(conditions) if conditions else 0
        
        return (entity_conf + condition_conf) / 2
    
    def _detect_comparison_and_separation(self, query: str, entities: List[QueryEntity]) -> Dict[str, Any]:
        """Intelligent detection of comparison and separation queries"""
        import re
        
        comparison_info = {
            "is_comparison": False,
            "is_separation": False,
            "materials": [],
            "comparison_target": None,
            "same_zeolite_required": False,
            "difference_metric": None
        }
        
        # Detect separation keywords (high priority)
        separation_keywords = ["分离", "区分", "分开", "选择性", "筛选出", "较好的分离", "separat", "distinguish", "selectivity"]
        for keyword in separation_keywords:
            if keyword in query:
                comparison_info["is_separation"] = True
                comparison_info["is_comparison"] = True  # Separation usually also requires comparison
                break
        
        # Detect comparison keywords
        comparison_keywords = ["比较", "对比", "差异", "相差", "不同", "区别", "哪几种", "哪些", "compare", "contrast", "differ", "separat", "strongest", "better", "best", "which zeolite", "which material", "stronger", "selectiv"]
        if not comparison_info["is_comparison"]:  # If not already marked as comparison
            for keyword in comparison_keywords:
                if keyword in query:
                    comparison_info["is_comparison"] = True
                    break
        
        # Extract candidate material names.
        # For chemical formulas (CO2, CH4) and Chinese — simple patterns suffice.
        # For natural-language names, the LLM in _extract_molecules_from_query handles it.
        detected_materials = []
        query_lower = query.lower()

        # 1. Chemical formulas: standalone uppercase+lowercase+digits like CO2, CH4, N2O, C2H6
        formula_pattern = re.compile(r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b')
        noise = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
                 'k', 'mol', 'log', 'max', 'min', 'doi', 'api', 'http', 'https',
                 'mfi', 'fau', 'cha', 'lta', 'ddr', 'mor', 'bea', 'mww', 'fer',
                 'zsm', 'si', 'al', 'na', 'ca', 'cu', 'fe', 'zn', 'ag', 'pt', 'li', 'mg', 'cs'}
        for m in formula_pattern.findall(query):
            if len(m) >= 2 and m.lower() not in noise:
                detected_materials.append(m.lower())

        # 2. Chinese molecule names between 和/与/、 (e.g. 甲烷和二氧化碳)
        cn_sep = re.search(r'([一-鿿]{1,6})(?:和|与|、)([一-鿿]{1,6})', query)
        if cn_sep:
            for g in [cn_sep.group(1), cn_sep.group(2)]:
                if g not in detected_materials:
                    detected_materials.append(g)

        comparison_info["materials"] = detected_materials[:5]
        
        # Detect if same zeolite is required
        same_zeolite_patterns = [
            "分子筛相同",
            "相同分子筛",
            "同一分子筛",
            "相同的分子筛",
            "在分子筛相同的情况下",
            "same zeolite"
        ]
        
        for pattern in same_zeolite_patterns:
            if pattern in query:
                comparison_info["same_zeolite_required"] = True
                break
        
        # Detect comparison metric
        if "扩散系数" in query or "diffusion" in query.lower():
            comparison_info["comparison_target"] = "diffusion_coefficient"
        
        if "相差" in query or "差异" in query:
            comparison_info["difference_metric"] = "numerical_difference"
        
        return comparison_info
