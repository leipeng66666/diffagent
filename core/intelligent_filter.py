"""
Intelligent Filtering Module - Use LLM to identify columns and conditions to filter
"""
from typing import Dict, List, Any, Optional
from loguru import logger
import json
import re


class IntelligentFilter:
    """Intelligent Filter - Use LLM to understand queries and identify filter conditions"""
    
    def __init__(self, llm_integration):
        """Initialize intelligent filter
        
        Args:
            llm_integration: LLM integration object
        """
        self.llm = llm_integration
        logger.info("Initializing intelligent filter")
    
    def identify_filter_columns(self, query: str, available_columns: List[str], 
                                sample_data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Identify columns and conditions to filter
        
        Args:
            query: User query
            available_columns: List of available column names
            sample_data: Sample data for each column
            
        Returns:
            Filter information dictionary
        """
        logger.info(f"Identifying filter conditions in query: {query}")
        
        # Pre-detect comparison intent from keywords before calling LLM
        q_lower = query.lower()
        _comp_kws = ["separation", "separate", "strongest", "stronger", "selectivity",
                     "selective", "best", "better", "which zeolite", "which molecular",
                     "compare", "contrast", "difference", "exhibit",
                     "比较", "对比", "分离", "哪种", "哪些", "筛选", "更能", "分离效果",
                     "separation performance", "stronger separation"]
        _predetected_comparison = any(kw in q_lower for kw in _comp_kws)
        
        # Build prompt
        prompt = self._build_filter_identification_prompt(query, available_columns, sample_data)
        
        # Call LLM
        messages = [
            {
                "role": "system",
                "content": """You are a data analysis expert. Analyze user queries to identify columns and conditions to filter.
Return JSON format, do not include any other text."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = self.llm._call_llm(messages, "analysis")
            content = response["content"]
            
            # Extract JSON
            filter_info = self._extract_json(content)
            
            if filter_info:
                # Override with pre-detected comparison if LLM missed it
                if _predetected_comparison:
                    filter_info["is_comparison"] = True
                    filter_info["query_type"] = "comparison"
                logger.info(f"Filter conditions identified: {filter_info}")
                return filter_info
            else:
                logger.warning("LLM returned invalid JSON")
                result = self._fallback_filter_identification(query, available_columns)
                if _predetected_comparison:
                    result["is_comparison"] = True
                    result["query_type"] = "comparison"
                return result
                
        except Exception as e:
            logger.error(f"LLM identification failed: {e}")
            result = self._fallback_filter_identification(query, available_columns)
            if _predetected_comparison:
                result["is_comparison"] = True
                result["query_type"] = "comparison"
            return result
    
    def _build_filter_identification_prompt(self, query: str, columns: List[str], 
                                           sample_data: Dict[str, List[Any]]) -> str:
        """Build filter identification prompt"""
        
        # Prepare column information
        columns_info = []
        for col in columns[:20]:  # Limit columns to avoid overly long prompt
            samples = sample_data.get(col, [])[:3]
            sample_str = ", ".join([str(s) for s in samples if s is not None])
            columns_info.append(f"  - {col}: sample data [{sample_str}]")
        
        prompt = f"""User query: "{query}"

Available columns:
{chr(10).join(columns_info)}

Please analyze the query and identify:
1. Columns to filter (column name)
2. Filter conditions (value or value range)
3. Columns to return (if explicitly specified in query)
4. Query type: is this a COMPARISON query (comparing multiple materials/zeolites, asking which is better/stronger, separation performance, selectivity) or a LOOKUP query (asking for specific values)?

Return in JSON format:
{{
  "query_type": "comparison" or "lookup",
  "is_comparison": true or false,
  "filter_columns": [
    {{
      "column": "column_name",
      "condition": "condition_type(equals/contains/range/greater/less)",
      "value": "filter_value",
      "reason": "why this column was selected"
    }}
  ],
  "return_columns": ["column1", "column2"],
  "reasoning": "overall analysis reasoning"
}}

Examples of COMPARISON queries:
- "which zeolite has the best separation between methane and CO2"
- "compare CH4 and CO2 diffusion coefficients"
- "which material shows stronger selectivity"
- "哪种分子筛对甲烷和二氧化碳的分离效果最好"

Examples of LOOKUP queries:
- "what is the diffusion coefficient for MFI at 300K"
- "show me data for FAU zeolite"
- "list all methane data"

Return only JSON, no other content:"""
        
        return prompt
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text"""
        # Try direct parsing
        try:
            return json.loads(text)
        except:
            pass
        
        # Try to extract JSON from code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Try to find content between first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
        
        return None
    
    def _fallback_filter_identification(self, query: str, columns: List[str]) -> Dict[str, Any]:
        """Fallback rule-based filter identification"""
        logger.info("Using fallback rules for filter identification")
        
        filter_columns = []
        query_lower = query.lower()
        
        # Check common column name patterns
        patterns = {
            'zeolite': ['分子筛', 'zeolite', 'mfi', 'lta', 'fau', 'mor', 'bea'],
            'material': ['材料', 'material', '物质', '甲烷', 'methane', '二氧化碳', 'co2', 'carbon dioxide'],
            'temperature': ['温度', 'temperature', 'k', '度'],
            'value': ['扩散系数', 'diffusion', 'coefficient', '数值', 'value'],
        }
        
        for col in columns:
            col_lower = col.lower()
            for pattern_key, keywords in patterns.items():
                if pattern_key in col_lower or col in ['zeolite', 'material', 'temperature', 'value']:
                    for keyword in keywords:
                        if keyword in query_lower:
                            # Extract value
                            value = self._extract_value_for_keyword(query, keyword)
                            if value:
                                filter_columns.append({
                                    "column": col,
                                    "condition": "contains",
                                    "value": value,
                                    "reason": f"Query contains keyword '{keyword}'"
                                })
                            break
        
        return {
            "filter_columns": filter_columns,
            "return_columns": columns[:10],  # Return first 10 columns by default
            "reasoning": "Fallback using rule matching"
        }
    
    def _extract_value_for_keyword(self, query: str, keyword: str) -> Optional[str]:
        """Extract related value for keyword"""
        query_lower = query.lower()
        
        # Temperature extraction
        if keyword in ['温度', 'temperature', 'k', '度']:
            temp_match = re.search(r'(\d+)\s*k', query_lower)
            if temp_match:
                return temp_match.group(1)
        
        # Zeolite name extraction
        if keyword in ['分子筛', 'zeolite']:
            zeolite_match = re.search(r'(mfi|lta|fau|mor|bea|zsm)', query_lower)
            if zeolite_match:
                return zeolite_match.group(1).upper()
        
        # Material name extraction
        if keyword in ['甲烷', 'methane']:
            return 'methane'
        if keyword in ['二氧化碳', 'co2', 'carbon dioxide']:
            return 'carbon dioxide'
        
        return None
    
    def apply_filters(self, data, filter_info: Dict[str, Any]) -> Any:
        """Apply filter conditions to data
        
        Args:
            data: SimpleDataFrame data
            filter_info: Filter information
            
        Returns:
            Filtered data
        """
        if not filter_info.get("filter_columns"):
            logger.info("No filter conditions, returning all data")
            return data
        
        logger.info(f"Applying {len(filter_info['filter_columns'])} filter conditions")
        
        # Get all row indices
        total_rows = data.shape[0]
        keep_indices = set(range(total_rows))
        
        for filter_col in filter_info["filter_columns"]:
            column = filter_col["column"]
            condition = filter_col["condition"]
            value = str(filter_col["value"]).lower()
            
            if column not in data.columns:
                logger.warning(f"Column '{column}' does not exist, skipping this filter condition")
                continue
            
            logger.info(f"Filtering: {column} {condition} {value}")
            
            # Get column data
            col_data = data.data[column]
            
            # Apply filter condition
            matching_indices = set()
            for idx in range(total_rows):
                cell_value = str(col_data[idx]).lower() if col_data[idx] is not None else ""
                
                if condition == "equals":
                    if cell_value == value:
                        matching_indices.add(idx)
                elif condition == "contains":
                    if value in cell_value:
                        matching_indices.add(idx)
                elif condition == "greater":
                    try:
                        if float(cell_value) > float(value):
                            matching_indices.add(idx)
                    except:
                        pass
                elif condition == "less":
                    try:
                        if float(cell_value) < float(value):
                            matching_indices.add(idx)
                    except:
                        pass
                elif condition == "range":
                    # value should be "min,max" format
                    try:
                        min_val, max_val = value.split(',')
                        cell_num = float(cell_value)
                        if float(min_val) <= cell_num <= float(max_val):
                            matching_indices.add(idx)
                    except:
                        pass
            
            # Intersection (all conditions must be satisfied)
            keep_indices = keep_indices & matching_indices
            logger.info(f"  Matched {len(matching_indices)} rows, remaining {len(keep_indices)} rows")
        
        # Create filtered data
        if not keep_indices:
            logger.warning("No matching data after filtering")
            return data  # Return empty DataFrame
        
        # Convert to list and sort
        keep_indices_list = sorted(list(keep_indices))
        
        # Create new data
        filtered_data = {}
        for col in data.columns:
            filtered_data[col] = [data.data[col][idx] for idx in keep_indices_list]
        
        # Import SimpleDataFrame
        from core.simple_dataframe import SimpleDataFrame
        result = SimpleDataFrame(filtered_data)
        
        logger.info(f"Filtering complete, kept {result.shape[0]} rows of data")
        return result


