"""
Data Extraction and Filtering Module - Filter table data based on query conditions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from core.semantic_parser import QueryCondition, QueryEntity
from core.simple_dataframe import SimpleDataFrame, SimpleDataLoader

class DataExtractor:
    """Data Extractor"""
    
    def __init__(self):
        """Initialize data extractor"""
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']
    
    def load_table(self, file_path: str) -> SimpleDataFrame:
        """Load table data"""
        logger.info(f"Loading table file: {file_path}")
        
        try:
            # Use SimpleDataFrame to avoid pandas compatibility issues
            loader = SimpleDataLoader()
            return loader.load_table(file_path)
        except Exception as e:
            logger.error(f"Failed to load table: {e}")
            raise
    
    def _load_csv_simple(self, file_path: str) -> pd.DataFrame:
        import csv
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows:
            raise ValueError("File is empty")
        
        headers = rows[0]
        data_rows = rows[1:]
        
        # Create DataFrame dictionary
        df_dict = {}
        for i, header in enumerate(headers):
            col_data = []
            for row in data_rows:
                if i < len(row):
                    col_data.append(str(row[i]))
                else:
                    col_data.append('')
            df_dict[header] = col_data
        
        # Create DataFrame using pandas with dictionary method
        try:
            df = pd.DataFrame(df_dict)
            logger.info(f"Successfully loaded CSV, shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"DataFrame creation failed: {e}")
            # If pandas fails, raise exception
            raise
    
    def _load_excel_simple(self, file_path: str) -> pd.DataFrame:
        """Load Excel using simple method"""
        try:
            df = pd.read_excel(file_path)
            df = df.fillna('')
            for col in df.columns:
                df[col] = df[col].astype(str)
            logger.info(f"Successfully loaded Excel, shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Excel loading failed: {e}")
            raise
    
    def _load_json_simple(self, file_path: str) -> pd.DataFrame:
        """Load JSON using simple method"""
        try:
            df = pd.read_json(file_path)
            df = df.fillna('')
            for col in df.columns:
                df[col] = df[col].astype(str)
            logger.info(f"Successfully loaded JSON, shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"JSON loading failed: {e}")
            raise
    
    def extract_data(self, df: pd.DataFrame, entities: List[QueryEntity], 
                    conditions: List[QueryCondition], 
                    column_mappings: Dict[str, str]) -> pd.DataFrame:
        """Extract data based on query conditions"""
        logger.info("Starting data extraction")
        
        # Create data copy
        result_df = df.copy()
        
        # 1. Filter relevant columns based on entities
        relevant_columns = self._get_relevant_columns(df, entities, column_mappings)
        logger.info(f"Relevant columns: {relevant_columns}")
        
        # 2. Apply condition filters
        for condition in conditions:
            result_df = self._apply_condition(result_df, condition, column_mappings)
        
        # 3. Select relevant columns
        if relevant_columns:
            available_columns = [col for col in relevant_columns if col in result_df.columns]
            if available_columns:
                result_df = result_df[available_columns]
        
        logger.info(f"Data shape after extraction: {result_df.shape}")
        return result_df
    
    def _get_relevant_columns(self, df: pd.DataFrame, entities: List[QueryEntity], 
                            column_mappings: Dict[str, str]) -> List[str]:
        """Get relevant columns"""
        relevant_columns = []
        
        for entity in entities:
            if entity.text in column_mappings:
                mapped_column = column_mappings[entity.text]
                if mapped_column in df.columns:
                    relevant_columns.append(mapped_column)
        
        # If no mapping found, try direct matching
        if not relevant_columns:
            for entity in entities:
                for col in df.columns:
                    if self._is_column_relevant(col, entity.text):
                        relevant_columns.append(col)
        
        return list(set(relevant_columns))
    
    def _is_column_relevant(self, column_name: str, entity_text: str) -> bool:
        """Check if column is relevant to entity"""
        column_lower = column_name.lower()
        entity_lower = entity_text.lower()
        
        # Direct match
        if entity_lower in column_lower or column_lower in entity_lower:
            return True
        
        # Partial match
        entity_words = set(entity_lower.split())
        column_words = set(column_lower.split())
        
        if entity_words.intersection(column_words):
            return True
        
        return False
    
    def _apply_condition(self, df: pd.DataFrame, condition: QueryCondition, 
                        column_mappings: Dict[str, str]) -> pd.DataFrame:
        """Apply single condition"""
        logger.info(f"Applying condition: {condition}")
        
        # Find corresponding column
        target_column = None
        if condition.entity in column_mappings:
            target_column = column_mappings[condition.entity]
        else:
            # Try direct column name match
            for col in df.columns:
                if self._is_column_relevant(col, condition.entity):
                    target_column = col
                    break
        
        if target_column is None or target_column not in df.columns:
            logger.warning(f"Column not found for condition: {condition.entity}")
            return df
        
        # Apply condition
        try:
            if condition.operator == ">":
                mask = df[target_column] > condition.value
            elif condition.operator == "<":
                mask = df[target_column] < condition.value
            elif condition.operator == ">=":
                mask = df[target_column] >= condition.value
            elif condition.operator == "<=":
                mask = df[target_column] <= condition.value
            elif condition.operator == "=":
                mask = df[target_column] == condition.value
            elif condition.operator == "contains":
                mask = df[target_column].astype(str).str.contains(str(condition.value), case=False, na=False)
            elif condition.operator == "between":
                min_val, max_val = condition.value
                mask = (df[target_column] >= min_val) & (df[target_column] <= max_val)
            else:
                logger.warning(f"Unsupported operator: {condition.operator}")
                return df
            
            filtered_df = df[mask]
            logger.info(f"Rows after condition filter: {len(filtered_df)}")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error applying condition: {e}")
            return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get data summary"""
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes,
            "null_counts": {col: sum(1 for val in df[col] if val is None or val == '' or val == 'nan') for col in df.columns},
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric column summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df.describe()
        
        # Categorical column summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary["categorical_summary"][col] = {
                "unique_count": df.nunique().get(col, 0),
                "top_values": df.value_counts_single(col)[:5]
            }
        
        return summary
    
    def get_statistical_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical insights"""
        insights = {
            "correlations": {},
            "trends": {},
            "outliers": {},
            "distributions": {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Calculate correlations
            corr_matrix = df[numeric_cols].corr()
            insights["correlations"] = corr_matrix
        
        # Analyze trends
        for col in numeric_cols:
            if df[col].dtype in ['datetime64[ns]', 'int64', 'float64']:
                try:
                    # Calculate linear trend
                    x = np.arange(len(df))
                    y = df[col].dropna()
                    if len(y) > 1:
                        slope = np.polyfit(x[:len(y)], y, 1)[0]
                        insights["trends"][col] = {
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
                insights["outliers"][col] = {
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100
                }
        
        return insights
    
    def prepare_data_for_visualization(self, df: pd.DataFrame, chart_type: str) -> Dict[str, Any]:
        """Prepare data for visualization"""
        prepared_data = {
            "data": df,
            "columns": list(df.columns),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": list(df.select_dtypes(include=['datetime64[ns]']).columns)
        }
        
        # Special handling based on chart type
        if chart_type in ["histogram", "density"]:
            # Prepare data for histogram
            prepared_data["bins"] = self._calculate_optimal_bins(df)
        
        elif chart_type in ["scatter", "line"]:
            # Prepare data for scatter and line charts
            prepared_data["x_axis_options"] = prepared_data["numeric_columns"]
            prepared_data["y_axis_options"] = prepared_data["numeric_columns"]
        
        elif chart_type in ["bar", "pie"]:
            # Prepare data for bar and pie charts
            prepared_data["category_options"] = prepared_data["categorical_columns"]
            prepared_data["value_options"] = prepared_data["numeric_columns"]
        
        return prepared_data
    
    def _calculate_optimal_bins(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate optimal bin count"""
        bins = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Use Sturges formula
            n = len(df[col].dropna())
            bins[col] = int(np.ceil(np.log2(n)) + 1) if n > 0 else 10
        
        return bins
    
    def filter_by_date_range(self, df: pd.DataFrame, date_column: str, 
                           start_date: str, end_date: str) -> pd.DataFrame:
        """Filter data by date range"""
        if date_column not in df.columns:
            return df
        
        try:
            # Ensure date column is datetime type
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Apply date filter
            mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
            return df[mask]
        except Exception as e:
            logger.error(f"Date filtering failed: {e}")
            return df
    
    def group_and_aggregate(self, df: pd.DataFrame, group_columns: List[str], 
                          agg_columns: List[str], agg_functions: List[str]) -> pd.DataFrame:
        """Group and aggregate data"""
        try:
            # Build aggregation dictionary
            agg_dict = {}
            for col in agg_columns:
                if col in df.columns:
                    agg_dict[col] = agg_functions
            
            # Execute group aggregation
            grouped = df.groupby(group_columns).agg(agg_dict)
            
            # Flatten column names
            grouped.columns = [f"{col}_{func}" for col, func in zip(grouped.columns.get_level_values(0), 
                                                                   grouped.columns.get_level_values(1))]
            
            return grouped.reset_index()
        except Exception as e:
            logger.error(f"Group aggregation failed: {e}")
            return df


