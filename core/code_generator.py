"""
Code Generation and Execution Module - For generating plot code based on user instructions
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from typing import Dict, List, Any, Optional
from loguru import logger
import traceback

from .llm_integration import LLMIntegration


class CodeGenerator:
    """Code Generator and Executor - For generating and executing visualization code"""
    
    def __init__(self, llm_integration: LLMIntegration):
        """Initialize code generator"""
        self.llm_integration = llm_integration
        logger.info("Code generator initialized")
    
    def detect_visualization_intent(self, query: str) -> bool:
        """Detect if user wants to create a chart"""
        visualization_keywords = [
            "画", "绘制", "绘图", "图表", "可视化", "显示图", "生成图",
            "plot", "chart", "graph", "visualize", "draw", "show chart"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in visualization_keywords)
    
    def generate_plot_code(self, query: str, df: pd.DataFrame, 
                          column_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use LLM to generate plot code"""
        logger.info(f"Generating plot code for query: {query}")
        
        # Prepare data summary
        data_summary = self._prepare_data_summary(df, column_info)
        
        # Add column semantic hints based on actual sample values
        column_hints = self._prepare_column_hints(df)
        
        # Build prompt
        prompt = f"""
You are a Python data visualization expert. Please generate complete Python code to create charts based on user requirements.

User requirement: {query}

Data information:
{data_summary}

COLUMN SEMANTICS (very important - use the correct column for filtering):
{column_hints}

Important understanding:
1. To filter by ZEOLITE TYPE (e.g. MFI, FAU, LTA), use the 'std_zeolite_name' column
   - Example: df[df['std_zeolite_name'].str.contains('MFI', case=False, na=False)]
2. To filter by GAS/MATERIAL (e.g. methane, CO2), use the 'guest_molecule' column
   - Example: df[df['guest_molecule'].str.contains('methane', case=False, na=False)]
3. Numeric columns (temperature_value, diffusion_coefficient_value, KD_A, PLD_A, etc.)
   are already pre-converted to float64. Use them directly — no str.extract needed.
   Drop NaN rows before plotting: df = df.dropna(subset=['temperature_value', ...])
4. Histogram MUST use ax.hist() or sns.histplot(), do NOT use line chart or bar chart
5. Always check if filtered data is empty before plotting, and print a warning if so

Requirements:
1. Generate complete, executable Python code
2. Code must use matplotlib or seaborn library
3. Code must include the following parts:
   - Import necessary libraries
   - Data filtering using CORRECT column (std_zeolite_name for zeolite, guest_molecule for gas)
   - Data preprocessing (extract numeric values, drop NaN/Variable)
   - Validate filtered data is not empty
   - Create chart
   - Set title and labels
   - Convert chart to base64 encoded image string named 'image_base64'
4. Ensure code can be executed directly
5. Use variable df to represent the data frame

Please only return Python code, no explanation.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a Python data visualization code generation expert. Only return executable Python code, do not include any explanation."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_integration._call_llm(messages, "visualization")
            code = response["content"]
            
            # Clean code (remove markdown markers)
            code = self._clean_code(code)
            
            logger.info(f"Generated code length: {len(code)} characters")
            logger.debug(f"生成的代码:\n{code}")
            
            return {
                "success": True,
                "code": code,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            return {
                "success": False,
                "error": str(e),
                "code": None
            }
    
    def execute_plot_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute plot code and return image"""
        logger.info("Executing plot code")
        logger.info(f"Generated code:\n{code}")
        
        try:
            # Ensure df is pandas DataFrame
            # If it's SimpleDataFrame, convert to pandas DataFrame
            if hasattr(df, 'data') and hasattr(df, 'columns') and not isinstance(df, pd.DataFrame):
                # This is SimpleDataFrame, need to convert
                logger.info("Detected SimpleDataFrame, converting to pandas DataFrame")
                # Deep clean data to ensure no nested numpy arrays
                clean_data = {}
                for col in df.columns:
                    col_values = []
                    for val in df.data[col]:
                        # Convert all values to Python native types
                        if isinstance(val, np.ndarray):
                            val = val.tolist() if val.size > 1 else (val.item() if val.size == 1 else None)
                        elif isinstance(val, (np.integer, np.floating)):
                            val = val.item()
                        elif isinstance(val, np.bool_):
                            val = bool(val)
                        col_values.append(val)
                    clean_data[col] = col_values
                pandas_df = pd.DataFrame(clean_data)
            elif isinstance(df, pd.DataFrame):
                # Even for pandas DataFrame, check and clean possible numpy arrays
                pandas_df = df.copy()
                for col in pandas_df.columns:
                    try:
                        if pandas_df[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                            pandas_df[col] = pandas_df[col].apply(
                                lambda x: x.tolist() if isinstance(x, np.ndarray) and x.size > 1 
                                else (x.item() if isinstance(x, np.ndarray) and x.size == 1 else x)
                            )
                    except Exception as col_err:
                        logger.warning(f"Column {col} cleanup failed: {col_err}")
            else:
                pandas_df = df
            
            # Ensure all numeric columns have correct types
            # All columns start as object (SimpleDataFrame stores everything as strings).
            # Two-pass heuristic to decide which ones to convert:
            #   Pass 1: >50% of ALL rows are numeric → convert (e.g. temperature, diffusion)
            #   Pass 2: >80% of NON-EMPTY rows are numeric AND ≥50 numeric values
            #            → convert (e.g. si_al_ratio 35%/90%, pressure_value 28%/98%)
            for col in list(pandas_df.columns):
                try:
                    numeric_col = pd.to_numeric(pandas_df[col], errors='coerce')
                    numeric_count = numeric_col.notna().sum()
                    total_count = len(numeric_col)
                    if numeric_count == 0:
                        continue
                    # Count original rows that are non-empty (not NaN / whitespace / "nan")
                    stripped = pandas_df[col].astype(str).str.strip()
                    non_empty_before = (
                        pandas_df[col].notna() &
                        (stripped != '') &
                        (stripped.str.lower() != 'nan')
                    ).sum()
                    convert = False
                    reason = ""
                    if numeric_count > total_count * 0.5:
                        convert = True
                        reason = f">50% of all rows ({numeric_count}/{total_count})"
                    elif non_empty_before > 0 and numeric_count >= 50 and (numeric_count / non_empty_before) > 0.8:
                        convert = True
                        reason = (f">80% of non-empty ({numeric_count}/{non_empty_before}) "
                                  f"with {numeric_count}≥50 numeric values")
                    if convert:
                        pandas_df[col] = numeric_col
                        logger.debug(f"Column '{col}' → numeric ({reason})")
                except Exception:
                    pass

            logger.info(f"Data preparation complete, shape: {pandas_df.shape}")
            logger.info(f"Data types:\n{pandas_df.dtypes}")
            
            # Prepare execution environment
            exec_globals = {
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'df': pandas_df,
                'io': io,
                'base64': base64,
                'matplotlib': matplotlib,
                '__builtins__': __builtins__
            }
            
            exec_locals = {}
            
            # Execute code
            exec(code, exec_globals, exec_locals)
            
            # Get generated image
            image_base64 = exec_locals.get('image_base64')
            
            if not image_base64:
                # Try to get from global variables
                image_base64 = exec_globals.get('image_base64')
            
            if not image_base64:
                # If code didn't generate image_base64, try to get from current figure
                if plt.get_fignums():
                    fig = plt.gcf()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close(fig)
            
            if image_base64:
                logger.info("Code executed successfully, image generated")
                return {
                    "success": True,
                    "image": image_base64,
                    "format": "png"
                }
            else:
                logger.warning("Code executed but no image found")
                return {
                    "success": False,
                    "error": "Code executed but no image generated. Please ensure the code creates a chart and converts to base64."
                }
                
        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            logger.error(f"Code execution failed: {error_msg}")
            logger.error(f"Full error traceback:\n{error_trace}")
            logger.error(f"Executed code:\n{code}")
            
            return {
                "success": False,
                "error": f"Code execution failed: {error_msg}",
                "traceback": error_trace
            }
    
    def _clean_code(self, code: str) -> str:
        """Clean code, remove markdown markers etc"""
        # Remove markdown code block markers
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        return code

    def _prepare_column_hints(self, df) -> str:
        """Prepare column semantic hints based on actual sample values"""
        hints = []
        for col in df.columns:
            try:
                if hasattr(df, 'data'):
                    samples = [str(df.data[col][i]) for i in range(min(3, len(df))) if df.data[col][i] is not None]
                else:
                    samples = [str(v) for v in df[col].dropna().head(3).tolist()]
                hints.append(f"  - '{col}': e.g. {', '.join(samples)}")
            except:
                hints.append(f"  - '{col}'")
        return '\n'.join(hints)

    def _prepare_data_summary(self, df: pd.DataFrame,
                             column_info: Dict[str, Any] = None) -> str:
        """Prepare data summary"""
        summary_parts = []
        
        summary_parts.append(f"Data shape: {df.shape[0]} rows x {df.shape[1]} columns")
        summary_parts.append(f"\nColumn names: {', '.join(df.columns)}")
        
        # Numeric column info (compatible with SimpleDataFrame)
        if hasattr(df, 'select_dtypes'):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = [col for col in df.columns 
                          if df.dtypes.get(col, 'object') in ['int64', 'float64']]
        
        if len(numeric_cols) > 0:
            summary_parts.append(f"\nNumeric columns ({len(numeric_cols)}):")
            for col in numeric_cols[:10]:  # Limit to first 10
                try:
                    # Compatible with SimpleDataFrame
                    if hasattr(df[col], 'dropna'):
                        col_data = df[col].dropna()
                        numeric_vals = [float(v) for v in col_data if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit())]
                    else:
                        numeric_vals = [float(v) for v in df[col] if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit())]
                    
                    if len(numeric_vals) > 0:
                        summary_parts.append(
                            f"  - {col}: range [{min(numeric_vals):.2f}, {max(numeric_vals):.2f}], "
                            f"mean {np.mean(numeric_vals):.2f}"
                        )
                except:
                    summary_parts.append(f"  - {col}: (cannot calculate statistics)")
        
        # Categorical column info (compatible with SimpleDataFrame)
        if hasattr(df, 'select_dtypes'):
            categorical_cols = df.select_dtypes(include=['object']).columns
        else:
            categorical_cols = [col for col in df.columns 
                              if df.dtypes.get(col, 'object') == 'object']
        if len(categorical_cols) > 0:
            summary_parts.append(f"\nCategorical columns ({len(categorical_cols)}):")
            for col in categorical_cols[:10]:  # Limit to first 10
                try:
                    # Compatible with SimpleDataFrame
                    if hasattr(df[col], 'nunique'):
                        unique_count = df[col].nunique()
                    elif hasattr(df[col], 'unique'):
                        unique_count = len(df[col].unique())
                    else:
                        # SimpleDataFrame: manually count unique values
                        unique_vals = set(str(v) for v in df[col] if v is not None and str(v).strip())
                        unique_count = len(unique_vals)
                    summary_parts.append(f"  - {col}: {unique_count} unique values")
                except:
                    summary_parts.append(f"  - {col}: (cannot count)")
        
        # Add column info (if provided)
        if column_info:
            summary_parts.append("\nColumn details:")
            for col, info in list(column_info.items())[:10]:
                summary_parts.append(f"  - {col}: {info.get('dtype', 'unknown')}")
        
        return "\n".join(summary_parts)

