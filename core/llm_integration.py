"""
Large Language Model Integration and Prompt Engineering Module
"""
from openai import OpenAI
from typing import Dict, List, Any, Optional
from loguru import logger
from config import settings, PROMPT_TEMPLATES
import time

class LLMIntegration:
    """Large Language Model Integration"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """Initialize LLM integration"""
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.model = model or "deepseek-v4-pro"
        
        # Configure OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_response(self, query: str, context: str,
                         response_type: str = "analysis", is_comparison: bool = None,
                         molecule_a: str = None, molecule_b: str = None) -> Dict[str, Any]:
        """Generate response"""
        logger.info(f"Generating {response_type} type response")

        mol1 = molecule_a or "molecule A"
        mol2 = molecule_b or "molecule B"

        # Build prompt
        prompt = self._build_prompt(query, context, response_type, is_comparison=is_comparison,
                                     mol1=mol1, mol2=mol2)

        # Build messages
        # If is_comparison not explicitly passed, detect from query keywords as fallback
        if is_comparison is None:
            is_comparison = any(keyword in query.lower() for keyword in ["比较", "对比", "分离", "相差", "差异", "筛选", "哪几种", "哪些", "separation", "separate", "strongest", "better", "best", "which zeolite", "which material", "stronger", "selectivity", "selective", "compare", "contrast", "difference"])

        if is_comparison:
            system_prompt = f"""You are a materials science expert specializing in zeolite diffusion analysis. Answer strictly in English.

Your task is to analyze the ZEOLITE RANKING TABLE and DETAILED PAIRED DATA TABLE provided below and give a structured comparison for {mol1} vs {mol2} separation.

Output strictly in the following format (one paragraph per point):
1. Best zeolite (direct evidence): State the rank-1 zeolite from the ZEOLITE RANKING TABLE and its Log10(ratio) value.
2. Evidence: Provide exact {mol1} and {mol2} diffusion coefficients at similar temperatures for the top 1-2 zeolites, with DOI sources.
3. Comparative analysis: Compare Log10(ratio) values across the ranked zeolites and explain the differences.
4. Predicted candidates: If a "PREDICTED CANDIDATES (Tier 2)" section is present, mention the top 1-2 predicted zeolites with their scores and clearly state these are model predictions without direct experimental data. If no Tier 2 section exists, skip this point.
5. Conclusion: State which zeolite is best for {mol1}/{mol2} separation and why.

Rules:
- MUST cite specific numbers from the provided tables; never invent data
- For every numeric datum cited, include its DOI source
- Use Log10(ratio) as the primary metric (higher = better separation)
- Express diffusion coefficients in scientific notation (e.g. 1.35E-15 m2/s)
- Tier 1 = direct evidence (DOI-backed measurements, cite authoritatively)
- Tier 2 = predicted candidates (model-based, always qualify with "predicted by molecular similarity modeling" or "may be a candidate worth experimental investigation")"""
        else:
            system_prompt = """You are a materials science expert. Answer directly and concisely in English.

Rules:
1. Maximum 3-4 sentences
2. State the conclusion first, then briefly explain
3. Support with specific data values from the table
4. For every specific numeric datum you cite, include its DOI source if available in the table
5. When available, include concentration and experimental method context, not just value and temperature
6. Do not use special characters (#, *, -, bullet points)
7. Do not use headings or section labels
8. Plain text only"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Call LLM
            response = self._call_llm(messages, response_type)
            
            answer = response["content"]
            
            # No longer limit length, allow AI to fully elaborate
            # Commented out previous length limits
            # if is_comparison and len(answer) > 600:
            #     answer = answer[:600] + "..."
            # elif not is_comparison and len(answer) > 150:
            #     answer = answer[:150] + "..."
            
            return {
                "answer": answer,
                "model": self.model,
                "tokens_used": response["total_tokens"],
                "response_type": response_type
            }
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                "answer": f"Sorry, an error occurred while generating response: {str(e)}",
                "model": self.model,
                "tokens_used": 0,
                "response_type": response_type,
                "error": str(e)
            }
    
    def _build_prompt(self, query: str, context: str, response_type: str, is_comparison: bool = None,
                      mol1: str = "molecule A", mol2: str = "molecule B") -> str:
        """Build prompt"""
        if response_type == "analysis":
            return PROMPT_TEMPLATES["data_analysis"].format(
                context=context,
                question=query
            )
        elif response_type == "visualization":
            return PROMPT_TEMPLATES["visualization"].format(
                data_description=context,
                chart_type="auto",
                data=context
            )
        else:
            # Use passed is_comparison, fallback to keyword detection
            if is_comparison is None:
                is_comparison = any(keyword in query.lower() for keyword in ["比较", "对比", "分离", "相差", "差异", "筛选", "separation", "separate", "strongest", "better", "best", "which zeolite", "which material", "stronger", "selectivity", "selective", "compare", "contrast", "difference"])

            if is_comparison:
                return f"""
{context}

Question: {query}

Instructions:
The data above contains a ZEOLITE RANKING TABLE and a DETAILED PAIRED DATA TABLE.

1. Use the ZEOLITE RANKING TABLE to identify top-performing zeolites by Log10(ratio)
2. Use the DETAILED PAIRED DATA TABLE for exact {mol1}/{mol2} values at similar temperatures
3. Use specific numbers from the table as evidence
4. For each key comparison, include concentration and experimental method if available
5. Use Log10_ratio = log10(D_max / D_min) as primary separation metric (higher = better)
6. Optionally reference FULL FILTERED DATA for extra context, but conclusions must follow ranking + paired evidence

Only use data from the table above. Do not use data not present in the table.
For every specific numeric datum you cite, append the corresponding DOI source from the table.
Cite exact diffusion coefficients, temperatures, concentration/method context, and Log10_ratio evidence.
"""
            else:
                # General prompt
                return f"""
{context}

Question: {query}

Instructions:
1. Answer directly in 3-4 sentences
2. State the conclusion first
3. Support with specific data values from the table
4. For every specific numeric datum you cite, include its DOI source if available in the table
5. When available, include concentration and experimental method context
6. Do not use special characters (#, *, -, bullet points)
7. Do not use headings or section labels
8. Plain text, numbers and basic punctuation only
9. Do NOT provide dataset overview/range/count summaries unless explicitly requested
"""
    
    def generate_analysis_report(self, df_summary: Dict[str, Any], 
                               insights: Dict[str, Any], 
                               query: str) -> str:
        """Generate analysis report"""
        prompt = f"""
As a professional data analyst, please generate a comprehensive analysis report based on the following information:

Data Overview:
- Data size: {df_summary.get('shape', 'N/A')}
- Number of columns: {df_summary.get('total_columns', 'N/A')}
- Numeric columns: {df_summary.get('numeric_columns', 'N/A')}
- Categorical columns: {df_summary.get('categorical_columns', 'N/A')}

Data Insights:
{insights}

User Query: {query}

Please generate a professional report including:
1. Executive Summary
2. Data Quality Assessment
3. Key Findings and Patterns
4. Statistical Analysis and Trends
5. Conclusions and Recommendations

Please write in English, maintaining professionalism and readability.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a senior data analyst, skilled at writing professional data analysis reports."},
                {"role": "user", "content": prompt}
            ]
            response = self._call_llm(messages, "analysis")
            return response["content"]
            
        except Exception as e:
            logger.error(f"Failed to generate analysis report: {e}")
            return f"Error generating analysis report: {str(e)}"
    
    def generate_visualization_description(self, chart_type: str, 
                                         data_description: str) -> str:
        """Generate visualization description"""
        prompt = f"""
Please generate a professional description for the following visualization:

Chart Type: {chart_type}
Data Description: {data_description}

Please provide:
1. Main features of the chart
2. Key patterns in data distribution
3. Notable trends or anomalies
4. Professional interpretation of the data

Please write concisely and clearly.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a data visualization expert, skilled at interpreting various charts."},
                {"role": "user", "content": prompt}
            ]
            response = self._call_llm(messages, "visualization")
            return response["content"]
            
        except Exception as e:
            logger.error(f"Failed to generate visualization description: {e}")
            return f"Error generating visualization description: {str(e)}"
    
    def suggest_visualizations(self, df_summary: Dict[str, Any], 
                             query: str) -> List[Dict[str, Any]]:
        """Suggest visualization schemes"""
        prompt = f"""
Based on the following data characteristics and user query, please suggest appropriate visualization schemes:

Data Characteristics:
- Numeric columns: {df_summary.get('numeric_columns', [])}
- Categorical columns: {df_summary.get('categorical_columns', [])}
- Data size: {df_summary.get('shape', 'N/A')}

User Query: {query}

For each suggestion, please provide:
1. Chart type
2. Data columns used
3. Chart purpose
4. Expected insights

Please return the suggestion list in JSON format.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a data visualization expert, skilled at recommending appropriate chart types based on data characteristics and query requirements."},
                {"role": "user", "content": prompt}
            ]
            response = self._call_llm(messages, "visualization")
            
            # Try to parse JSON response
            import json
            try:
                suggestions = json.loads(response["content"])
                return suggestions if isinstance(suggestions, list) else []
            except:
                # If JSON parsing fails, return default suggestions
                return self._get_default_visualization_suggestions(df_summary)
            
        except Exception as e:
            logger.error(f"Failed to generate visualization suggestions: {e}")
            return self._get_default_visualization_suggestions(df_summary)
    
    def _get_default_visualization_suggestions(self, df_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get default visualization suggestions"""
        suggestions = []
        
        numeric_cols = df_summary.get('numeric_columns', [])
        categorical_cols = df_summary.get('categorical_columns', [])
        
        if numeric_cols:
            suggestions.append({
                "chart_type": "histogram",
                "columns": numeric_cols[:1],
                "purpose": "Data distribution analysis",
                "insights": "Understand the distribution characteristics of numeric variables"
            })
        
        if len(numeric_cols) >= 2:
            suggestions.append({
                "chart_type": "scatter",
                "columns": numeric_cols[:2],
                "purpose": "Correlation analysis",
                "insights": "Explore the relationship between two numeric variables"
            })
        
        if categorical_cols and numeric_cols:
            suggestions.append({
                "chart_type": "bar",
                "columns": [categorical_cols[0], numeric_cols[0]],
                "purpose": "Category comparison",
                "insights": "Compare numeric features across different categories"
            })
        
        return suggestions
    
    def extract_visualization_requirements(self, query: str) -> Dict[str, Any]:
        """Extract visualization requirements from query"""
        prompt = f"""
Please analyze the following user query and extract visualization requirements:

User Query: {query}

Please identify:
1. Required chart type
2. Data column requirements
3. Visualization purpose
4. Special requirements

Please return results in JSON format.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a data visualization requirements analyst, skilled at extracting visualization requirements from natural language queries."},
                {"role": "user", "content": prompt}
            ]
            response = self._call_llm(messages, "visualization")
            
            import json
            try:
                requirements = json.loads(response["content"])
                return requirements if isinstance(requirements, dict) else {}
            except:
                return {}
            
        except Exception as e:
            logger.error(f"Failed to extract visualization requirements: {e}")
            return {}
    
    def generate_comparison_analysis(self, data1: Dict[str, Any], 
                                   data2: Dict[str, Any], 
                                   comparison_type: str) -> str:
        """Generate comparison analysis"""
        prompt = f"""
Please perform a comparative analysis of the following two datasets:

Dataset 1: {data1}
Dataset 2: {data2}
Comparison Type: {comparison_type}

Please provide:
1. Key differences
2. Similarity analysis
3. Statistical significance
4. Business implications

Please write objectively and professionally.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a professional statistical analysis expert, skilled at conducting comparative data analysis."},
                {"role": "user", "content": prompt}
            ]
            response = self._call_llm(messages, "analysis")
            return response["content"]
            
        except Exception as e:
            logger.error(f"Failed to generate comparison analysis: {e}")
            return f"Error generating comparison analysis: {str(e)}"
    
    def _call_llm(self, messages: List[Dict], response_type: str) -> Dict[str, Any]:
        """Call LLM and process response"""
        # Set max tokens based on response type (increased to support more detailed responses)
        if response_type == "analysis":
            max_tokens = 8000
        elif response_type == "visualization":
            max_tokens = 4000
        else:
            max_tokens = 4000
        
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                break
            except Exception as e:
                logger.error(f"Error occurred: {e}")
                if 'Please reduce the length of the messages' in str(e):
                    logger.info('Truncating message length')
                    if len(messages) > 3:
                        messages.pop(3)
                    else:
                        messages.pop(1)
                elif 'per min' in str(e):
                    logger.info("Waiting 15 seconds")
                    time.sleep(15)
                else:
                    raise e
        
        return {
            "content": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.prompt_tokens + response.usage.completion_tokens
        }
