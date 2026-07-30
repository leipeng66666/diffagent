"""
Table Data Visualization Q&A AI Agent - Main Class
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import os
import json

# Import core modules
from core.semantic_parser import SemanticParser
from core.synonym_mapper import SynonymMapper
from core.data_extractor import DataExtractor
from core.rag_engine import HybridRAGEngine
from core.llm_integration import LLMIntegration
from core.visualization_engine import VisualizationEngine
from core.intelligent_column_mapper import IntelligentColumnMapper
from core.intelligent_filter import IntelligentFilter
from core.graphrag_engine import GraphRAGEngine
from core.code_generator import CodeGenerator
from core.prediction_integrator import PredictionIntegrator
from core.project_graph_builder import ProjectGraphBuilder
from config import settings

class TableAgent:
    """Table Data Visualization Q&A AI Agent"""
    
    def __init__(self, api_key: str = None):
        """Initialize AI Agent"""
        logger.info("Initializing Table Data AI Agent")
        
        # Initialize components
        self.semantic_parser = SemanticParser()
        self.synonym_mapper = SynonymMapper()
        self.data_extractor = DataExtractor()
        
        # Initialize RAG engine with local model path
        local_model_path = getattr(settings, 'LOCAL_MODEL_PATH', './models/all-MiniLM-L6-v2')
        self.rag_engine = HybridRAGEngine(local_model_path=local_model_path)
        
        self.llm_integration = LLMIntegration(api_key=api_key)
        self.visualization_engine = VisualizationEngine()
        
        # Initialize intelligent column mapper
        self.column_mapper = IntelligentColumnMapper(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL
        )
        
        # Initialize intelligent filter
        self.intelligent_filter = IntelligentFilter(self.llm_integration)

        # Initialize project graph builder (shared pre-built graph from project1+project2)
        self.project_graph_builder = ProjectGraphBuilder()

        # Initialize GraphRAG engine with project graph builder
        self.graphrag_engine = GraphRAGEngine(
            self.rag_engine, self.llm_integration,
            project_graph_builder=self.project_graph_builder
        )
        
        # Initialize code generator
        self.code_generator = CodeGenerator(self.llm_integration)

        # Initialize prediction integrator (Project0 bridge for Tier 2 predicted candidates)
        self.prediction_integrator = PredictionIntegrator(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL
        )

        # Current data state
        self.current_data = None
        self.current_file_path = None
        self.data_summary = None
        
        logger.info("AI Agent initialization complete")
    
    def load_table(self, file_path: str) -> Dict[str, Any]:
        """Load table data"""
        logger.info(f"Loading table file: {file_path}")
        
        try:
            # Load data
            self.current_data = self.data_extractor.load_table(file_path)
            self.current_file_path = file_path
            
            # Generate data summary
            self.data_summary = self.data_extractor.get_data_summary(self.current_data)
            
            # Build RAG index
            self.rag_engine.index_data(self.current_data, os.path.basename(file_path))
            
            # Build GraphRAG knowledge graph
            try:
                self.graphrag_engine.build_knowledge_graph(self.current_data)
                logger.info("GraphRAG knowledge graph built successfully")
            except Exception as e:
                logger.warning(f"GraphRAG knowledge graph build failed: {e}")
            
            logger.info(f"Table loaded successfully, shape: {self.current_data.shape}")
            
            return {
                "success": True,
                "message": "Table loaded successfully",
                "data_summary": self.data_summary,
                "shape": self.current_data.shape
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to load table: {e}")
            return {
                "success": False,
                "message": f"Failed to load table: {str(e)}",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query (auto-routes between GraphRAG and QA based on LLM understanding)"""
        logger.info(f"Processing query: {query}")

        if self.current_data is None:
            return {
                "success": False,
                "message": "Please load table data first",
                "error": "No data loaded"
            }

        # Detect if it's a visualization request
        is_visualization_request = self.code_generator.detect_visualization_intent(query)
        
        if is_visualization_request:
            logger.info("Visualization request detected, using code generation")
            return self._process_visualization_query(query)
        
        try:
            # 1. Semantic parsing (including unit recognition)
            parsed_query = self.semantic_parser.parse_query(query)
            logger.info(f"Parse result: {parsed_query}")
            
            # 2. Process unit information
            unit_analysis = self._analyze_units_in_data(parsed_query.get("units", []))
            
            # 3. Use intelligent filter to identify columns to filter
            logger.info("📊 Using LLM to intelligently identify filter conditions...")
            sample_data = {}
            for col in self.current_data.columns[:20]:  # Limit column count
                sample_data[col] = self.current_data.data[col][:5]  # First 5 rows as sample

            # Get 3-layer column mapping from IntelligentColumnMapper
            mapping = self.column_mapper.map_query_to_columns(
                query,
                list(self.current_data.columns),
                sample_data
            )
            logger.info(f"✓ Column mapping result: material_col={mapping.get('material_column')}, zeolite_col={mapping.get('zeolite_column')}")

            # Use LLM query understanding from the 3-layer mapper (replaces keyword-based detection)
            detected_molecules = mapping.get("detected_materials", [])
            query_mode = mapping.get("query_type", "general")
            is_separation = mapping.get("is_separation", False)
            has_specific_zeolite = mapping.get("specific_zeolite") is not None

            # Fallback: if LLM understanding failed, use semantic parser's keyword detection
            kw_materials = parsed_query.get("comparison_info", {}).get("materials", [])
            kw_is_sep = parsed_query.get("comparison_info", {}).get("is_separation", False)
            if not detected_molecules and kw_materials:
                detected_molecules = kw_materials
                logger.info(f"✓ Fallback to keyword molecules: {detected_molecules}")
            if query_mode == "general":
                if kw_is_sep:
                    is_separation = True
                    query_mode = "ranking"
                    logger.info(f"✓ Fallback to keyword separation detection")
                elif len(detected_molecules) >= 2:
                    query_mode = "ranking"
                    logger.info(f"✓ Fallback: 2+ molecules → ranking mode")

            needs_prediction = mapping.get("needs_prediction", False)
            query_route = mapping.get("route", "qa")
            entity_count = mapping.get("entity_count", 0)
            logger.info(f"✓ Query understanding: mode={query_mode}, separation={is_separation}, "
                        f"molecules={detected_molecules}, zeolite={mapping.get('specific_zeolite')}, "
                        f"predict={needs_prediction}, route={query_route}, entities={entity_count}")

            # =================================================================
            # AUTO-ROUTE: GraphRAG for single-entity exploration queries
            # =================================================================
            if query_route == "graphrag":
                anchor_type = None
                anchor_value = None

                if detected_molecules:
                    anchor_type = "guest"
                    anchor_value = detected_molecules[0]
                elif mapping.get("specific_zeolite"):
                    anchor_type = "zeolite"
                    anchor_value = mapping.get("specific_zeolite")
                elif mapping.get("zeolite_column"):
                    # Try to extract zeolite from the data itself
                    anchor_type = "zeolite"
                    anchor_value = mapping.get("specific_zeolite") or "MFI"

                if anchor_type and anchor_value:
                    logger.info(f"→ GraphRAG route: anchor={anchor_type}:{anchor_value}")
                    graph_result = self.graphrag_engine.analyze_with_project_graph(
                        query, anchor_type, anchor_value
                    )
                    if "error" in graph_result:
                        logger.warning(f"GraphRAG failed: {graph_result['error']}, falling back to QA")
                        # Fall through to standard QA pipeline below
                    else:
                        return {
                            "success": True,
                            "query": query,
                            "parsed_query": {
                                "query": query,
                                "intent": query_mode,
                                "comparison_info": {"materials": detected_molecules},
                                "route": "graphrag",
                            },
                            "response": graph_result.get("response", {}),
                            "visualizations": graph_result.get("visualizations", []),
                            "insights": {},
                            "method_used": "graphrag",
                            "graph_info": graph_result.get("graph_info", {}),
                        }
                else:
                    logger.warning("GraphRAG route but no anchor found, falling back to QA")

            # Trigger Tier 2 predictions early if LLM says this query needs them
            if needs_prediction and len(detected_molecules) >= 2:
                try:
                    logger.info(f"LLM requested predictions — fetching now for {detected_molecules}")
                    self._predicted_candidates = self.prediction_integrator.get_predictions(
                        detected_molecules[0], detected_molecules[1], set(), top_k=10
                    )
                    logger.info(f"Predictions ready: {len(self._predicted_candidates)} candidates")
                except Exception as e:
                    logger.warning(f"Early prediction fetch failed: {e}")
                    self._predicted_candidates = []

            filter_info = self.intelligent_filter.identify_filter_columns(
                query,
                self.current_data.columns,
                sample_data
            )
            logger.info(f"✓ Recognition result: {filter_info}")

            # 4. Apply intelligent filtering
            if filter_info.get("filter_columns"):
                logger.info("Applying intelligent filtering...")
                filtered_data = self.intelligent_filter.apply_filters(
                    self.current_data,
                    filter_info
                )
            else:
                filtered_data = self.current_data

            # 5. Determine comparison handling — use LLM query understanding
            is_comparison = (len(detected_molecules) >= 2) or is_separation
            comparison_info = parsed_query.get("comparison_info", {})

            if is_comparison:
                comparison_info["is_comparison"] = True
                comparison_info["is_separation"] = is_separation  # From LLM, not hard-coded
                comparison_info["materials"] = detected_molecules
                comparison_info["same_zeolite_required"] = (query_mode == 'ranking')
                comparison_info["query_mode"] = query_mode
                logger.info(f"✓ Comparison query: mode={query_mode}, separation={is_separation}, molecules={detected_molecules}")

                # Apply comparison query logic on already filtered data
                filtered_data = self._handle_comparison_query(
                    comparison_info, query=query, base_data=filtered_data, mapping=mapping
                )
            
            # 6. Sort by relevance
            if len(filtered_data) > 0:
                filtered_data = self._rank_by_relevance(filtered_data, query)
                logger.info(f"Relevance sorting complete")
            
            # 7. Apply unit conversion
            if unit_analysis.get("has_units"):
                filtered_data = self._apply_unit_conversion(filtered_data, unit_analysis)
            
            logger.info(f"Data filtering complete, shape: {filtered_data.shape}")
            
            # 5. [KEY MODIFICATION] Completely skip RAG, use filtered data directly
            # Reason: RAG may contain residual info from original data, causing LLM to analyze wrong materials
            
            # Generate detailed data table
            # For comparison/separation queries, limit to top 100 most important pairs
            if comparison_info.get("is_separation") or comparison_info.get("is_comparison"):
                data_table = self._generate_data_table(filtered_data, max_rows=100)
                logger.info(f"Comparison query: limited to top 100 pairs")
                logger.info(f"✓ Generated data table length: {len(data_table)} chars, {len(data_table.split(chr(10)))} rows")
            else:
                # Normal query: show data with highest relevance
                data_table = self._generate_data_table(filtered_data, max_rows=100)
                logger.info(f"Normal query: showing top 100 relevant rows")
                logger.info(f"✓ Generated data table length: {len(data_table)} chars, {len(data_table.split(chr(10)))} rows")
            
            # Generate text description of data (supplementing RAG's role)
            data_description = self._generate_data_description(filtered_data)
            
            # 6. Build context (using only filtered data)
            material_constraint = ""
            if parsed_query.get("comparison_info") and parsed_query["comparison_info"].get("materials"):
                materials_list = parsed_query["comparison_info"]["materials"]
                material_constraint = f"""
⚠️⚠️⚠️ STRICT LIMITATION ⚠️⚠️⚠️
You MUST:
1. Only use data from the following materials: {', '.join(materials_list)}
2. Strictly forbidden to analyze or mention other materials (such as nitrogen, sodium ion, sodium, mixtures, etc.)
3. If other materials appear in the data table below, ignore them directly
4. Only answer based on the data table below, do not use other information
"""
            
            # Keep context focused: send only filtered table data (no overview paragraph)
            enhanced_context = f"""{data_table}

{material_constraint}"""
            
            # Append Tier 2 predictions to context if fetched
            predicted = getattr(self, '_predicted_candidates', [])
            if predicted:
                pred_text = "\n" + "=" * 80 + "\n"
                pred_text += "PREDICTED CANDIDATES (Tier 2)\n"
                pred_text += "(No direct experimental data — based on similar-guest analogy and modeling)\n"
                pred_text += "=" * 80 + "\n"
                pred_text += "Rank,Zeolite,Modification,Evidence_Type,Diff_Score,Evid_Score,Total,Confidence,Reasoning\n"
                for c in predicted[:5]:
                    pred_text += f"{c.rank},{c.zeolite},{c.modification or 'none'},{c.evidence_type},{c.diffusion_score},{c.evidence_score},{c.total_score},{c.confidence},{c.reasoning[:150].replace(',',';')}\n"
                pred_text += "NOTE: Treat these as model predictions, not measured data.\n"
                enhanced_context += pred_text
                logger.info(f"✓ Appended Tier 2 predictions to context ({len(predicted)} candidates)")

            logger.info(f"✓ Context length passed to LLM: {len(enhanced_context)} chars")
            logger.info(f"✓ Data table contains {len(filtered_data)} rows")
            
            # Check context length, auto-reduce if too large
            MAX_CONTEXT_LENGTH = 60000  # Safe limit of ~120K tokens
            if len(enhanced_context) > MAX_CONTEXT_LENGTH:
                logger.warning(f"Context too long ({len(enhanced_context)} chars), auto-reducing data")
                # Regenerate more compact data table
                reduced_rows = int(len(filtered_data) * MAX_CONTEXT_LENGTH / len(enhanced_context) * 0.8)
                logger.info(f"Reducing to {reduced_rows} rows")
                data_table = self._generate_data_table(filtered_data, max_rows=reduced_rows)
                enhanced_context = f"""{data_table}

{material_constraint}"""
                logger.info(f"✓ Context length after reduction: {len(enhanced_context)} chars")

            # Standard Q&A pipeline (GraphRAG route already returned early above if applicable)
            logger.info("Using standard Q&A pipeline")
            response = self.llm_integration.generate_response(
                query, enhanced_context, "analysis",
                is_comparison=comparison_info.get("is_comparison", False) or comparison_info.get("is_separation", False),
                molecule_a=detected_molecules[0] if len(detected_molecules) > 0 else None,
                molecule_b=detected_molecules[1] if len(detected_molecules) > 1 else None
            )
            graph_info = {}

            # 8. Generate visualizations (temporarily disabled due to SimpleDataFrame compatibility issues)
            # TODO: Fix visualization_engine to fully support SimpleDataFrame
            # visualizations = self._generate_visualizations(
            #     filtered_data, parsed_query, query
            # )
            visualizations = []
            
            # 9. Generate insights
            insights = self.rag_engine.generate_insights(filtered_data, query)
            
            # Convert objects in parsed_query to dictionaries
            serializable_parsed_query = {
                "query": parsed_query.get("query", ""),
                "entities": [e.to_dict() if hasattr(e, 'to_dict') else e for e in parsed_query.get("entities", [])],
                "conditions": [c.to_dict() if hasattr(c, 'to_dict') else c for c in parsed_query.get("conditions", [])],
                "intent": parsed_query.get("intent", ""),
                "visualization_needs": parsed_query.get("visualization_needs", []),
                "units": unit_analysis.get("units", []),  # Use converted units
                "comparison_info": parsed_query.get("comparison_info", {}),
                "confidence": parsed_query.get("confidence", 0.0)
            }
            
            return {
                "success": True,
                "query": query,
                "parsed_query": serializable_parsed_query,
                "unit_analysis": unit_analysis,
                "filtered_data_shape": filtered_data.shape,
                "response": response,
                "visualizations": visualizations,
                "insights": insights,
                "context_used": enhanced_context,  # Return full context, no longer truncated
                "method_used": "qa",
                "graph_info": graph_info
            }
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                "success": False,
                "message": f"Failed to process query: {str(e)}",
                "error": str(e)
            }
    
    def _map_entities_to_columns(self, entities: List) -> Dict[str, str]:
        """Map entities to column names"""
        column_mappings = {}
        
        for entity in entities:
            if hasattr(entity, 'text'):
                entity_text = entity.text
            else:
                entity_text = str(entity)
            
            # Get column mapping suggestions
            mappings = self.synonym_mapper.map_entity_to_columns(
                entity_text, list(self.current_data.columns)
            )
            
            if mappings:
                # Select best matching column
                best_match = mappings[0]
                column_mappings[entity_text] = best_match[0]
        
        return column_mappings
    
    def _generate_visualizations(self, data: pd.DataFrame, 
                                parsed_query: Dict[str, Any], 
                                query: str) -> List[Dict[str, Any]]:
        """Generate visualization charts"""
        visualizations = []
        
        try:
            # Check if there are visualization needs
            viz_needs = parsed_query.get("visualization_needs", [])
            
            if not viz_needs:
                # Auto-generate visualization suggestions
                suggestions = self.visualization_engine.get_visualization_suggestions(data, query)
                viz_needs = [s["type"] for s in suggestions]
            
            # Generate charts
            for viz_type in viz_needs:
                viz_result = self._create_visualization(data, viz_type, query)
                if viz_result and "error" not in viz_result:
                    visualizations.append(viz_result)
            
            # If no type specified, generate default charts
            if not visualizations:
                auto_viz = self.visualization_engine.auto_generate_visualizations(data, query)
                visualizations.extend(auto_viz[:3])  # Limit quantity
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
        
        return visualizations
    
    def _create_visualization(self, data: pd.DataFrame, 
                            viz_type: str, query: str) -> Optional[Dict[str, Any]]:
        """Create a single visualization chart"""
        try:
            # SimpleDataFrame's select_dtypes returns DataFrame, need to get its columns
            # Note: SimpleDataFrame uses string type identifiers
            numeric_df = data.select_dtypes(include=['int64', 'float64'])
            categorical_df = data.select_dtypes(include=['object'])
            
            # columns is already list, no need for tolist()
            numeric_cols = numeric_df.columns if numeric_df.columns else []
            categorical_cols = categorical_df.columns if categorical_df.columns else []
            
            if viz_type == "histogram" and numeric_cols:
                return self.visualization_engine.create_histogram(
                    data, numeric_cols[0], title=f"{numeric_cols[0]} Distribution"
                )
            
            elif viz_type == "scatter" and len(numeric_cols) >= 2:
                return self.visualization_engine.create_scatter_plot(
                    data, numeric_cols[0], numeric_cols[1],
                    title=f"{numeric_cols[0]} vs {numeric_cols[1]} Scatter Plot"
                )
            
            elif viz_type == "bar" and categorical_cols and numeric_cols:
                return self.visualization_engine.create_bar_chart(
                    data, categorical_cols[0], numeric_cols[0],
                    title=f"{numeric_cols[0]} Comparison by {categorical_cols[0]}"
                )
            
            elif viz_type == "pie" and categorical_cols:
                return self.visualization_engine.create_pie_chart(
                    data, categorical_cols[0], title=f"{categorical_cols[0]} Distribution"
                )
            
            elif viz_type == "heatmap" and len(numeric_cols) > 1:
                return self.visualization_engine.create_heatmap(
                    data, numeric_cols[:5], title="Variable Correlation Heatmap"
                )
            
            elif viz_type == "line" and len(numeric_cols) >= 2:
                return self.visualization_engine.create_line_plot(
                    data, numeric_cols[0], numeric_cols[1],
                    title=f"{numeric_cols[1]} Trend"
                )
            
        except Exception as e:
            logger.error(f"Failed to create visualization chart: {e}")
        
        return None
    
    def get_data_preview(self, max_rows: int = 10) -> Dict[str, Any]:
        """Get data preview"""
        if self.current_data is None:
            return {"error": "No data loaded"}
        
        preview_data = self.current_data.head(max_rows)
        
        return {
            "data": preview_data.to_dict('records'),
            "columns": list(self.current_data.columns),
            "dtypes": self.current_data.dtypes,
            "shape": self.current_data.shape
        }
    
    def get_column_info(self) -> Dict[str, Any]:
        """Get column information"""
        if self.current_data is None:
            return {"error": "No data loaded"}
        
        column_info = {}
        
        for col in self.current_data.columns:
            col_data = self.current_data[col]
            null_count = sum(1 for val in col_data if val is None or val == '' or val == 'nan')
            
            # Get data type from dtypes dictionary
            col_dtype = self.current_data.dtypes.get(col, 'object')
            
            info = {
                "dtype": col_dtype,
                "null_count": null_count,
                "null_percentage": (null_count / len(self.current_data)) * 100 if len(self.current_data) > 0 else 0,
                "unique_count": self.current_data.nunique().get(col, 0)
            }
            
            # Check if it's numeric type
            if col_dtype in ['int64', 'float64']:
                # For numeric columns, calculate statistics
                try:
                    numeric_values = [float(v) for v in col_data if v is not None and v != '' and v != 'nan']
                    if numeric_values:
                        info.update({
                            "min": min(numeric_values),
                            "max": max(numeric_values),
                            "mean": sum(numeric_values) / len(numeric_values),
                            "std": (sum((x - sum(numeric_values)/len(numeric_values))**2 for x in numeric_values) / len(numeric_values)) ** 0.5
                        })
                except:
                    pass
            else:
                # For non-numeric columns, get top values
                try:
                    value_counts = self.current_data.value_counts_single(col)
                    info["top_values"] = {item['value']: item['count'] for item in value_counts[:5]}
                except:
                    info["top_values"] = {}
            
            column_info[col] = info
        
        return column_info
    
    def export_analysis_report(self, query: str, output_path: str) -> Dict[str, Any]:
        """Export analysis report"""
        try:
            # Process query
            result = self.process_query(query)
            
            if not result["success"]:
                return {"success": False, "error": result["message"]}
            
            # Generate report
            report = {
                "query": query,
                "timestamp": pd.Timestamp.now().isoformat(),
                "data_summary": self.data_summary,
                "analysis": result["response"],
                "insights": result["insights"],
                "visualizations": result["visualizations"]
            }
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return {"success": True, "output_path": output_path}
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return {"success": False, "error": str(e)}
    
    def add_custom_synonym_mapping(self, key: str, synonyms: List[str]):
        """Add custom synonym mapping"""
        self.synonym_mapper.add_synonym_mapping(key, synonyms)
        logger.info(f"Added synonym mapping: {key} -> {synonyms}")
    
    def _analyze_units_in_data(self, units: List) -> Dict[str, Any]:
        """Analyze unit information in data"""
        if not units:
            return {"has_units": False, "units": [], "conversion_needed": False}
        
        # Convert UnitInfo objects to dictionaries (for JSON serialization)
        units_dict = []
        for unit in units:
            if hasattr(unit, 'to_dict'):
                units_dict.append(unit.to_dict())
            elif isinstance(unit, dict):
                units_dict.append(unit)
            else:
                # If other type, try to convert to string
                units_dict.append(str(unit))
        
        unit_analysis = {
            "has_units": True,
            "units": units_dict,
            "conversion_needed": False,
            "unit_categories": {},
            "recommendations": []
        }
        
        # Group units by category
        for unit in units:
            category = self.semantic_parser.unit_recognizer._find_unit_category(unit.unit if hasattr(unit, 'unit') else str(unit))
            if category:
                if category not in unit_analysis["unit_categories"]:
                    unit_analysis["unit_categories"][category] = []
                # Convert to dictionary
                if hasattr(unit, 'to_dict'):
                    unit_analysis["unit_categories"][category].append(unit.to_dict())
                elif isinstance(unit, dict):
                    unit_analysis["unit_categories"][category].append(unit)
                else:
                    unit_analysis["unit_categories"][category].append(str(unit))
        
        # Check if unit conversion is needed
        for category, category_units in unit_analysis["unit_categories"].items():
            if len(category_units) > 1:
                unit_analysis["conversion_needed"] = True
                unit_analysis["recommendations"].append(
                    f"Found multiple units in {category} category, recommend unifying units"
                )
        
        return unit_analysis
    
    def _apply_unit_conversion(self, data: pd.DataFrame, unit_analysis: Dict[str, Any]) -> pd.DataFrame:
        """Apply unit conversion"""
        if not unit_analysis["conversion_needed"]:
            return data
        
        converted_data = data.copy()
        
        for category, units in unit_analysis["unit_categories"].items():
            if len(units) > 1:
                # Select standard unit for conversion
                standard_unit = units[0].unit
                
                for unit_info in units[1:]:
                    # Find corresponding column
                    for col in converted_data.columns:
                        if self._column_contains_unit(col, unit_info.unit):
                            # Apply unit conversion
                            converted_values = converted_data[col].apply(
                                lambda x: self._convert_value_with_unit(x, unit_info.unit, standard_unit, category)
                            )
                            converted_data[col] = converted_values
        
        return converted_data
    
    def _column_contains_unit(self, column_name: str, unit: str) -> bool:
        """Check if column name contains specific unit"""
        column_lower = column_name.lower()
        unit_lower = unit.lower()
        
        # Simple heuristic rules
        unit_indicators = {
            '°C': ['temp', 'temperature', '温度'],
            '°F': ['temp', 'temperature', '温度'],
            'K': ['temp', 'temperature', '温度'],
            'bar': ['pressure', 'press', '压力'],
            'Pa': ['pressure', 'press', '压力'],
            'M': ['concentration', 'conc', '浓度'],
            'g': ['mass', 'weight', '质量'],
            'kg': ['mass', 'weight', '质量'],
            'L': ['volume', 'vol', '体积'],
            'mL': ['volume', 'vol', '体积']
        }
        
        if unit in unit_indicators:
            indicators = unit_indicators[unit]
            return any(indicator in column_lower for indicator in indicators)
        
        return False
    
    def _convert_value_with_unit(self, value, from_unit: str, to_unit: str, category: str) -> float:
        """Convert value with unit"""
        try:
            # If value is already numeric, convert directly
            if isinstance(value, (int, float)):
                return self.semantic_parser.unit_recognizer.convert_units(
                    value, from_unit, to_unit, category
                )
            
            # If value is string, try to extract numeric value
            if isinstance(value, str):
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', value)
                if match:
                    numeric_value = float(match.group(1))
                    return self.semantic_parser.unit_recognizer.convert_units(
                        numeric_value, from_unit, to_unit, category
                    )
            
            return value
        except:
            return value
    
    def _classify_query_intent(self, query: str, detected_molecules: List[str],
                               has_specific_zeolite: bool) -> str:
        """
        Classify query into 'ranking' or 'comparison' mode.

        Ranking: user wants to know which zeolite is BEST for separation
        Comparison: user wants a direct comparison of two molecules on a specific zeolite/condition
        """
        query_lower = query.lower()

        ranking_keywords = [
            'best', 'strongest', 'which zeolite', 'which material', 'which molecular',
            'recommend', 'top', 'rank', 'highest', 'largest', 'greatest',
            '最好', '推荐', '最强', '哪些', '排名', '哪个', '最优', '最大',
            'what is the best', 'most effective', 'most selective'
        ]

        comparison_keywords = [
            'difference', 'compare', 'contrast', 'versus', 'vs',
            '差多少', '对比', '比较', '相比', '区别',
            'how different', 'how much', 'what is the difference'
        ]

        ranking_score = sum(1 for kw in ranking_keywords if kw in query_lower)
        comparison_score = sum(1 for kw in comparison_keywords if kw in query_lower)

        if ranking_score > comparison_score:
            return 'ranking'
        elif comparison_score > ranking_score:
            return 'comparison'
        elif len(detected_molecules) >= 2 and not has_specific_zeolite:
            return 'ranking'
        else:
            return 'comparison'

    def _has_specific_zeolite_mentioned(self, query: str) -> bool:
        """Check if user is asking about a specific zeolite vs. general recommendation."""
        import re
        query_lower = query.lower()
        zeolite_patterns = [r'mfi', r'fau', r'zsm-?\d', r'cha', r'lta', r'ltl', r'mor',
                           r'beta', r'bea', r'x-type', r'y-type', r'a-type']
        for pat in zeolite_patterns:
            if re.search(pat, query_lower):
                return True
        return False

    def _handle_comparison_query(self, comparison_info: Dict[str, Any], query: str = "", base_data=None, mapping: Dict[str, Any] = None):
        """Intelligently handle comparison and separation queries"""
        from core.simple_dataframe import SimpleDataFrame

        # Use base_data if provided (already pre-filtered), else fall back to current_data
        source_data = base_data if base_data is not None else self.current_data
        # Save reference so _generate_data_table can use correct indices
        self._comparison_source_data = source_data

        logger.info(f"Intelligent comparison query processing: {comparison_info}")

        # Build mapping if not provided (backward-compatible)
        if mapping is None:
            sample_data = {}
            for col in source_data.columns[:10]:
                sample_data[col] = [str(source_data[col][i]) for i in range(min(3, len(source_data)))]

            mapping = self.column_mapper.map_query_to_columns(
                query=query,
                available_columns=list(source_data.columns),
                sample_data=sample_data
            )

        logger.info(f"Column mapping: material_col={mapping.get('material_column')}, zeolite_col={mapping.get('zeolite_column')}")

        # Get material info from mapping result
        materials = mapping.get("detected_materials", comparison_info.get("materials", []))
        material_keywords = mapping.get("material_keywords", [])
        material_col = mapping.get("material_column")
        zeolite_col = mapping.get("zeolite_column")

        # Use task_columns from 3-layer mapper for role-based column lookups
        task_cols = mapping.get("task_columns", {})

        same_zeolite = comparison_info.get("same_zeolite_required", False)

        if not materials and not material_keywords:
            logger.warning("No material name detected, returning all data")
            return source_data

        # Use LLM mapped column name
        if not material_col:
            logger.warning("LLM failed to map material column, returning all data")
            return source_data

        # 2. Filter rows containing any specified material
        filtered_rows = []

        # Use LLM provided keywords; if none, generate from molecule names
        if material_keywords:
            search_keywords = material_keywords
        else:
            search_keywords = []
            for material in materials:
                variants = [
                    material.lower(),
                    material.lower().replace(' ', '_'),
                    material.lower().replace(' ', ''),
                ]
                search_keywords.extend(variants)
        
        logger.info(f"Searching with keywords: {search_keywords}")
        
        for idx in range(len(source_data)):
            row_material = str(source_data[material_col][idx]).lower().strip()
            
            # More precise matching:
            # 1. Remove content in parentheses (e.g. (adsorbed))
            import re
            clean_material = re.sub(r'\s*\([^)]*\)', '', row_material).strip()
            
            # 2. Check if exact match or keyword
            matched = False
            for keyword in search_keywords:
                keyword_lower = keyword.lower().strip()
                if clean_material == keyword_lower or re.search(r'\b' + re.escape(keyword_lower) + r'\b', clean_material):
                    matched = True
                    break

            if matched:
                filtered_rows.append(idx)
        
        if not filtered_rows:
            # Smart fallback: keyword matching failed — try mapping against actual data values
            unique_materials = set()
            for idx in range(len(source_data)):
                val = str(source_data[material_col][idx]).lower().strip()
                val = re.sub(r'\s*\([^)]*\)', '', val).strip()
                if val:
                    unique_materials.add(val)
            unique_list = sorted(unique_materials)[:200]  # Limit for LLM context

            logger.warning(f"No data found for materials {materials} with keywords {search_keywords}")
            logger.info(f"Smart fallback: matching against {len(unique_list)} unique data values via LLM")

            try:
                from core.llm_integration import LLMIntegration
                fallback_llm = LLMIntegration()
                mapping_prompt = f"""Map these query terms to the EXACT matching guest molecule names from the database list below.
Query terms: {json.dumps(materials)}
Available database values: {json.dumps(unique_list)}
Return ONLY valid JSON: {{"mapped": ["exact_value_from_list", ...]}}
If a term has no match, omit it. Match chemical formulas to full names (CO2→carbon dioxide, CH4→methane, etc)."""
                fb_resp = fallback_llm.generate_response(mapping_prompt, "", "general")
                fb_text = fb_resp.get("answer", "{}")
                # Extract JSON from response
                json_match = re.search(r'\{[^}]*"mapped"[^}]*\}', fb_text, re.DOTALL)
                if json_match:
                    mapped = json.loads(json_match.group(0)).get("mapped", [])
                    logger.info(f"Smart fallback mapped: {materials} → {mapped}")
                    # Retry search with mapped values
                    if mapped:
                        materials = mapped
                        search_keywords = []
                        for m in mapped:
                            m_lower = m.lower().strip()
                            search_keywords.extend([m_lower, m_lower.replace(' ', '_'), m_lower.replace(' ', '')])
                        # Re-run the search loop
                        for idx in range(len(source_data)):
                            row_material = str(source_data[material_col][idx]).lower().strip()
                            clean_material = re.sub(r'\s*\([^)]*\)', '', row_material).strip()
                            for keyword in search_keywords:
                                if clean_material == keyword or re.search(r'\b' + re.escape(keyword) + r'\b', clean_material):
                                    filtered_rows.append(idx)
                                    break
            except Exception as e:
                logger.warning(f"Smart fallback failed: {e}")

        if not filtered_rows:
            logger.warning(f"No data found for materials {materials} after smart fallback")
            return source_data
        
        # [Validation] Print first few rows of filtered data to confirm correctness
        logger.info(f"✓ Filtered {len(filtered_rows)} rows")
        logger.info(f"✓ First 5 material examples:")
        for i, idx in enumerate(filtered_rows[:5]):
            mat_val = str(source_data[material_col][idx])
            logger.info(f"   Row {idx}: {mat_val}")
        if len(filtered_rows) > 5:
            logger.info(f"   ... {len(filtered_rows) - 5} more rows")
        
        # 3. If same zeolite required, group data
        if same_zeolite:
            # Use LLM mapped zeolite column, if none try to find
            if not zeolite_col:
                for col in source_data.columns:
                    if 'zeolite' in col.lower():
                        zeolite_col = col
                        break
            
            if zeolite_col:
                # Group by zeolite, find zeolites containing all specified materials
                zeolite_groups = {}
                for idx in filtered_rows:
                    zeolite_name = str(source_data[zeolite_col][idx])
                    if zeolite_name not in zeolite_groups:
                        zeolite_groups[zeolite_name] = []
                    zeolite_groups[zeolite_name].append(idx)
                
                # Find zeolites containing all materials
                valid_rows = []
                for zeolite_name, rows in zeolite_groups.items():
                    # Check if this zeolite contains all specified materials
                    found_materials = set()
                    for idx in rows:
                        row_material = str(source_data[material_col][idx]).lower().strip()
                        # Remove parentheses content
                        import re
                        clean_material = re.sub(r'\s*\([^)]*\)', '', row_material).strip()
                        
                        for material in materials:
                            material_lower = material.lower().strip()
                            variants = [material_lower, material_lower.replace(' ', '_'), material_lower.replace(' ', '')]
                            # Use exact matching
                            for variant in variants:
                                if clean_material == variant or re.search(r'\b' + re.escape(variant) + r'\b', clean_material):
                                    found_materials.add(material)
                                    break
                    
                    # If all materials found, add to result
                    if len(found_materials) >= len(materials):
                        valid_rows.extend(rows)
                        logger.info(f"Zeolite {zeolite_name} contains all materials: {found_materials}")
                
                filtered_rows = valid_rows if valid_rows else filtered_rows
        
        # 4. Temperature filtering: always apply for comparison/separation queries to find similar-temperature pairs
        temperature_col = mapping.get("temperature_column")
        if not temperature_col:
            for col in source_data.columns:
                if 'temperature' in col.lower() or 'temp' in col.lower():
                    temperature_col = col
                    break
        
        if temperature_col:
            filtered_rows = self._filter_similar_temperature(
                filtered_rows, temperature_col, material_col, zeolite_col, materials,
                source_data=source_data
            )
            logger.info(f"{len(filtered_rows)} rows remaining after temperature pairing")
        else:
            logger.warning("No temperature column found; skipping similar-temperature pairing")
        
        # 5. Intelligent reduction: if data volume too large, keep representative data per zeolite group
        if len(filtered_rows) > 100:
            filtered_rows = self._smart_reduce_data(
                filtered_rows, material_col, zeolite_col, temperature_col,
                source_data=source_data
            )
            logger.info(f"{len(filtered_rows)} rows remaining after intelligent reduction")
        
        # 6. Create filtered DataFrame
        if not filtered_rows:
            logger.warning("No data satisfies conditions")
            return source_data
        
        # 6.5 Sort by pair difference if pairs exist - prioritize rows in top pairs
        if hasattr(self, '_paired_data') and self._paired_data:
            # Get all row indices from top pairs (sorted by relative_diff)
            pair_rows = set()
            for pair in self._paired_data[:50]:  # Top 50 pairs
                pair_rows.add(pair['idx1'])
                pair_rows.add(pair['idx2'])
            
            # Sort filtered_rows: rows in pairs come first, maintain pair order
            def sort_key(idx):
                if idx in pair_rows:
                    # Find the highest ranking pair this row belongs to
                    for rank, pair in enumerate(self._paired_data[:50]):
                        if idx == pair['idx1'] or idx == pair['idx2']:
                            return (0, rank)  # (priority, pair_rank)
                    return (0, 999)  # Should not reach here
                else:
                    return (1, 0)  # Non-pair rows come after
            
            filtered_rows = sorted(filtered_rows, key=sort_key)
            logger.info(f"Sorted rows by pair difference, top {len(pair_rows)} rows from pairs prioritized")
        
        filtered_data = {}
        for col in source_data.columns:
            filtered_data[col] = [source_data[col][idx] for idx in filtered_rows]
        
        result = SimpleDataFrame(filtered_data)
        logger.info(f"Final filtered data shape: {result.shape}")

        # 7. Get predicted candidates for zeolites without direct evidence (Tier 2)
        self._predicted_candidates = []
        if len(materials) >= 2:
            try:
                # Collect zeolites that already have direct pairings
                direct_zeolites = set()
                paired = getattr(self, '_paired_data', None)
                if paired:
                    for pair in paired:
                        zeo = pair.get('zeolite', '')
                        if zeo:
                            direct_zeolites.add(zeo.lower().strip())

                mol_a = materials[0]
                mol_b = materials[1]
                logger.info(f"Fetching predicted candidates for {mol_a}/{mol_b} "
                            f"(excluding {len(direct_zeolites)} direct-evidence zeolites)")

                self._predicted_candidates = self.prediction_integrator.get_predictions(
                    mol_a, mol_b, direct_zeolites, top_k=10
                )
                logger.info(f"Got {len(self._predicted_candidates)} predicted candidates")
            except Exception as e:
                logger.warning(f"Prediction integration failed: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                self._predicted_candidates = []
        else:
            logger.info(f"Skipping predictions: only {len(materials)} materials detected")

        return result
    
    def _filter_similar_temperature(self, rows: List[int], temperature_col: str, 
                                   material_col: str, zeolite_col: str,
                                   materials: List[str], source_data=None) -> List[int]:
        """
        Filter data with similar temperatures: for the same zeolite, find pairs with
        similar temperatures between different materials, sorted by diffusion coefficient
        difference (larger difference = better separation effect)
        """
        if not rows or not zeolite_col or not material_col:
            return rows
        
        # Use provided source_data or fall back to current_data
        data = source_data if source_data is not None else self.current_data
        
        import re
        
        # Find diffusion coefficient column and unit column
        value_col = None
        unit_col = None
        for col in data.columns:
            if 'converted_value' in col.lower() or col.lower() == 'value':
                value_col = col
                if 'converted' in col.lower():
                    break
        for col in data.columns:
            if 'converted_unit' in col.lower() or col.lower() == 'unit':
                unit_col = col
                break
        
        if not value_col:
            logger.warning("Diffusion coefficient column not found, cannot calculate difference")
            return rows
        
        logger.info(f"Using diffusion coefficient column: {value_col}, unit column: {unit_col}")
        
        # Group by zeolite
        zeolite_groups = {}
        for idx in rows:
            zeolite_name = str(data[zeolite_col][idx])
            if zeolite_name not in zeolite_groups:
                zeolite_groups[zeolite_name] = []
            zeolite_groups[zeolite_name].append(idx)
        
        logger.info(f"Found {len(zeolite_groups)} zeolite groups")
        
        # Store pair info: (diff, idx1, idx2, zeolite, mat1, mat2, temp1, temp2, val1, val2)
        pairs = []
        seen_pairs = set()  # For deduplication: prevent duplicate pairs
        tolerance = 20  # Temperature tolerance ±20K
        
        for zeolite_name, group_rows in zeolite_groups.items():
            # Group data for this zeolite by material
            material_data = {}
            seen_material_entries = {}  # For deduplication: prevent same data being added multiple times
            for idx in group_rows:
                row_material = str(data[material_col][idx]).lower()
                temp_str = str(data[temperature_col][idx])
                match = re.search(r'(\d+(?:\.\d+)?)', temp_str)
                if not match:
                    # NaN / unknown temperature: still include data, mark as NaN
                    # These can pair with other NaN-temperature data (both unknown)
                    temp_value = float('nan')
                else:
                    temp_value = float(match.group(1))
                
                # Get diffusion coefficient
                value_str = str(data[value_col][idx])
                try:
                    # Handle scientific notation
                    value_num = float(value_str)
                except:
                    continue
                
                # Filter by unit: only use m2/s compatible units, skip s^-1 and other incompatible units
                if unit_col:
                    unit_str = str(data[unit_col][idx]).lower().strip()
                    # Normalize unicode superscripts to ASCII (e.g. m²/s -> m2/s)
                    unit_str_ascii = unit_str.encode('ascii', errors='ignore').decode('ascii')
                    # Skip jump diffusion (1/s, s^-1) and dimensionless units
                    if any(bad in unit_str_ascii for bad in ['1/s', 's-1', 's?1', 'none', 'dimensionless']):
                        continue
                    # Only keep diffusion coefficient units (m2/s variants)
                    if not any(good in unit_str_ascii for good in ['m2', 'm^2', 'm2/s', 'msd']):
                        # Also accept if original string contains /s and starts with m (catches m²/s unicode variants)
                        if not (('/s' in unit_str or 's-1' not in unit_str) and unit_str.startswith('m')):
                            continue
                
                # Filter out physically unreasonable values (diffusion coeff typically 1e-20 to 1e-4 m2/s)
                if value_num <= 0 or value_num > 1e-3 or value_num < 1e-22:
                    continue
                
                # Identify material type (using exact matching)
                clean_material = re.sub(r'\s*\([^)]*\)', '', row_material).strip()
                
                for material in materials:
                    # Dynamic variant generation (no hardcoded molecule lists)
                    material_lower = material.lower().strip()
                    variants = [
                        material_lower,
                        material_lower.replace(' ', '_'),
                        material_lower.replace(' ', ''),
                    ]
                    matched = False
                    for variant in variants:
                        if clean_material == variant or re.search(r'\b' + re.escape(variant) + r'\b', clean_material):
                            matched = True
                            break
                    
                    if matched:
                        if material not in material_data:
                            material_data[material] = []
                        
                        # Create unique key to prevent duplicates (zeolite+material+temp+value)
                        entry_key = (zeolite_name, material, round(temp_value, 2), round(value_num, 25))
                        
                        if entry_key not in seen_material_entries:
                            seen_material_entries[entry_key] = True
                            material_data[material].append((idx, temp_value, value_num))
                        
                        break
            
            # If this zeolite has data for multiple materials, do temperature pairing
            if len(material_data) >= 2:
                material_list = list(material_data.keys())
                for i in range(len(material_list)):
                    for j in range(i + 1, len(material_list)):
                        mat1, mat2 = material_list[i], material_list[j]
                        # Find pairs with similar temperatures
                        for idx1, temp1, val1 in material_data[mat1]:
                            for idx2, temp2, val2 in material_data[mat2]:
                                # Temperature pairing: match if both are NaN (unknown) AND same zeolite/source,
                                # or if |temp1 - temp2| <= tolerance
                                import math
                                both_nan = (math.isnan(temp1) and math.isnan(temp2))
                                if both_nan or abs(temp1 - temp2) <= tolerance:
                                    # Create pair unique key (prevent duplicate pairs)
                                    pair_key = (
                                        zeolite_name, 
                                        mat1, mat2,
                                        round(temp1, 2), round(temp2, 2),
                                        round(val1, 25), round(val2, 25)
                                    )
                                    
                                    # If this pair already exists, skip
                                    if pair_key in seen_pairs:
                                        continue
                                    
                                    seen_pairs.add(pair_key)
                                    
                                    # Calculate diffusion coefficient difference (absolute value)
                                    diff = abs(val1 - val2)
                                    # Calculate ratio difference (large/small value ratio, for more accurate comparison)
                                    # Larger ratio = better separation effect
                                    max_val = max(val1, val2)
                                    min_val = min(val1, val2)
                                    if min_val > 0:
                                        ratio = max_val / min_val  # Ratio > 1
                                        import math
                                        relative_diff = math.log10(ratio)  # Log10 scale: higher = better separation
                                    else:
                                        relative_diff = float('inf')  # If min value is 0, difference is infinity
                                    
                                    pairs.append({
                                        'diff': diff,
                                        'relative_diff': relative_diff,
                                        'idx1': idx1,
                                        'idx2': idx2,
                                        'zeolite': zeolite_name,
                                        'mat1': mat1,
                                        'mat2': mat2,
                                        'temp1': temp1,
                                        'temp2': temp2,
                                        'val1': val1,
                                        'val2': val2
                                    })
        
        if not pairs:
            logger.warning("No pairs with similar temperatures found, returning all filtered data")
            return rows
        
        # Sort by relative difference (descending), pairs with larger difference come first
        pairs.sort(key=lambda x: x['relative_diff'], reverse=True)
        
        logger.info(f"Found {len(pairs)} pairs, sorted by diffusion coefficient difference")
        
        # Output top 5 pairs with largest difference
        for i, pair in enumerate(pairs[:5]):
            logger.info(f"Top{i+1}: {pair['zeolite']} - {pair['mat1']}@{pair['temp1']}K({pair['val1']:.2e}) vs "
                       f"{pair['mat2']}@{pair['temp2']}K({pair['val2']:.2e}), relative_diff={pair['relative_diff']:.2%}")
        
        # Save pair info to instance variable for later table generation
        self._paired_data = pairs
        
        # Note: return all original filtered rows here, not just rows in pairs
        # This ensures LLM can see all relevant data, not just a few paired rows
        # Pair info is passed to _generate_data_table via self._paired_data
        logger.info(f"Found {len(pairs)} pairs, keeping original {len(rows)} rows for analysis")
        return rows
    
    def _rank_by_relevance(self, df, query: str) -> 'SimpleDataFrame':
        """Sort data by relevance to query
        
        Args:
            df: Filtered data
            query: User query
            
        Returns:
            Data sorted by relevance
        """
        if len(df) == 0:
            return df
        
        try:
            # 为每行数据生成文本描述
            row_texts = []
            for i in range(len(df)):
                row_text_parts = []
                for col in df.columns:
                    value = str(df[col][i])
                    if value and value != 'nan' and value != '':
                        row_text_parts.append(f"{col}: {value}")
                row_texts.append(" | ".join(row_text_parts))
            
            # Use RAG engine to calculate similarity
            from core.rag_engine import HybridRAGEngine
            
            # Calculate relevance score for each row
            scores = []
            for row_text in row_texts:
                # Use simple keyword matching to calculate relevance
                score = 0
                query_lower = query.lower()
                row_text_lower = row_text.lower()
                
                # Query word exact match
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 2:  # Ignore short words
                        if word in row_text_lower:
                            score += 10
                
                # Query overall similarity (simple version)
                common_chars = sum(1 for c in query_lower if c in row_text_lower)
                score += common_chars / len(query_lower) if len(query_lower) > 0 else 0
                
                scores.append(score)
            
            # Sort by score (descending)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            # Reorganize data
            sorted_data = {}
            for col in df.columns:
                sorted_data[col] = [df[col][i] for i in sorted_indices]
            
            from core.simple_dataframe import SimpleDataFrame
            result = SimpleDataFrame(sorted_data)
            
            logger.info(f"Relevance sorting: Top 5 scores = {[scores[i] for i in sorted_indices[:5]]}")
            return result
            
        except Exception as e:
            logger.warning(f"Relevance sorting failed: {e}, returning original data")
            return df
    
    def _smart_reduce_data(self, rows: List[int], material_col: str, 
                          zeolite_col: str, temperature_col: str, source_data=None) -> List[int]:
        """Intelligently reduce data: group by zeolite, keep representative data per group"""
        if not rows or not zeolite_col:
            return rows[:100]  # If no zeolite column, just take first 100
        
        # Use provided source_data or fall back to current_data
        data = source_data if source_data is not None else self.current_data
        
        # Group by zeolite
        zeolite_groups = {}
        for idx in rows:
            zeolite_name = str(data[zeolite_col][idx])
            if zeolite_name not in zeolite_groups:
                zeolite_groups[zeolite_name] = []
            zeolite_groups[zeolite_name].append(idx)
        
        # Keep representative data per zeolite group
        selected_rows = []
        max_per_group = max(10, 100 // len(zeolite_groups))  # At least 10, total not exceeding 100
        
        for zeolite_name, group_rows in zeolite_groups.items():
            # Within each zeolite group, group by material
            material_groups = {}
            for idx in group_rows:
                material_name = str(data[material_col][idx]).lower()
                if material_name not in material_groups:
                    material_groups[material_name] = []
                material_groups[material_name].append(idx)
            
            # Keep a few representative samples per material
            samples_per_material = max(3, max_per_group // len(material_groups))
            
            for material_name, material_rows in material_groups.items():
                # If temperature column exists, select samples with different temperatures
                if temperature_col:
                    # Sort by temperature, select dispersed samples
                    temp_sorted = []
                    for idx in material_rows:
                        temp_str = str(data[temperature_col][idx])
                        import re
                        match = re.search(r'(\d+(?:\.\d+)?)', temp_str)
                        temp = float(match.group(1)) if match else 0
                        temp_sorted.append((idx, temp))
                    
                    temp_sorted.sort(key=lambda x: x[1])
                    
                    # Uniform sampling
                    step = max(1, len(temp_sorted) // samples_per_material)
                    selected = [temp_sorted[i][0] for i in range(0, len(temp_sorted), step)]
                    selected_rows.extend(selected[:samples_per_material])
                else:
                    # No temperature column, just take first few
                    selected_rows.extend(material_rows[:samples_per_material])
        
        logger.info(f"Smart reduction: {len(zeolite_groups)} zeolites, "
                   f"reduced from {len(rows)} to {len(selected_rows)} rows")
        
        return selected_rows[:100]  # Ensure not exceeding 100 rows
    
    def _generate_data_description(self, df) -> str:
        """Generate text description of data, supplementing RAG's role"""
        if df is None or len(df) == 0:
            return "No data"
        
        lines = []
        lines.append("=" * 60)
        lines.append("Filtered Data Overview")
        lines.append("=" * 60)
        
        # Data volume
        lines.append(f"\nTotal rows: {len(df)}")
        
        # Material statistics
        if 'material' in [c.lower() for c in df.columns]:
            material_col = [c for c in df.columns if 'material' in c.lower()][0]
            materials = {}
            for i in range(len(df)):
                mat = str(df[material_col][i])
                materials[mat] = materials.get(mat, 0) + 1
            lines.append(f"\nMaterials included:")
            for mat, count in materials.items():
                lines.append(f"  - {mat}: {count} records")
        
        # Zeolite statistics
        if 'zeolite' in [c.lower() for c in df.columns]:
            zeolite_col = [c for c in df.columns if 'zeolite' in c.lower()][0]
            zeolites = {}
            for i in range(len(df)):
                zeo = str(df[zeolite_col][i])
                zeolites[zeo] = zeolites.get(zeo, 0) + 1
            lines.append(f"\nZeolites included (top 10):")
            for i, (zeo, count) in enumerate(list(zeolites.items())[:10]):
                lines.append(f"  - {zeo}: {count} records")
            if len(zeolites) > 10:
                lines.append(f"  ... {len(zeolites) - 10} more zeolites")
        
        # Temperature range
        if 'temperature' in [c.lower() for c in df.columns]:
            temp_col = [c for c in df.columns if 'temperature' in c.lower()][0]
            import re
            temps = []
            for i in range(len(df)):
                temp_str = str(df[temp_col][i])
                match = re.search(r'(\d+(?:\.\d+)?)', temp_str)
                if match:
                    temps.append(float(match.group(1)))
            if temps:
                lines.append(f"\nTemperature range: {min(temps)}K - {max(temps)}K")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
    
    def _generate_data_table(self, df, max_rows: int = None) -> str:
        """Generate compact format data table (CSV format, not Markdown)
        
        For comparison/separation queries with paired data, this is the PRIMARY analysis table.
        The paired data table should be the main focus, not the original data.
        
        Args:
            df: Data frame
            max_rows: Maximum rows to display, None means display all
        """
        if df is None or len(df) == 0:
            return "No data"
        
        lines = []
        
        # If there is paired data, show comparison table as PRIMARY analysis
        if hasattr(self, '_paired_data') and self._paired_data:
            # Detect the two molecules from the first pair
            first_pair = self._paired_data[0]
            mol1 = first_pair.get('mat1', 'Molecule_A')
            mol2 = first_pair.get('mat2', 'Molecule_B')
            mol1_title = mol1.replace('_', ' ').title()
            mol2_title = mol2.replace('_', ' ').title()

            lines.append("=" * 80)
            lines.append(f"PAIRED DATA ANALYSIS - {mol1_title} vs {mol2_title} Separation Performance")
            lines.append("=" * 80)
            lines.append("")

            # Step 1: Zeolite ranking by best separation performance
            zeolite_stats = {}
            for pair in self._paired_data:
                zeolite = pair['zeolite']
                if zeolite not in zeolite_stats:
                    zeolite_stats[zeolite] = {
                        'count': 0,
                        'max_diff': 0,
                        'avg_diff': 0,
                        'best_pair': None,
                        'all_diffs': []
                    }
                zeolite_stats[zeolite]['count'] += 1
                zeolite_stats[zeolite]['all_diffs'].append(pair['relative_diff'])
                if pair['relative_diff'] > zeolite_stats[zeolite]['max_diff']:
                    zeolite_stats[zeolite]['max_diff'] = pair['relative_diff']
                    zeolite_stats[zeolite]['best_pair'] = pair
            
            # Calculate average difference for each zeolite
            for zeolite, stats in zeolite_stats.items():
                stats['avg_diff'] = sum(stats['all_diffs']) / len(stats['all_diffs']) if stats['all_diffs'] else 0
            
            # Sort by best difference descending
            sorted_zeolites = sorted(zeolite_stats.items(), 
                                    key=lambda x: x[1]['max_diff'], 
                                    reverse=True)
            
            # Generate zeolite ranking section
            lines.append("ZEOLITE RANKING BY SEPARATION PERFORMANCE")
            lines.append(f"(Ranked by maximum Log10(D_max/D_min) between {mol1} and {mol2} -- higher = better separation)")
            lines.append("")
            
            for rank, (zeolite, stats) in enumerate(sorted_zeolites, 1):
                best = stats['best_pair']
                lines.append(f"Rank {rank}: {zeolite}")
                lines.append(f"  - Number of {mol1}/{mol2} pairs at similar temperatures: {stats['count']}")
                lines.append(f"  - Maximum Log10(ratio): {stats['max_diff']:.2f} orders of magnitude")
                lines.append(f"  - Average Log10(ratio): {stats['avg_diff']:.2f} orders of magnitude")
                lines.append(f"  - Best pair: {best['mat1']}({best['val1']:.2e} m2/s @ {best['temp1']:.0f}K) vs "
                           f"{best['mat2']}({best['val2']:.2e} m2/s @ {best['temp2']:.0f}K)")
                lines.append("")
            
            lines.append("=" * 80)
            lines.append("DETAILED PAIRED DATA TABLE")
            lines.append(f"(All {mol1}/{mol2} pairs at similar temperatures, sorted by separation difference)")
            lines.append("=" * 80)
            lines.append("")

            # Step 2: Detailed paired data table with dynamic column headers
            lines.append(f"Rank,Zeolite,{mol1}_DiffCoef(m2/s),{mol1}_Temp(K),{mol1}_DOI,"
                        f"{mol2}_DiffCoef(m2/s),{mol2}_Temp(K),{mol2}_DOI,"
                        f"Log10_ratio,{mol1}_Conc,{mol2}_Conc,Method")
            
            # Paired data also considers limit
            pair_limit = min(len(self._paired_data), max_rows) if max_rows else len(self._paired_data)
            for i, pair in enumerate(self._paired_data[:pair_limit], 1):
                # Get concentration and experimental method from original data
                idx1, idx2 = pair['idx1'], pair['idx2']
                
                # Safely get concentration/DOI and experimental method
                conc1 = 'N/A'
                method1 = 'N/A'
                doi1 = 'N/A'
                conc2 = 'N/A'
                method2 = 'N/A'
                doi2 = 'N/A'
                
                # Always use _comparison_source_data - indices are relative to it
                _src = getattr(self, '_comparison_source_data', self.current_data)
                _src_len = len(_src)
                
                if 'concentration' in _src.columns:
                    conc1 = str(_src['concentration'][idx1]) if idx1 < _src_len else 'N/A'
                    conc2 = str(_src['concentration'][idx2]) if idx2 < _src_len else 'N/A'

                # Keep DOI source for each specific data point
                doi_col = None
                for _c in _src.columns:
                    if _c.lower() == 'doi' or 'doi' in _c.lower():
                        doi_col = _c
                        break
                if doi_col:
                    doi1 = str(_src[doi_col][idx1]) if idx1 < _src_len else 'N/A'
                    doi2 = str(_src[doi_col][idx2]) if idx2 < _src_len else 'N/A'
                
                if 'experimental_method' in _src.columns:
                    method1 = str(_src['experimental_method'][idx1]) if idx1 < _src_len else 'N/A'
                    method2 = str(_src['experimental_method'][idx2]) if idx2 < _src_len else 'N/A'
                
                # Use first method if both available
                method = method1 if method1 != 'N/A' else method2
                if len(method) > 15:
                    method = method[:12] + '...'
                
                # Determine which value belongs to which molecule (generic, not CH4/CO2-specific)
                mat1_is_mol1 = (mol1.lower() in pair['mat1'].lower())
                if mat1_is_mol1:
                    val_a, temp_a, conc_a, doi_a = pair['val1'], pair['temp1'], conc1, doi1
                    val_b, temp_b, conc_b, doi_b = pair['val2'], pair['temp2'], conc2, doi2
                else:
                    val_a, temp_a, conc_a, doi_a = pair['val2'], pair['temp2'], conc2, doi2
                    val_b, temp_b, conc_b, doi_b = pair['val1'], pair['temp1'], conc1, doi1

                log10_val = pair['relative_diff'] if pair['relative_diff'] != float('inf') else 999.0
                lines.append(
                    f"{i},{pair['zeolite']},"
                    f"{val_a:.2e},{temp_a:.0f},{doi_a},"
                    f"{val_b:.2e},{temp_b:.0f},{doi_b},"
                    f"{log10_val:.2f},"
                    f"{conc_a},{conc_b},"
                    f"{method}"
                )
            
            if len(self._paired_data) > pair_limit:
                lines.append(f"# Showing first {pair_limit} pairs (total {len(self._paired_data)} pairs)")
            else:
                lines.append(f"# Total {len(self._paired_data)} pairs")
            
            lines.append("")
            # --- Tier 2: Predicted Candidates (Project0 integration) ---
            # Placed BEFORE interpretation guide to ensure LLM doesn't skip it
            predicted = getattr(self, '_predicted_candidates', [])
            logger.info(f"Tier 2 predictions available: {len(predicted)} candidates")
            if predicted:
                logger.info(f"Adding Tier 2 section to data table: {[c.zeolite for c in predicted[:3]]}")
                lines.append("=" * 80)
                lines.append("PREDICTED CANDIDATES (Tier 2)")
                lines.append("(No direct experimental data for these zeolites — based on similar-guest analogy")
                lines.append(" and molecular property modeling. Treat with appropriate caution.)")
                lines.append("=" * 80)
                lines.append("")
                lines.append("Rank,Zeolite,Modification,Evidence_Type,Diffusion_Score,"
                           "Evidence_Score,Total_Score,Confidence,Preferred_Guest,Reasoning")
                for c in predicted:
                    zeo = c.zeolite.replace(',', ';')
                    mod = c.modification.replace(',', ';') if c.modification else 'none'
                    reasoning = c.reasoning.replace(',', ';')[:200]
                    lines.append(
                        f"{c.rank},{zeo},{mod},{c.evidence_type},"
                        f"{c.diffusion_score},{c.evidence_score},{c.total_score},"
                        f"{c.confidence},{c.preferred_guest},{reasoning}"
                    )
                lines.append("")
                lines.append("NOTE: These candidates lack direct experimental measurements "
                           "for the target molecule pair. They are ranked by a 4-dimensional "
                           "scoring model (evidence quality + diffusion selectivity + "
                           "modification relevance + mechanism plausibility).")
                lines.append("")

            lines.append("=" * 80)
            lines.append("INTERPRETATION GUIDE FOR LLM")
            lines.append("=" * 80)
            lines.append("")
            lines.append(f"1. BEST ZEOLITE: The zeolite ranked #1 above shows the strongest {mol1}/{mol2} separation performance")
            lines.append(f"2. SEPARATION METRIC: 'Log10_ratio' = log10(D_max / D_min) between {mol1} and {mol2} diffusion coefficients")
            lines.append("   - Higher Log10_ratio = better separation")
            lines.append("   - A Log10_ratio >= 2 is considered significant (100x difference)")
            lines.append("3. COMPARISON: Compare Log10_ratio values across zeolites -- differences of several orders of magnitude matter")
            lines.append("4. EVIDENCE: Cite specific diffusion coefficients, temperatures, and DOI sources from the paired table above")
            if predicted:
                lines.append("5. PREDICTIONS: There is a PREDICTED CANDIDATES (Tier 2) table above. ALWAYS mention the top predicted zeolites in your answer as candidates worth investigating.")
            lines.append("")

            # Also provide full filtered rows with ALL columns preserved for LLM context
            lines.append("=" * 80)
            lines.append("FULL FILTERED DATA (ALL COLUMNS PRESERVED)")
            lines.append("(Rows are filtered, but NO columns are removed)")
            lines.append("=" * 80)
            lines.append("")

            full_columns = list(df.columns)
            lines.append(",".join(full_columns))

            full_limit = min(len(df), max_rows) if max_rows else len(df)
            for i in range(full_limit):
                row_data = [str(df[col][i]) for col in full_columns]
                lines.append(",".join(row_data))

            if len(df) > full_limit:
                lines.append(f"# Showing first {full_limit} rows from filtered data (total {len(df)} rows)")
            else:
                lines.append(f"# Total {len(df)} filtered rows (all columns preserved)")
            lines.append("")
            
            return "\n".join(lines)
        
        # If NO paired data, show original data (fallback for non-comparison queries)
        lines.append(f"# Filtered Data ({len(df)} rows)")
        
        # Use all columns
        key_columns = list(df.columns)
        
        # CSV header
        lines.append(",".join(key_columns))
        
        # Display data rows (CSV format)
        display_rows = len(df) if max_rows is None else min(max_rows, len(df))
        for i in range(display_rows):
            row_data = [str(df[col][i]) for col in key_columns]
            lines.append(",".join(row_data))
        
        if display_rows < len(df):
            lines.append(f"# ... omitting {len(df) - display_rows} rows")
        
        return "\n".join(lines)
    
    def _process_visualization_query(self, query: str) -> Dict[str, Any]:
        """Process visualization request"""
        logger.info("Processing visualization request")
        
        try:
            # Get column info
            column_info = self.get_column_info()
            
            # Generate plot code
            code_result = self.code_generator.generate_plot_code(
                query, self.current_data, column_info
            )
            
            if not code_result["success"]:
                return {
                    "success": False,
                    "message": f"Failed to generate code: {code_result.get('error', 'Unknown error')}",
                    "error": code_result.get("error")
                }
            
            # Execute code
            execute_result = self.code_generator.execute_plot_code(
                code_result["code"], self.current_data
            )
            
            if not execute_result["success"]:
                return {
                    "success": False,
                    "message": f"Failed to execute code: {execute_result.get('error', 'Unknown error')}",
                    "error": execute_result.get("error"),
                    "code": code_result["code"]  # Return code for debugging
                }
            
            # Return success result
            return {
                "success": True,
                "query": query,
                "response": {
                    "answer": f"Chart generated as requested: {query}",
                    "model": self.llm_integration.model,
                    "tokens_used": 0,
                    "response_type": "visualization"
                },
                "visualization": {
                    "image": execute_result["image"],
                    "format": execute_result["format"],
                    "title": query
                },
                "code": code_result["code"]  # Optional: return generated code
            }
            
        except Exception as e:
            logger.error(f"Failed to process visualization request: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Failed to process visualization request: {str(e)}",
                "error": str(e)
            }
    
    # set_method removed — auto-routing eliminates need for manual toggle

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        pg_info = {}
        if hasattr(self, 'project_graph_builder') and self.project_graph_builder:
            pg_info = {
                "ready": self.project_graph_builder._built,
                "guests": len(self.project_graph_builder._guest_index),
                "zeolites": len(self.project_graph_builder._zeolite_index),
            }
        return {
            "data_loaded": self.current_data is not None,
            "data_shape": self.current_data.shape if self.current_data is not None else None,
            "current_file": self.current_file_path,
            "routing": "auto",
            "components": {
                "semantic_parser": "ready",
                "synonym_mapper": "ready",
                "data_extractor": "ready",
                "rag_engine": "ready",
                "llm_integration": "ready",
                "visualization_engine": "ready",
                "unit_recognizer": "ready",
                "graphrag_engine": "ready",
                "code_generator": "ready",
                "project_graph": pg_info,
            }
        }
