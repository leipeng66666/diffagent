"""
GraphRAG Engine - Knowledge retrieval augmented generation based on graph structure
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import networkx as nx
from collections import defaultdict

from .rag_engine import HybridRAGEngine
from .llm_integration import LLMIntegration


class GraphRAGEngine:
    """GraphRAG Engine - Use graph structure for knowledge retrieval and augmented generation"""
    
    def __init__(self, rag_engine: HybridRAGEngine, llm_integration: LLMIntegration,
                 project_graph_builder=None):
        """Initialize GraphRAG engine

        Args:
            rag_engine: Hybrid RAG engine for semantic retrieval
            llm_integration: LLM integration for response generation
            project_graph_builder: Optional ProjectGraphBuilder for pre-built
                                   project1/project2 knowledge graph
        """
        self.rag_engine = rag_engine
        self.llm_integration = llm_integration
        self.knowledge_graph = nx.DiGraph()  # On-the-fly graph from CSV
        self.entity_relations = defaultdict(list)  # Entity relations

        # Pre-built project graph (lazy load)
        self.project_graph_builder = project_graph_builder
        self._project_graph_ready = False

        logger.info("GraphRAG engine initialization complete")
    
    def build_knowledge_graph(self, df: pd.DataFrame) -> None:
        """Build knowledge graph"""
        logger.info("Starting to build knowledge graph")
        
        # Clear existing graph
        self.knowledge_graph.clear()
        self.entity_relations.clear()
        
        # Identify entities and relations
        entities = self._extract_entities(df)
        relations = self._extract_relations(df, entities)
        
        # Build graph
        for entity in entities:
            self.knowledge_graph.add_node(entity['id'], **entity['attributes'])
        
        for relation in relations:
            source = relation['source']
            target = relation['target']
            rel_type = relation['type']
            weight = relation.get('weight', 1.0)
            
            self.knowledge_graph.add_edge(
                source, target,
                relation_type=rel_type,
                weight=weight,
                **relation.get('attributes', {})
            )
            
            # Record relation
            self.entity_relations[source].append({
                'target': target,
                'type': rel_type,
                'weight': weight
            })
        
        logger.info(f"Knowledge graph build complete: {len(self.knowledge_graph.nodes)} nodes, "
                   f"{len(self.knowledge_graph.edges)} edges")
    
    def _extract_entities(self, df) -> List[Dict[str, Any]]:
        """Extract entities from data"""
        entities = []
        entity_id = 0
        
        # Extract categorical columns as entities (compatible with SimpleDataFrame)
        if hasattr(df, 'select_dtypes'):
            categorical_cols = df.select_dtypes(include=['object']).columns
        else:
            # SimpleDataFrame: Determine based on dtypes
            categorical_cols = [col for col in df.columns 
                              if df.dtypes.get(col, 'object') == 'object']
        
        for col in categorical_cols:
            # Compatible with SimpleDataFrame: Manually get unique values
            if hasattr(df[col], 'unique'):
                unique_values = df[col].unique()
            else:
                # SimpleDataFrame: Manual deduplication
                unique_values = list(set(str(v) for v in df[col] if v is not None and str(v).strip()))
            
            for value in unique_values:
                if pd.notna(value) and str(value).strip():
                    entities.append({
                        'id': f"{col}_{value}_{entity_id}",
                        'type': col,
                        'value': str(value),
                        'attributes': {
                            'column': col,
                            'value': str(value),
                            'frequency': sum(1 for v in df[col] if str(v) == str(value))
                        }
                    })
                    entity_id += 1
        
        # Extract statistical entities for numeric columns (compatible with SimpleDataFrame)
        if hasattr(df, 'select_dtypes'):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        else:
            # SimpleDataFrame: Determine based on dtypes
            numeric_cols = [col for col in df.columns 
                          if df.dtypes.get(col, 'object') in ['int64', 'float64']]
        
        for col in numeric_cols:
            col_data = df[col]
            # Filter non-numeric data
            numeric_values = []
            for val in col_data:
                try:
                    numeric_values.append(float(str(val).replace('∞', 'inf')))
                except (ValueError, TypeError):
                    continue
            
            if len(numeric_values) > 0:
                stats = {
                    'mean': float(np.mean(numeric_values)),
                    'std': float(np.std(numeric_values)),
                    'min': float(np.min(numeric_values)),
                    'max': float(np.max(numeric_values))
                }
                entities.append({
                    'id': f"stats_{col}",
                    'type': 'statistics',
                    'value': col,
                    'attributes': {
                        'column': col,
                        'statistics': stats
                    }
                })
        
        return entities
    
    def _extract_relations(self, df, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations from data (compatible with SimpleDataFrame)"""
        relations = []
        
        # Build relations based on data rows (compatible with SimpleDataFrame)
        if hasattr(df, 'iterrows'):
            rows_iter = df.iterrows()
        else:
            # SimpleDataFrame: Manual iteration
            rows_iter = [(i, {col: df[col][i] for col in df.columns}) 
                        for i in range(len(df))]
        
        for idx, row in rows_iter:
            row_entities = []
            
            # Collect all entities in this row
            for entity in entities:
                if 'column' in entity['attributes']:
                    col = entity['attributes']['column']
                    if col in df.columns:
                        # Compatible with SimpleDataFrame and pandas
                        if hasattr(row, '__getitem__'):
                            row_val = row[col]
                        else:
                            row_val = row.get(col) if isinstance(row, dict) else None
                        
                        if str(row_val) == entity['attributes'].get('value', ''):
                            row_entities.append(entity['id'])
            
            # Build relations between entities in the same row
            for i, source in enumerate(row_entities):
                for target in row_entities[i+1:]:
                    relations.append({
                        'source': source,
                        'target': target,
                        'type': 'co_occurrence',
                        'weight': 1.0,
                        'attributes': {
                            'row_index': idx
                        }
                    })
        
        # Build relations based on numeric correlation (compatible with SimpleDataFrame)
        if hasattr(df, 'select_dtypes'):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = [col for col in df.columns 
                          if df.dtypes.get(col, 'object') in ['int64', 'float64']]
        
        if len(numeric_cols) > 1:
            # Calculate correlation (compatible with SimpleDataFrame)
            if hasattr(df, 'corr'):
                corr_matrix = df[numeric_cols].corr()
            else:
                # Manual correlation calculation
                corr_matrix = {}
                for col1 in numeric_cols:
                    corr_matrix[col1] = {}
                    for col2 in numeric_cols:
                        if col1 == col2:
                            corr_matrix[col1][col2] = 1.0
                        else:
                            # Simple correlation calculation
                            try:
                                vals1 = [float(v) for v in df[col1] if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit())]
                                vals2 = [float(v) for v in df[col2] if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit())]
                                if len(vals1) == len(vals2) and len(vals1) > 1:
                                    corr = np.corrcoef(vals1, vals2)[0, 1]
                                    corr_matrix[col1][col2] = float(corr) if not np.isnan(corr) else 0.0
                                else:
                                    corr_matrix[col1][col2] = 0.0
                            except:
                                corr_matrix[col1][col2] = 0.0
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.3:  # Only keep stronger correlations
                        source_id = f"stats_{col1}"
                        target_id = f"stats_{col2}"
                        relations.append({
                            'source': source_id,
                            'target': target_id,
                            'type': 'correlation',
                            'weight': abs(corr),
                            'attributes': {
                                'correlation': float(corr)
                            }
                        })
        
        return relations
    
    def query_with_graph(self, query: str, df: pd.DataFrame, top_k: int = 10) -> Dict[str, Any]:
        """Query using graph structure"""
        logger.info(f"GraphRAG query: {query}")
        
        # 1. Use LLM to identify key entities in query
        query_entities = self._identify_query_entities(query)
        logger.info(f"Identified query entities: {query_entities}")
        
        # 2. Find relevant nodes in graph
        relevant_nodes = self._find_relevant_nodes(query_entities)
        logger.info(f"Found {len(relevant_nodes)} relevant nodes")
        
        # 3. Use graph traversal to find relevant data
        relevant_data = self._traverse_graph(relevant_nodes, top_k)
        
        # 4. Combine with traditional RAG retrieval
        rag_results = self.rag_engine.search(query, top_k=top_k)
        
        # 5. Merge results
        enhanced_context = self._merge_results(relevant_data, rag_results, query)
        
        return {
            'context': enhanced_context,
            'query_entities': query_entities,
            'relevant_nodes': relevant_nodes,
            'graph_paths': self._find_graph_paths(query_entities)
        }
    
    def _identify_query_entities(self, query: str) -> List[str]:
        """Use LLM to identify entities in query"""
        prompt = f"""
Please extract key entities (such as material names, zeolite names, property names, etc.) from the following query:

Query: {query}

Please return the entity list in JSON format:
{{
    "entities": ["entity1", "entity2", ...]
}}
"""
        try:
            messages = [
                {"role": "system", "content": "You are an entity recognition expert, skilled at extracting key entities from natural language."},
                {"role": "user", "content": prompt}
            ]
            response = self.llm_integration._call_llm(messages, "analysis")
            
            import json
            result = json.loads(response["content"])
            entities = result.get("entities", [])
            
            # Also try to match from graph
            matched_entities = []
            for entity in entities:
                # Find similar nodes in graph
                for node_id in self.knowledge_graph.nodes():
                    node_data = self.knowledge_graph.nodes[node_id]
                    if 'value' in node_data:
                        if entity.lower() in str(node_data['value']).lower() or \
                           str(node_data['value']).lower() in entity.lower():
                            matched_entities.append(node_id)
            
            return matched_entities if matched_entities else entities
            
        except Exception as e:
            logger.warning(f"Entity recognition failed: {e}")
            return []
    
    def _find_relevant_nodes(self, query_entities: List[str]) -> List[str]:
        """Find relevant nodes in graph"""
        relevant_nodes = set()
        
        # Direct match
        for entity in query_entities:
            if isinstance(entity, str) and entity in self.knowledge_graph.nodes():
                relevant_nodes.add(entity)
        
        # Fuzzy match
        for entity in query_entities:
            entity_lower = str(entity).lower()
            for node_id in self.knowledge_graph.nodes():
                node_data = self.knowledge_graph.nodes[node_id]
                if 'value' in node_data:
                    node_value = str(node_data.get('value', '')).lower()
                    if entity_lower in node_value or node_value in entity_lower:
                        relevant_nodes.add(node_id)
        
        return list(relevant_nodes)
    
    def _traverse_graph(self, start_nodes: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
        """Traverse graph to find relevant data"""
        results = []
        visited = set()
        
        for start_node in start_nodes:
            if start_node not in self.knowledge_graph.nodes():
                continue
            
            # BFS traversal
            queue = [(start_node, 0)]  # (node, depth)
            visited.add(start_node)
            
            while queue and len(results) < max_results:
                current_node, depth = queue.pop(0)
                
                if depth > 2:  # Limit depth
                    continue
                
                # Get node data
                node_data = self.knowledge_graph.nodes[current_node]
                if 'row_index' in node_data:
                    results.append({
                        'node_id': current_node,
                        'row_index': node_data['row_index'],
                        'depth': depth,
                        'node_data': node_data
                    })
                
                # Add neighbor nodes
                for neighbor in self.knowledge_graph.neighbors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        
        return results[:max_results]
    
    def _find_graph_paths(self, query_entities: List[str]) -> List[List[str]]:
        """Find paths between entities"""
        paths = []
        
        if len(query_entities) < 2:
            return paths
        
        # Find shortest path between entities
        for i, source in enumerate(query_entities):
            if source not in self.knowledge_graph.nodes():
                continue
            for target in query_entities[i+1:]:
                if target not in self.knowledge_graph.nodes():
                    continue
                
                try:
                    path = nx.shortest_path(
                        self.knowledge_graph,
                        source=source,
                        target=target
                    )
                    if len(path) > 1:
                        paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    def _merge_results(self, graph_data: List[Dict[str, Any]], 
                      rag_results: List[Dict[str, Any]], 
                      query: str) -> str:
        """Merge graph data and RAG results"""
        context_parts = []
        
        # Add graph structure information
        if graph_data:
            context_parts.append("=== Related data based on knowledge graph ===")
            for item in graph_data[:5]:  # Limit quantity
                node_data = item.get('node_data', {})
                context_parts.append(f"Node: {item['node_id']}")
                context_parts.append(f"Attributes: {node_data}")
        
        # Add RAG results
        if rag_results:
            context_parts.append("\n=== Related data based on semantic retrieval ===")
            for result in rag_results[:5]:
                context_parts.append(result.get('content', ''))
        
        return "\n".join(context_parts)
    
    def analyze_with_graph(self, query: str, df: pd.DataFrame, 
                          context: str = None) -> Dict[str, Any]:
        """Use GraphRAG for data analysis"""
        # If graph hasn't been built yet, build it first
        if len(self.knowledge_graph.nodes()) == 0:
            self.build_knowledge_graph(df)
        
        # Use graph for query
        graph_result = self.query_with_graph(query, df)
        
        # Merge context
        if context:
            enhanced_context = f"{context}\n\n{graph_result['context']}"
        else:
            enhanced_context = graph_result['context']
        
        # Use LLM to generate analysis
        response = self.llm_integration.generate_response(
            query, enhanced_context, "analysis"
        )
        
        return {
            'response': response,
            'graph_info': {
                'query_entities': graph_result['query_entities'],
                'relevant_nodes': graph_result['relevant_nodes'],
                'graph_paths': graph_result['graph_paths']
            }
        }

    def analyze_with_project_graph(self, query: str, anchor_type: str,
                                   anchor_value: str) -> Dict[str, Any]:
        """Use pre-built project1/project2 graph for exploratory analysis.

        Args:
            query: Original user query
            anchor_type: "guest" or "zeolite" -- what the subgraph anchors on
            anchor_value: The specific guest or zeolite name

        Returns:
            Dict with 'response' (LLM answer + optional visualizations) and 'graph_info'
        """
        if not self.project_graph_builder:
            return {"error": "Project graph builder not available"}

        # Ensure graph is built
        if not self._project_graph_ready:
            try:
                self.project_graph_builder.build()
                self._project_graph_ready = True
            except Exception as e:
                logger.error(f"Failed to build project graph: {e}")
                return {"error": f"Failed to build project graph: {e}"}

        # Extract subgraph
        subgraph = self.project_graph_builder.get_subgraph(
            anchor_type, anchor_value, max_depth=2
        )

        if subgraph.get("error"):
            return {"error": subgraph["error"]}

        # Build enhanced context for LLM
        context_parts = [subgraph["summary"]]

        # Add available entities context
        entities = self.project_graph_builder.get_available_entities()
        context_parts.append(
            f"\nKnowledge base: {entities['guest_count']} guest molecules, "
            f"{entities['zeolite_count']} zeolite structures, "
            f"{len(entities['topologies'])} topologies."
        )

        # Add table data if available
        for tbl in subgraph.get("tables", [])[:3]:
            context_parts.append(f"\n--- Table: {tbl['name']} ---")
            context_parts.append(tbl["csv_text"][:3000])

        enhanced_context = "\n".join(context_parts)

        # Build system prompt for exploratory graph analysis
        anchor_label = "guest molecule" if anchor_type == "guest" else "zeolite"
        system_prompt = f"""You are a materials science expert analyzing zeolite diffusion data from a knowledge graph.

The data below shows statistical summaries extracted from the knowledge graph anchored on {anchor_label} '{anchor_value}'.

Output format:
1. Overview: Summarize the key diffusion characteristics of this {anchor_label}.
2. Ranking: Rank the top {'zeolites by diffusion speed for this guest' if anchor_type == 'guest' else 'guest molecules by diffusion speed through this zeolite'}. Explain WHY certain ones are faster/slower.
3. Patterns: What patterns do you observe? Consider:
   - Correlation with kinetic diameter (if available) — do larger molecules diffuse slower?
   - Topology effects — which framework types favor/restrict diffusion?
   - Ion exchange effects — how do different cations affect diffusion?
   - Temperature sensitivity — which have high/low activation energy (Ea)?
4. Notable findings: Any outliers, surprising results, or entities that behave differently than expected. Mention unusually large D_range values (multiple orders of magnitude spread).

Rules:
- Cite specific mean_logD values and sample counts (n) from the data
- Use logD scale: higher = faster diffusion (each +1 = 10x faster)
- Mention kinetic diameters and activation energies when available
- Note when statistical significance may be limited by small sample size (n < 5)
- Do not invent data not present in the provided context
- Write in plain English paragraphs and tables — no markdown headers beyond what's specified
- For each key claim, reference the specific data value"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{enhanced_context}\n\nQuestion: {query}"}
        ]

        try:
            response = self.llm_integration._call_llm(messages, "analysis")
            answer = response["content"]

            return {
                "response": {
                    "answer": answer,
                    "model": self.llm_integration.model,
                    "tokens_used": response["total_tokens"],
                    "response_type": "analysis"
                },
                "visualizations": subgraph.get("figures", []),
                "graph_info": {
                    "anchor": subgraph["anchor"],
                    "node_count": len(subgraph["nodes"]),
                    "edge_count": len(subgraph["edges"]),
                    "source": "project_graph",
                    "figures_count": len(subgraph.get("figures", [])),
                    "tables_count": len(subgraph.get("tables", [])),
                }
            }
        except Exception as e:
            logger.error(f"Project graph analysis failed: {e}")
            return {"error": str(e)}

