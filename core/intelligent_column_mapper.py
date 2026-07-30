"""
Intelligent Column Name Mapping Module - 3-Layer Framework
Layer 1: Data type classification (rule-based)
Layer 2: LLM semantic role inference
Layer 3: Role-to-task mapping
"""
from typing import Dict, List, Any, Optional
from loguru import logger
from openai import OpenAI
import json


class IntelligentColumnMapper:
    """Intelligent Column Name Mapper with 3-layer framework"""

    def __init__(self, api_key: str, base_url: str, model: str):
        """Initialize"""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # =========================================================================
    # Layer 1: Data Type Classification (rule-based, no LLM)
    # =========================================================================

    def _classify_column_types(self, columns: List[str],
                                sample_data: Optional[Dict[str, List[Any]]]) -> Dict[str, str]:
        """
        Layer 1: Classify each column by data type.
        Returns: {column_name: 'numeric'|'categorical'|'text'}
        """
        col_types = {}
        for col in columns:
            samples = sample_data.get(col, []) if sample_data else []
            col_types[col] = self._detect_column_dtype(col, samples)
        return col_types

    def _detect_column_dtype(self, col_name: str, samples: List[Any]) -> str:
        """Detect single column's data type from name hints + sample values"""
        col_lower = col_name.lower()

        # Name-based hints
        text_hints = ['doi', 'filename', 'file', 'url', 'link', 'reference']
        categorical_hints = ['name', 'type', 'category', 'method', 'ion', 'variable',
                             'guest', 'molecule', 'adsorbate', 'species', 'compound',
                             'framework', 'zeolite', 'structure', 'unit']
        numeric_hints = ['value', 'coefficient', 'ratio', 'temp', 'press',
                         'conc', 'load', 'si_al', 'silica', 'alumina', 'density',
                         'count', 'number', 'amount', 'weight', 'mass']

        if any(w in col_lower for w in text_hints):
            return 'text'
        if any(w in col_lower for w in numeric_hints):
            return 'numeric'
        if any(w in col_lower for w in categorical_hints):
            return 'categorical'

        # Sample-based detection
        if not samples:
            return 'categorical'

        numeric_count = 0
        valid_samples = []
        for s in samples:
            if s is None:
                continue
            try:
                float(str(s).replace('∞', 'inf'))
                numeric_count += 1
                valid_samples.append(s)
            except (ValueError, TypeError):
                valid_samples.append(s)

        if not valid_samples:
            return 'text'

        if numeric_count / len(valid_samples) >= 0.6:
            return 'numeric'
        elif len(set(str(s)[:30] for s in valid_samples)) <= len(valid_samples) * 0.5:
            return 'categorical'
        else:
            return 'text'

    # =========================================================================
    # Layer 2: LLM Semantic Role Inference
    # =========================================================================

    def _semantic_role_inference(self, query: str, columns: List[str],
                                  col_types: Dict[str, str],
                                  sample_data: Optional[Dict[str, List[Any]]]) -> Dict[str, str]:
        """
        Layer 2: Use LLM to infer business role for each column.
        Returns: {actual_column_name: business_role_name}
        """
        # Build column info with type hints
        column_info = ""
        for col in columns:
            dtype = col_types.get(col, 'unknown')
            column_info += f"- {col} (type: {dtype})"
            if sample_data and col in sample_data:
                samples = [str(s) for s in sample_data[col][:3] if s is not None]
                if samples:
                    column_info += f" [samples: {', '.join(samples)}]"
            column_info += "\n"

        prompt = f"""
You are a data schema analyst. Given a CSV with the following columns and sample values,
identify the business role of each column.

User query (for context): {query}

Column information:
{column_info}

Map each column to ONE of these business roles (use null if none match):
- guest_molecule: the guest molecule/adsorbate name (e.g., methane, CO2, propane)
- guest_composition: the molecular formula or composition (e.g., C2H6, H2O)
- zeolite_name: the zeolite/framework name as reported in original paper
- std_zeolite_name: standardized zeolite name (e.g., MFI, FAU-NaY)
- si_al_ratio: Si/Al ratio of the zeolite framework
- modified_ion: modifying ion or metal (e.g., Pt, Na, H)
- loading_value: metal loading or guest molecule loading amount (numeric)
- loading_unit: unit of loading (e.g., wt%, mol/kg)
- diffusion_coefficient_value: diffusion coefficient numeric value (m2/s or cm2/s)
- diffusion_coefficient_unit: unit of diffusion coefficient
- temperature_value: temperature numeric value (typically in K)
- temperature_unit: unit of temperature
- concentration_value: concentration numeric value
- concentration_unit: unit of concentration
- adsorption_loading_value: adsorption loading numeric value
- adsorption_loading_unit: unit of adsorption loading
- pressure_value: pressure numeric value
- pressure_unit: unit of pressure
- experimental_method: experimental method name
- method_category: method category classification
- distinguishing_variable: factor that distinguishes otherwise similar data points
- doi: DOI or paper reference identifier
- filename: source file name

Rules:
1. Every column MUST map to exactly one role (use the most specific match)
2. If unsure, prefer null over guessing
3. Use column name + sample values to infer meaning
4. guest_molecule and guest_composition are different: guest_molecule is the NAME, guest_composition is the chemical FORMULA

Return ONLY valid JSON:
{{
    "column_roles": {{
        "guest_molecule": "actual_column_name_or_null",
        "guest_composition": "actual_column_name_or_null",
        "zeolite_name": "actual_column_name_or_null",
        "std_zeolite_name": "actual_column_name_or_null",
        "si_al_ratio": "actual_column_name_or_null",
        "modified_ion": "actual_column_name_or_null",
        "loading_value": "actual_column_name_or_null",
        "loading_unit": "actual_column_name_or_null",
        "diffusion_coefficient_value": "actual_column_name_or_null",
        "diffusion_coefficient_unit": "actual_column_name_or_null",
        "temperature_value": "actual_column_name_or_null",
        "temperature_unit": "actual_column_name_or_null",
        "concentration_value": "actual_column_name_or_null",
        "concentration_unit": "actual_column_name_or_null",
        "adsorption_loading_value": "actual_column_name_or_null",
        "adsorption_loading_unit": "actual_column_name_or_null",
        "pressure_value": "actual_column_name_or_null",
        "pressure_unit": "actual_column_name_or_null",
        "experimental_method": "actual_column_name_or_null",
        "method_category": "actual_column_name_or_null",
        "distinguishing_variable": "actual_column_name_or_null",
        "doi": "actual_column_name_or_null",
        "filename": "actual_column_name_or_null"
    }},
    "reasoning": "brief explanation of key mappings"
}}
"""
        try:
            messages = [
                {"role": "system", "content": "You are a data schema analyst. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0, max_tokens=2000
            )
            result_text = response.choices[0].message.content.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            return result.get("column_roles", {})
        except Exception as e:
            logger.error(f"Semantic role inference failed: {e}")
            return {}

    # =========================================================================
    # Layer 3: Role-to-Task Mapping
    # =========================================================================

    def _role_to_task_mapping(self, column_roles: Dict[str, str]) -> Dict[str, str]:
        """
        Layer 3: Extract task-relevant columns from role mapping.
        Returns flat dict: {role_key: actual_column_name}
        Only includes roles that were actually mapped (non-null).
        """
        task_cols = {}
        for role, col_name in column_roles.items():
            if col_name and col_name != 'null' and col_name.lower() != 'none':
                task_cols[role] = col_name
        return task_cols

    # =========================================================================
    # Unified Query Understanding (LLM)
    # =========================================================================

    def _understand_query(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to understand the user query holistically.
        Returns: molecules, query_type, is_separation, specific_zeolite, needs_prediction, entity_count, route
        """
        prompt = f"""
Analyze this query about zeolite materials. Return ONLY valid JSON.

Query: {query}

Return JSON:
{{
    "molecules": ["standard_english_name", ...],
    "query_type": "ranking" | "comparison" | "general",
    "is_separation": true | false,
    "specific_zeolite": "zeolite_code" | null,
    "needs_prediction": true | false,
    "entity_count": <integer>,
    "route": "graphrag" | "qa"
}}

Field rules:
- molecules: ALL guest molecules mentioned EXPLICITLY or IMPLICITLY. Infer from application context: "natural gas purification" → ["methane","carbon dioxide"], "carbon capture" → ["carbon dioxide"], "dehydration" → ["water"], "air separation" → ["nitrogen","dioxygen"]. CO2→carbon dioxide, CH4→methane, N2→nitrogen, H2O→water, C2H6→ethane, etc. Handle Chinese.
- query_type:
  "ranking" = asking which zeolite is BEST (e.g. "which zeolite is best for...", "推荐最好的...")
  "comparison" = asking HOW DIFFERENT two entities are (e.g. "how different are X and Y in ZSM-5")
  "general" = neither — includes exploratory questions, pattern analysis, single-entity lookups (e.g. "show data above 300K", "tell me about MFI", "分析CO2的扩散规律")
- is_separation: true ONLY if about separating/distinguishing TWO OR MORE specific molecules from each other. If only ONE molecule is mentioned, is_separation MUST be false even if the word "selectivity" or "separation" appears.
- specific_zeolite: zeolite code if asking about a SPECIFIC zeolite, else null
- needs_prediction: true ONLY when ALL of: (a) query_type is "ranking", (b) 2+ molecules mentioned, (c) no specific zeolite mentioned
- entity_count: count of DISTINCT entities mentioned. A specific zeolite = 1 entity. A specific guest molecule = 1 entity. A specific application domain (e.g. "natural gas purification", "dehydration") = 1 entity. Count them all.
  "tell me about MFI" → 1. "CO2 diffusion" → 1. "best zeolite for para-xylene" → 1. "natural gas purification" → 1. "CO2 in MFI" → 2. "CO2 vs CH4" → 2. "best zeolite for CO2/CH4 separation" → 2.
- route: "graphrag" when entity_count <= 1 (single-entity or single-domain exploration → GraphRAG can show all related zeolites/guests). "qa" for entity_count > 1 (comparisons, multi-entity lookups).

Examples:
- "Which zeolite is best for separating CO2 and CH4?" → {{"molecules":["carbon dioxide","methane"],"query_type":"ranking","is_separation":true,"specific_zeolite":null,"needs_prediction":true,"entity_count":2,"route":"qa"}}
- "How different are ethane and ethene in ZSM-5?" → {{"molecules":["ethane","ethene"],"query_type":"comparison","is_separation":false,"specific_zeolite":"MFI","needs_prediction":false,"entity_count":3,"route":"qa"}}
- "二氧化碳在MFI中的扩散" → {{"molecules":["carbon dioxide"],"query_type":"general","is_separation":false,"specific_zeolite":"MFI","needs_prediction":false,"entity_count":2,"route":"qa"}}
- "Which zeolite has the best natural gas purification performance→ {"molecules":["methane","carbon dioxide"],"query_type":"ranking","is_separation":true,"specific_zeolite":null,"needs_prediction":true,"entity_count":2,"route":"qa"}}
- "Tell me about MFI zeolite diffusion properties" → {{"molecules":[],"query_type":"general","is_separation":false,"specific_zeolite":"MFI","needs_prediction":false,"entity_count":1,"route":"graphrag"}}
- "CO2 diffusion patterns across different zeolites" → {{"molecules":["carbon dioxide"],"query_type":"general","is_separation":false,"specific_zeolite":null,"needs_prediction":false,"entity_count":1,"route":"graphrag"}}
- "Which molecular sieve is most favorable for the selectivity of para-xylene?" → {{"molecules":["1,4-dimethylbenzene"],"query_type":"ranking","is_separation":false,"specific_zeolite":null,"needs_prediction":false,"entity_count":1,"route":"graphrag"}}
- "分析MFI分子筛中扩散的规律" → {{"molecules":[],"query_type":"general","is_separation":false,"specific_zeolite":"MFI","needs_prediction":false,"entity_count":1,"route":"graphrag"}}
- "分析二氧化碳在不同分子筛中的扩散规律" → {{"molecules":["carbon dioxide"],"query_type":"general","is_separation":false,"specific_zeolite":null,"needs_prediction":false,"entity_count":1,"route":"graphrag"}}
- "show all data above 300K" → {{"molecules":[],"query_type":"general","is_separation":false,"specific_zeolite":null,"needs_prediction":false,"entity_count":0,"route":"qa"}}
"""
        try:
            messages = [
                {"role": "system", "content": "You are a query understanding expert. Return JSON only."},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0, max_tokens=2000
            )
            result_text = response.choices[0].message.content.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            return {
                "molecules": result.get("molecules", []),
                "query_type": result.get("query_type", "general"),
                "is_separation": result.get("is_separation", False),
                "specific_zeolite": result.get("specific_zeolite"),
                "needs_prediction": result.get("needs_prediction", False),
                "entity_count": result.get("entity_count", len(result.get("molecules", []))),
                "route": result.get("route", "qa"),
            }
        except Exception as e:
            logger.error(f"Query understanding failed: {e}, falling back to keyword extraction")
            return {"molecules": self._extract_molecules_fallback(query),
                    "query_type": "general", "is_separation": False, "specific_zeolite": None,
                    "needs_prediction": False, "entity_count": 0, "route": "qa"}

    def _extract_molecules_fallback(self, query: str) -> List[str]:
        """Keyword-based fallback when LLM query understanding fails."""
        query_lower = query.lower()
        molecules = []
        # Chemical formula detection
        import re
        for m in re.findall(r'\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b', query):
            noise = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'mol', 'log', 'max', 'min',
                     'doi', 'mfi', 'fau', 'cha', 'lta', 'ddr', 'mor', 'bea', 'zsm',
                     'si', 'al', 'na', 'ca', 'cu', 'fe', 'zn'}
            if m.lower() not in noise and len(m) >= 2:
                molecules.append(m.lower())
        return molecules

    # Keep backward compatibility
    def _extract_molecules_from_query(self, query: str) -> List[str]:
        """Deprecated: use _understand_query instead."""
        return self._understand_query(query).get("molecules", [])

    def _generate_material_keywords(self, molecules: List[str]) -> List[str]:
        """Generate search keywords for a list of molecule names."""
        keywords = []
        for mol in molecules:
            mol_lower = mol.lower().strip()
            keywords.append(mol_lower)
            keywords.append(mol_lower.replace(' ', '_'))
            keywords.append(mol_lower.replace(' ', ''))
            # Add common Chinese translations
            cn_map = {
                'methane': '甲烷', 'carbon dioxide': '二氧化碳', 'ethane': '乙烷',
                'propane': '丙烷', 'nitrogen': '氮气', 'oxygen': '氧气',
                'hydrogen': '氢气', 'water': '水', 'ammonia': '氨',
                'carbon monoxide': '一氧化碳', 'dihydrogen': '氢气',
            }
            if mol_lower in cn_map:
                keywords.append(cn_map[mol_lower])
        return keywords

    # =========================================================================
    # Main Orchestration: 3-Layer Pipeline
    # =========================================================================

    def map_query_to_columns(self, query: str, available_columns: List[str],
                            sample_data: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Use 3-layer framework to intelligently map query to data columns.

        Returns mapping result including column_roles, task columns, and detected molecules.
        """
        logger.info(f"3-Layer mapping for query: {query[:80]}...")

        # Layer 1: Data type classification (rule-based)
        col_types = self._classify_column_types(available_columns, sample_data)
        logger.info(f"L1 data types: {len(col_types)} columns classified")

        # Layer 2: LLM semantic role inference
        column_roles = self._semantic_role_inference(query, available_columns, col_types, sample_data)
        logger.info(f"L2 semantic roles mapped: {sum(1 for v in column_roles.values() if v and v != 'null')} roles assigned")

        # Layer 3: Role-to-task mapping
        task_cols = self._role_to_task_mapping(column_roles)
        logger.info(f"L3 task columns: {len(task_cols)} columns for tasks")

        # Unified LLM query understanding (molecules + intent + separation + zeolite)
        understanding = self._understand_query(query)
        detected_molecules = understanding.get("molecules", [])
        logger.info(f"LLM query understanding: {understanding}")

        # If LLM layer returned no useful roles, use fallback
        if not task_cols:
            logger.warning("LLM mapping returned no roles, using fallback")
            fb = self._fallback_mapping(query, available_columns)
            fb["query_type"] = understanding.get("query_type", "general")
            fb["is_separation"] = understanding.get("is_separation", False)
            fb["specific_zeolite"] = understanding.get("specific_zeolite")
            fb["entity_count"] = understanding.get("entity_count", 0)
            fb["route"] = understanding.get("route", "qa")
            return fb

        # Build result (backward-compatible keys + LLM query understanding)
        result = {
            "column_roles": column_roles,
            "task_columns": task_cols,
            "material_column": task_cols.get("guest_molecule"),
            "zeolite_column": task_cols.get("std_zeolite_name") or task_cols.get("zeolite_name"),
            "value_column": task_cols.get("diffusion_coefficient_value"),
            "temperature_column": task_cols.get("temperature_value"),
            "concentration_column": task_cols.get("concentration_value"),
            "method_column": task_cols.get("method_category") or task_cols.get("experimental_method"),
            "pressure_column": task_cols.get("pressure_value"),
            "doi_column": task_cols.get("doi"),
            "guest_composition_column": task_cols.get("guest_composition"),
            "si_al_ratio_column": task_cols.get("si_al_ratio"),
            "modified_ion_column": task_cols.get("modified_ion"),
            "loading_value_column": task_cols.get("loading_value"),
            "distinguishing_variable_column": task_cols.get("distinguishing_variable"),
            "detected_materials": detected_molecules,
            "material_keywords": self._generate_material_keywords(detected_molecules),
            # LLM query understanding fields (replaces keyword-based detection)
            "query_type": understanding.get("query_type", "general"),
            "is_separation": understanding.get("is_separation", False),
            "specific_zeolite": understanding.get("specific_zeolite"),
            "needs_prediction": understanding.get("needs_prediction", False),
            "entity_count": understanding.get("entity_count", 0),
            "route": understanding.get("route", "qa"),
            "reasoning": f"3-layer mapping + LLM understanding: "
                         f"type={understanding.get('query_type')}, "
                         f"sep={understanding.get('is_separation')}, "
                         f"mols={len(detected_molecules)}, "
                         f"route={understanding.get('route', 'qa')}"
        }

        logger.info(f"Mapping complete: material_col={result['material_column']}, zeolite_col={result['zeolite_column']}")
        return result

    # =========================================================================
    # Fallback Mapping (Rule-Based)
    # =========================================================================

    def _fallback_mapping(self, query: str, columns: List[str]) -> Dict[str, Any]:
        """Rule-based fallback mapping when LLM fails"""
        logger.info("Using fallback rules for column mapping")

        column_roles = {}

        # Role -> keyword hints for fallback matching
        role_hints = {
            'guest_molecule': ['guest', 'material', 'molecule', 'adsorbate', 'compound', 'species'],
            'guest_composition': ['composition', 'formula', '组分', '组成'],
            'std_zeolite_name': ['std_zeolite', 'standard', 'framework_code'],
            'zeolite_name': ['zeolite', 'framework', '分子筛'],
            'si_al_ratio': ['si_al', 'silica', 'alumina', 'si/al', 'sio2', '硅铝'],
            'modified_ion': ['modified', 'ion', 'metal', 'cation', '改性', '离子', '金属'],
            'loading_value': ['loading', 'load', '负载'],
            'loading_unit': ['loading_unit', '负载单位'],
            'diffusion_coefficient_value': ['diffusion', 'diffusivity', 'coefficient', 'd_coeff', '扩散系数'],
            'diffusion_coefficient_unit': ['diffusion_unit', '扩散单位'],
            'temperature_value': ['temperature', 'temp', '温度'],
            'temperature_unit': ['temperature_unit', '温度单位'],
            'concentration_value': ['concentration', 'conc', '浓度'],
            'concentration_unit': ['concentration_unit', '浓度单位'],
            'adsorption_loading_value': ['adsorption_loading', '吸附'],
            'adsorption_loading_unit': ['adsorption_loading_unit'],
            'pressure_value': ['pressure', 'press', '压力'],
            'pressure_unit': ['pressure_unit', '压力单位'],
            'method_category': ['method_category', 'category'],
            'experimental_method': ['experimental_method', 'experiment', 'method', '实验方法'],
            'distinguishing_variable': ['distinguishing', 'variable', '区分', '变量'],
            'doi': ['doi', 'reference', '文献'],
            'filename': ['filename', 'file', 'source', '文件'],
        }

        for role, hints in role_hints.items():
            for col in columns:
                col_lower = col.lower()
                if any(hint in col_lower for hint in hints):
                    column_roles[role] = col
                    break
            if role not in column_roles:
                column_roles[role] = None

        # Detect molecules from query keywords
        detected = []
        query_lower = query.lower()
        mol_hints = {
            'methane': ['methane', 'ch4', '甲烷'],
            'carbon dioxide': ['carbon dioxide', 'co2', '二氧化碳'],
            'ethane': ['ethane', 'c2h6', '乙烷'],
            'propane': ['propane', 'c3h8', '丙烷'],
            'nitrogen': ['nitrogen', 'n2', '氮气'],
            'oxygen': ['oxygen', 'o2', '氧气'],
            'hydrogen': ['hydrogen', 'h2', '氢气', 'dihydrogen'],
            'carbon monoxide': ['carbon monoxide', 'co ', '一氧化碳'],
            'water': ['water', 'h2o', '水'],
        }
        for mol, keywords in mol_hints.items():
            if any(kw in query_lower for kw in keywords):
                detected.append(mol)

        task_cols = self._role_to_task_mapping(column_roles)

        return {
            "column_roles": column_roles,
            "task_columns": task_cols,
            "material_column": task_cols.get("guest_molecule"),
            "zeolite_column": task_cols.get("std_zeolite_name") or task_cols.get("zeolite_name"),
            "value_column": task_cols.get("diffusion_coefficient_value"),
            "temperature_column": task_cols.get("temperature_value"),
            "concentration_column": task_cols.get("concentration_value"),
            "method_column": task_cols.get("method_category") or task_cols.get("experimental_method"),
            "pressure_column": task_cols.get("pressure_value"),
            "doi_column": task_cols.get("doi"),
            "guest_composition_column": task_cols.get("guest_composition"),
            "si_al_ratio_column": task_cols.get("si_al_ratio"),
            "modified_ion_column": task_cols.get("modified_ion"),
            "loading_value_column": task_cols.get("loading_value"),
            "distinguishing_variable_column": task_cols.get("distinguishing_variable"),
            "detected_materials": detected,
            "material_keywords": self._generate_material_keywords(detected),
            "reasoning": "Rule-based fallback (Layer 1 only)"
        }
