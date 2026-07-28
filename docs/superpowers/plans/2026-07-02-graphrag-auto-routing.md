# GraphRAG Auto-Routing + Project1/2 Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the manual GraphRAG toggle button. Auto-route queries: single-entity exploration queries → GraphRAG (with pre-built project1_guest + project2_zeolite knowledge graph); multi-entity comparison/ranking queries → existing Q&A pipeline.

**Architecture:** Extend `_understand_query` LLM call to return a `route` decision. Build a pre-built knowledge graph from project1_guest and project2_zeolite statistical summaries at startup. Rewire `process_query` to auto-route based on the LLM's decision. Remove all manual GraphRAG toggle UI and backend code.

**Tech Stack:** Python 3.12, FastAPI, networkx, DeepSeek v4 Pro

**Decisions confirmed:**
- GraphRAG output includes: text analysis + statistical tables + charts (base64 PNG from project1/project2 figures)
- Default route when LLM fails: **QA** (safer, preserves existing behavior)

## Global Constraints

- All LLM query understanding happens in a single `_understand_query` call — no new LLM calls for routing
- Project1/project2 graph is built once at startup, not per-query
- GraphRAG subgraph anchoring: when query mentions a guest molecule → fix subgraph anchored on that guest; when query mentions a zeolite → fix subgraph anchored on that zeolite
- Backward compatible: existing Q&A pipeline (separation ranking, Tier 1/Tier 2, temperature pairing) must continue to work unchanged

---

## Routing Logic (defined in `_understand_query` prompt)

| Condition | Route | Example |
|-----------|-------|---------|
| 1 entity (zeolite OR guest only), no comparison/ranking | `graphrag` | "tell me about MFI", "CO2 diffusion patterns", "分析MFI的扩散规律" |
| 2+ entities (two zeolites, two guests, or zeolite+guest) | `qa` | "CO2 vs CH4 in MFI", "best zeolite for CO2/CH4 separation" |
| Explicit comparison/ranking/separation | `qa` | "which zeolite is best for...", "compare X and Y" |
| Specific zeolite + specific guest = 2 entities | `qa` | "CO2 diffusion in MFI at 300K" |

**Key insight:** "CO2 diffusion in MFI" = 2 entities (CO2 + MFI) → Q&A (needs specific data lookup). "Tell me about CO2 diffusion" = 1 entity (CO2) → GraphRAG (exploratory, shows patterns across all zeolites).

---

### Task 1: Extend `_understand_query` with routing decision

**Files:**
- Modify: `core/intelligent_column_mapper.py:215-276`

**Interfaces:**
- Consumes: Nothing new (existing method)
- Produces: `_understand_query` now returns `{"molecules": [...], "query_type": "...", "is_separation": bool, "specific_zeolite": str|None, "needs_prediction": bool, "route": "graphrag"|"qa"}`

- [ ] **Step 1: Update the `_understand_query` prompt to include route decision**

In `core/intelligent_column_mapper.py`, replace the `_understand_query` method's prompt and return value:

```python
def _understand_query(self, query: str) -> Dict[str, Any]:
    """
    Use LLM to understand the user query holistically.
    Returns: molecules, query_type, is_separation, specific_zeolite, needs_prediction, route
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
- molecules: ALL guest molecules mentioned. CO2→carbon dioxide, CH4→methane, N2→nitrogen, H2O→water, C2H6→ethane, etc. Handle Chinese.
- query_type:
  "ranking" = asking which zeolite is BEST (e.g. "which zeolite is best for...", "推荐最好的...")
  "comparison" = asking HOW DIFFERENT two entities are (e.g. "how different are X and Y in ZSM-5")
  "general" = neither (e.g. "show data above 300K", "tell me about...", "分析...的规律")
- is_separation: true if about separating/distinguishing two molecules
- specific_zeolite: zeolite code if asking about a SPECIFIC zeolite, else null
- needs_prediction: true ONLY when ALL of: (a) query_type is "ranking", (b) 2+ molecules, (c) no specific zeolite mentioned
- entity_count: count of DISTINCT entities mentioned. A specific zeolite = 1 entity. A specific guest molecule = 1 entity. Count them.
  Examples: "tell me about MFI" → 1 (only MFI). "CO2 diffusion" → 1 (only CO2). "CO2 in MFI" → 2 (CO2 + MFI). "CO2 vs CH4" → 2. "best zeolite for CO2/CH4" → 2 (two guests, no specific zeolite).
- route: "graphrag" when entity_count == 1 AND query_type is NOT "ranking" (exploratory/single-entity question). "qa" for everything else (comparisons, rankings, multi-entity lookups).

Examples:
- "Which zeolite is best for separating CO2 and CH4?" → {{"molecules":["carbon dioxide","methane"],"query_type":"ranking","is_separation":true,"specific_zeolite":null,"needs_prediction":true,"entity_count":2,"route":"qa"}}
- "How different are ethane and ethene in ZSM-5?" → {{"molecules":["ethane","ethene"],"query_type":"comparison","is_separation":false,"specific_zeolite":"MFI","needs_prediction":false,"entity_count":3,"route":"qa"}}
- "二氧化碳在MFI中的扩散" → {{"molecules":["carbon dioxide"],"query_type":"general","is_separation":false,"specific_zeolite":"MFI","needs_prediction":false,"entity_count":2,"route":"qa"}}
- "Tell me about MFI zeolite diffusion properties" → {{"molecules":[],"query_type":"general","is_separation":false,"specific_zeolite":"MFI","needs_prediction":false,"entity_count":1,"route":"graphrag"}}
- "CO2 diffusion patterns across different zeolites" → {{"molecules":["carbon dioxide"],"query_type":"general","is_separation":false,"specific_zeolite":null,"needs_prediction":false,"entity_count":1,"route":"graphrag"}}
- "分析MFI分子筛中扩散的规律" → {{"molecules":[],"query_type":"general","is_separation":false,"specific_zeolite":"MFI","needs_prediction":false,"entity_count":1,"route":"graphrag"}}
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
            "route": result.get("route", "qa"),  # Default to qa for safety
        }
    except Exception as e:
        logger.error(f"Query understanding failed: {e}, falling back to keyword extraction")
        return {"molecules": self._extract_molecules_fallback(query),
                "query_type": "general", "is_separation": False, "specific_zeolite": None,
                "needs_prediction": False, "entity_count": 0, "route": "qa"}
```

- [ ] **Step 2: Update `map_query_to_columns` to pass through `route`**

In the same file, update the result dict in `map_query_to_columns` (around line 372-386):

```python
# In map_query_to_columns, add route to result:
result = {
    # ... existing fields ...
    "detected_materials": detected_molecules,
    "material_keywords": self._generate_material_keywords(detected_molecules),
    "query_type": understanding.get("query_type", "general"),
    "is_separation": understanding.get("is_separation", False),
    "specific_zeolite": understanding.get("specific_zeolite"),
    "needs_prediction": understanding.get("needs_prediction", False),
    "route": understanding.get("route", "qa"),  # NEW
    "entity_count": understanding.get("entity_count", 0),  # NEW
    # ... rest ...
}
```

Also update the fallback result in `map_query_to_columns` to include `route` and `entity_count`.

- [ ] **Step 3: Verify syntax**

Run: `py -3.13 -m py_compile core\intelligent_column_mapper.py`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add core/intelligent_column_mapper.py
git commit -m "feat: extend _understand_query with route decision for GraphRAG vs QA auto-routing"
```

---

### Task 2: Build pre-built knowledge graph from project1_guest + project2_zeolite

**Files:**
- Create: `core/project_graph_builder.py`
- Read: `project1_guest/output/stats/*.json`
- Read: `project2_zeolite/output/stats/*.json`

**Interfaces:**
- Consumes: Nothing (reads from disk)
- Produces: `ProjectGraphBuilder.build()` → `nx.DiGraph`, `ProjectGraphBuilder.get_subgraph(anchor_type, anchor_value)` → `Dict[str, Any]`

- [ ] **Step 1: Create `core/project_graph_builder.py`**

```python
"""
Project Graph Builder - Build knowledge graph from project1_guest and project2_zeolite data.

Graph Structure:
  Nodes:
    - Guest: guest molecule (e.g., CO2, CH4, H2O)
    - Zeolite: specific zeolite (e.g., MFI, H-ZSM-5, DD3R)
    - Topology: framework topology (e.g., MFI, FAU, LTA, DDR)
    - Ion: exchange ion (e.g., H+, Na+, K+, Cu+)

  Edges:
    - Guest --[diffuses_in]--> Zeolite : {mean_logD, median_logD, std_logD, count, D_range_orders, Ea, R2, n_arrhenius}
    - Guest --[diffuses_in]--> Topology : {mean_logD, median_logD, std_logD, count}
    - Guest --[affected_by]--> Ion : {mean_logD, median_logD, std_logD, count}
    - Zeolite --[has_topology]--> Topology : {}
    - Zeolite --[hosts]--> Guest : {mean_logD, median_logD, std_logD, count, kinetic_diameter_A, Ea, R2, n_arrhenius}
    - Guest --[has_size]--> KineticDiameter : {value_A}

  Two perspectives:
    project1 (guest-centric): For a guest, which zeolites/topologies/ions affect its diffusion?
    project2 (zeolite-centric): For a zeolite, which guests diffuse through it?
"""
import json
import os
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from collections import defaultdict


class ProjectGraphBuilder:
    """Build and query knowledge graph from project1_guest and project2_zeolite statistics."""

    def __init__(self, project1_dir: str = None, project2_dir: str = None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project1_dir = project1_dir or os.path.join(base, "project1_guest", "output", "stats")
        self.project2_dir = project2_dir or os.path.join(base, "project2_zeolite", "output", "stats")
        self.graph = nx.DiGraph()
        self._guest_index = defaultdict(list)   # guest_name → [node_ids]
        self._zeolite_index = defaultdict(list)  # zeolite_name → [node_ids]
        self._topology_index = defaultdict(list) # topology → [node_ids]
        self._built = False

    def build(self) -> nx.DiGraph:
        """Build the complete knowledge graph. Idempotent — clears and rebuilds."""
        logger.info("Building project knowledge graph from project1 + project2 data...")
        self.graph.clear()
        self._guest_index.clear()
        self._zeolite_index.clear()
        self._topology_index.clear()

        self._load_project1_guest()
        self._load_project2_zeolite()

        self._built = True
        logger.info(f"Graph built: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        logger.info(f"  Guests: {len(self._guest_index)}, Zeolites: {len(self._zeolite_index)}, Topologies: {len(self._topology_index)}")
        return self.graph

    def _load_project1_guest(self):
        """Load project1 data: guest-centric view.
        For each guest molecule, add edges to zeolites, topologies, and ions.
        """
        if not os.path.isdir(self.project1_dir):
            logger.warning(f"project1_guest stats dir not found: {self.project1_dir}")
            return

        for fname in sorted(os.listdir(self.project1_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self.project1_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")
                continue

            guest_name = data.get("guest_name", fname.replace(".json", ""))
            guest_node = f"guest:{guest_name}"
            self.graph.add_node(guest_node, type="guest", name=guest_name,
                              n_total=data.get("n_total", 0))
            self._guest_index[guest_name.lower()].append(guest_node)

            # Guest → Zeolite edges
            for z_entry in data.get("by_zeolite", []):
                z_name = z_entry["zeolite_name"]
                z_node = f"zeolite:{z_name}"
                self.graph.add_node(z_node, type="zeolite", name=z_name)
                self._zeolite_index[z_name.lower()].append(z_node)
                self.graph.add_edge(guest_node, z_node, relation="diffuses_in",
                    mean_logD=z_entry.get("mean_logD"), median_logD=z_entry.get("median_logD"),
                    std_logD=z_entry.get("std_logD"), count=z_entry.get("count"),
                    D_range_orders=z_entry.get("D_range_orders"))

            # Guest → Topology edges
            for t_entry in data.get("by_topology", []):
                t_name = t_entry.get("topology", t_entry.get("拓扑", "unknown"))
                t_node = f"topology:{t_name}"
                self.graph.add_node(t_node, type="topology", name=t_name)
                self._topology_index[t_name.lower()].append(t_node)
                self.graph.add_edge(guest_node, t_node, relation="diffuses_in_topology",
                    mean_logD=t_entry.get("mean_logD"), median_logD=t_entry.get("median_logD"),
                    std_logD=t_entry.get("std_logD"), count=t_entry.get("n", t_entry.get("count")))

            # Guest → Ion edges (if present)
            for i_entry in data.get("by_ion", []):
                i_name = i_entry.get("ion", i_entry.get("离子", "unknown"))
                if not i_name or i_name == "(none)":
                    continue
                i_node = f"ion:{i_name}"
                self.graph.add_node(i_node, type="ion", name=i_name)
                self.graph.add_edge(guest_node, i_node, relation="affected_by_ion",
                    mean_logD=i_entry.get("mean_logD"), count=i_entry.get("n", i_entry.get("count")))

            # Arrhenius: guest overall Ea
            arr = data.get("arrhenius", {})
            if arr:
                self.graph.nodes[guest_node]["Ea_kJ_mol"] = arr.get("Ea_kJ_mol")
                self.graph.nodes[guest_node]["Ea_R2"] = arr.get("R2")

        logger.info(f"Loaded project1_guest: {len(self._guest_index)} guests")

    def _load_project2_zeolite(self):
        """Load project2 data: zeolite-centric view.
        For each zeolite, add edges to guest molecules.
        """
        if not os.path.isdir(self.project2_dir):
            logger.warning(f"project2_zeolite stats dir not found: {self.project2_dir}")
            return

        for fname in sorted(os.listdir(self.project2_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self.project2_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")
                continue

            z_name = data.get("zeolite_name", fname.replace(".json", ""))
            z_node = f"zeolite:{z_name}"
            # May already exist from project1; update with project2 data
            if z_node not in self.graph.nodes:
                self.graph.add_node(z_node, type="zeolite", name=z_name)
            self._zeolite_index[z_name.lower()].append(z_node)

            topology = data.get("topology")
            if topology:
                self.graph.nodes[z_node]["topology"] = topology
                t_node = f"topology:{topology}"
                if t_node not in self.graph.nodes:
                    self.graph.add_node(t_node, type="topology", name=topology)
                self.graph.add_edge(z_node, t_node, relation="has_topology")

            self.graph.nodes[z_node]["n_total"] = data.get("n_total", 0)

            # Zeolite → Guest edges
            for g_entry in data.get("by_guest", []):
                g_name = g_entry["guest_molecule"]
                g_node = f"guest:{g_name}"
                if g_node not in self.graph.nodes:
                    self.graph.add_node(g_node, type="guest", name=g_name)
                self._guest_index[g_name.lower()].append(g_node)

                kd = g_entry.get("kinetic_diameter_A")
                self.graph.add_edge(z_node, g_node, relation="hosts",
                    mean_logD=g_entry.get("mean_logD"), median_logD=g_entry.get("median_logD"),
                    std_logD=g_entry.get("std_logD"), count=g_entry.get("count"),
                    D_range_orders=g_entry.get("D_range_orders"),
                    kinetic_diameter_A=kd)

                # If kinetic diameter is known, add as node property
                if kd is not None:
                    self.graph.nodes[g_node]["kinetic_diameter_A"] = kd

            # Arrhenius per guest
            for a_entry in data.get("arrhenius", []):
                g_name = a_entry.get("guest_molecule", a_entry.get("客体分子"))
                if g_name:
                    g_node = f"guest:{g_name}"
                    if g_node in self.graph.nodes:
                        # Update the zeolite→guest edge with Arrhenius data
                        if self.graph.has_edge(z_node, g_node):
                            self.graph.edges[z_node, g_node]["Ea_kJ_mol"] = a_entry.get("Ea_kJ_mol")
                            self.graph.edges[z_node, g_node]["Ea_R2"] = a_entry.get("R2")
                            self.graph.edges[z_node, g_node]["n_arrhenius"] = a_entry.get("n")

        logger.info(f"Loaded project2_zeolite: {len(self._zeolite_index)} zeolites")

    # =========================================================================
    # Subgraph Extraction
    # =========================================================================

    def get_subgraph(self, anchor_type: str, anchor_value: str, max_depth: int = 1) -> Dict[str, Any]:
        """Extract a subgraph anchored on a specific guest or zeolite.

        Args:
            anchor_type: "guest" or "zeolite"
            anchor_value: name of the guest or zeolite (case-insensitive match)
            max_depth: BFS depth from anchor node (default 1 = direct neighbors)

        Returns:
            {
                "anchor": {"type": str, "name": str, "node_id": str},
                "nodes": [{"id": str, "type": str, "name": str, "properties": {...}}, ...],
                "edges": [{"source": str, "target": str, "relation": str, "properties": {...}}, ...],
                "summary": str  # Text summary for LLM context
            }
        """
        if not self._built:
            self.build()

        # Find anchor node
        anchor_node = None
        anchor_lower = anchor_value.lower().strip()

        if anchor_type == "guest":
            candidates = self._guest_index.get(anchor_lower, [])
        elif anchor_type == "zeolite":
            candidates = self._zeolite_index.get(anchor_lower, [])
        else:
            candidates = []

        # Fuzzy match if exact not found
        if not candidates:
            for node_id, data in self.graph.nodes(data=True):
                if data.get("type") == anchor_type:
                    name = data.get("name", "").lower()
                    if anchor_lower in name or name in anchor_lower:
                        candidates.append(node_id)

        if not candidates:
            return {"anchor": None, "nodes": [], "edges": [], "summary": "",
                    "error": f"No {anchor_type} matching '{anchor_value}' found"}

        anchor_node = candidates[0]  # Use first match

        # BFS to extract subgraph
        visited = set()
        subgraph_nodes = []
        subgraph_edges = []

        from collections import deque
        queue = deque([(anchor_node, 0)])
        visited.add(anchor_node)

        while queue:
            current, depth = queue.popleft()
            node_data = dict(self.graph.nodes[current])
            subgraph_nodes.append({
                "id": current,
                "type": node_data.get("type", "unknown"),
                "name": node_data.get("name", current),
                "properties": {k: v for k, v in node_data.items() if k not in ("type", "name")}
            })

            if depth >= max_depth:
                continue

            # Outgoing edges
            for _, neighbor, edge_data in self.graph.out_edges(current, data=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                subgraph_edges.append({
                    "source": current,
                    "target": neighbor,
                    "relation": edge_data.get("relation", "unknown"),
                    "properties": {k: v for k, v in edge_data.items() if k != "relation"}
                })

            # Incoming edges
            for predecessor, _, edge_data in self.graph.in_edges(current, data=True):
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append((predecessor, depth + 1))
                subgraph_edges.append({
                    "source": predecessor,
                    "target": current,
                    "relation": edge_data.get("relation", "unknown"),
                    "properties": {k: v for k, v in edge_data.items() if k != "relation"}
                })

        # Generate text summary
        summary = self._summarize_subgraph(anchor_type, anchor_value, subgraph_nodes, subgraph_edges)

        return {
            "anchor": {"type": anchor_type, "name": anchor_value, "node_id": anchor_node},
            "nodes": subgraph_nodes,
            "edges": subgraph_edges,
            "summary": summary
        }

    def _summarize_subgraph(self, anchor_type: str, anchor_value: str,
                           nodes: List[Dict], edges: List[Dict]) -> str:
        """Generate a text summary of the subgraph for LLM context."""
        lines = [f"=== Knowledge Graph: {anchor_type} '{anchor_value}' ===\n"]

        # Organize by edge relation
        by_relation = defaultdict(list)
        for edge in edges:
            by_relation[edge["relation"]].append(edge)

        if anchor_type == "guest":
            # Guest → Zeolite diffusion stats
            lines.append(f"## Diffusion across zeolites for {anchor_value}\n")
            diff_edges = by_relation.get("diffuses_in", []) + by_relation.get("hosts", [])
            # Deduplicate by target
            seen = set()
            zeolite_stats = []
            for e in diff_edges:
                target = e["target"]
                if target in seen:
                    continue
                seen.add(target)
                props = e["properties"]
                zeolite_stats.append({
                    "zeolite": target.replace("zeolite:", ""),
                    "mean_logD": props.get("mean_logD"),
                    "D_range": props.get("D_range_orders"),
                    "count": props.get("count"),
                    "Ea": props.get("Ea_kJ_mol"),
                })
            # Sort by mean_logD (higher = faster diffusion)
            zeolite_stats.sort(key=lambda x: x["mean_logD"] if x["mean_logD"] is not None else -999, reverse=True)
            for zs in zeolite_stats[:15]:
                ea_str = f", Ea={zs['Ea']:.1f} kJ/mol" if zs.get("Ea") is not None else ""
                lines.append(f"  {zs['zeolite']}: logD={zs['mean_logD']:.2f}, range={zs['D_range']} orders, n={zs['count']}{ea_str}")

            # Topology stats
            topo_edges = by_relation.get("diffuses_in_topology", [])
            if topo_edges:
                lines.append(f"\n## Topology influence\n")
                for e in topo_edges[:10]:
                    t_name = e["target"].replace("topology:", "")
                    props = e["properties"]
                    lines.append(f"  {t_name}: mean_logD={props.get('mean_logD', 'N/A')}, n={props.get('count', 'N/A')}")

            # Ion effects
            ion_edges = by_relation.get("affected_by_ion", [])
            if ion_edges:
                lines.append(f"\n## Ion effects\n")
                for e in ion_edges[:10]:
                    i_name = e["target"].replace("ion:", "")
                    props = e["properties"]
                    lines.append(f"  {i_name}: mean_logD={props.get('mean_logD', 'N/A')}, n={props.get('count', 'N/A')}")

        elif anchor_type == "zeolite":
            # Zeolite → Guest diffusion stats
            lines.append(f"## Guest molecules diffusing through {anchor_value}\n")
            guest_edges = by_relation.get("hosts", []) + by_relation.get("diffuses_in", [])
            seen = set()
            guest_stats = []
            for e in guest_edges:
                target = e["target"]
                if target in seen:
                    continue
                seen.add(target)
                props = e["properties"]
                guest_stats.append({
                    "guest": target.replace("guest:", ""),
                    "mean_logD": props.get("mean_logD"),
                    "D_range": props.get("D_range_orders"),
                    "count": props.get("count"),
                    "kd_A": props.get("kinetic_diameter_A"),
                    "Ea": props.get("Ea_kJ_mol"),
                })
            guest_stats.sort(key=lambda x: x["mean_logD"] if x["mean_logD"] is not None else -999, reverse=True)
            for gs in guest_stats[:15]:
                kd_str = f", d={gs['kd_A']}Å" if gs.get("kd_A") is not None else ""
                ea_str = f", Ea={gs['Ea']:.1f} kJ/mol" if gs.get("Ea") is not None else ""
                lines.append(f"  {gs['guest']}: logD={gs['mean_logD']:.2f}, range={gs['D_range']} orders, n={gs['count']}{kd_str}{ea_str}")

        lines.append(f"\nTotal: {len(nodes)} nodes, {len(edges)} edges in subgraph")
        return "\n".join(lines)

    def get_available_entities(self) -> Dict[str, List[str]]:
        """Return available guests and zeolites in the graph."""
        if not self._built:
            self.build()
        return {
            "guests": sorted(self._guest_index.keys()),
            "zeolites": sorted(self._zeolite_index.keys()),
            "topologies": sorted(self._topology_index.keys()),
            "guest_count": len(self._guest_index),
            "zeolite_count": len(self._zeolite_index),
        }
```

- [ ] **Step 2: Verify the builder loads correctly**

Run: `py -3.13 -c "from core.project_graph_builder import ProjectGraphBuilder; b = ProjectGraphBuilder(); b.build(); info = b.get_available_entities(); print(f'Guests: {info[\"guest_count\"]}, Zeolites: {info[\"zeolite_count\"]}')"`
Expected: Output showing guest count > 0 and zeolite count > 0.

- [ ] **Step 3: Test subgraph extraction for a guest**

Run: `py -3.13 -c "from core.project_graph_builder import ProjectGraphBuilder; b = ProjectGraphBuilder(); b.build(); sg = b.get_subgraph('guest', 'CO2'); print(sg['summary'][:500])"`
Expected: Summary showing CO2 diffusion stats across zeolites.

- [ ] **Step 4: Test subgraph extraction for a zeolite**

Run: `py -3.13 -c "from core.project_graph_builder import ProjectGraphBuilder; b = ProjectGraphBuilder(); b.build(); sg = b.get_subgraph('zeolite', 'MFI'); print(sg['summary'][:500])"`
Expected: Summary showing guest molecules in MFI.

- [ ] **Step 5: Commit**

```bash
git add core/project_graph_builder.py
git commit -m "feat: add ProjectGraphBuilder for pre-built knowledge graph from project1+project2"
```

---

### Task 3: Rebuild GraphRAG engine to use pre-built project graph

**Files:**
- Modify: `core/graphrag_engine.py`

**Interfaces:**
- Consumes: `ProjectGraphBuilder` instance (from Task 2)
- Produces: `GraphRAGEngine.analyze_with_graph(query, df, context)` — updated to use pre-built graph when available, fall back to on-the-fly CSV graph

- [ ] **Step 1: Update `GraphRAGEngine.__init__` to accept ProjectGraphBuilder**

Replace the `__init__` method in `core/graphrag_engine.py`:

```python
def __init__(self, rag_engine: HybridRAGEngine, llm_integration: LLMIntegration,
             project_graph_builder=None):
    """Initialize GraphRAG engine"""
    self.rag_engine = rag_engine
    self.llm_integration = llm_integration
    self.knowledge_graph = nx.DiGraph()  # On-the-fly graph from CSV
    self.entity_relations = defaultdict(list)
    
    # Pre-built project graph (lazy load)
    self.project_graph_builder = project_graph_builder
    self._project_graph_ready = False
    
    logger.info("GraphRAG engine initialization complete")
```

- [ ] **Step 2: Add method to analyze with pre-built graph**

Add a new method `analyze_with_project_graph` after the existing `analyze_with_graph`:

```python
def analyze_with_project_graph(self, query: str, anchor_type: str, 
                               anchor_value: str) -> Dict[str, Any]:
    """Use pre-built project1/project2 graph for exploratory analysis.
    
    Args:
        query: Original user query
        anchor_type: "guest" or "zeolite" — what the subgraph anchors on
        anchor_value: The specific guest or zeolite name
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
    subgraph = self.project_graph_builder.get_subgraph(anchor_type, anchor_value, max_depth=2)
    
    if subgraph.get("error"):
        return {"error": subgraph["error"]}
    
    # Build context for LLM
    context = subgraph["summary"]
    
    # Add available entities context
    entities = self.project_graph_builder.get_available_entities()
    context += f"\n\nAvailable guests in knowledge base: {len(entities['guests'])} molecules"
    context += f"\nAvailable zeolites in knowledge base: {len(entities['zeolites'])} structures"
    
    # Generate LLM response with enhanced prompt
    system_prompt = f"""You are a materials science expert analyzing zeolite diffusion data from a knowledge graph.

The data below shows statistical summaries extracted from {subgraph['anchor']['type']} '{subgraph['anchor']['name']}' and its related entities.

Output format:
1. Overview: Summarize the key characteristics of this {'guest molecule' if anchor_type == 'guest' else 'zeolite'}.
2. {'Which zeolites show the fastest/slowest diffusion for this guest? Rank them.' if anchor_type == 'guest' else 'Which guest molecules diffuse fastest/slowest through this zeolite? Rank them.'}
3. Patterns: What patterns do you observe? (e.g., correlation with kinetic diameter, topology effects, ion effects, temperature sensitivity via activation energy)
4. Notable findings: Any outliers, surprising results, or zeolites/guests that behave differently than expected.

Rules:
- Cite specific mean_logD values and count (n) from the data
- Use logD scale: higher = faster diffusion, each +1 = 10x faster
- Mention kinetic diameters and activation energies when available
- Be precise about statistical significance (note when n is small)
- Do not invent data not present in the provided context"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
    ]
    
    try:
        response = self.llm_integration._call_llm(messages, "analysis")
        return {
            "response": {
                "answer": response["content"],
                "model": self.llm_integration.model,
                "tokens_used": response["total_tokens"],
                "response_type": "analysis"
            },
            "graph_info": {
                "anchor": subgraph["anchor"],
                "node_count": len(subgraph["nodes"]),
                "edge_count": len(subgraph["edges"]),
                "source": "project_graph"
            }
        }
    except Exception as e:
        logger.error(f"Project graph analysis failed: {e}")
        return {"error": str(e)}
```

- [ ] **Step 3: Verify syntax**

Run: `py -3.13 -m py_compile core\graphrag_engine.py`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add core/graphrag_engine.py
git commit -m "feat: add analyze_with_project_graph for pre-built knowledge graph analysis"
```

---

### Task 4: Auto-route queries in table_agent.py

**Files:**
- Modify: `table_agent.py` (lines 54-55, 73, 92-97, 116-135, 314-330, 1692-1712)

**Interfaces:**
- Consumes: `route` field from `_understand_query` (Task 1), `ProjectGraphBuilder` (Task 2), updated `GraphRAGEngine` (Task 3)
- Produces: `process_query` auto-routes between GraphRAG and Q&A pipeline

- [ ] **Step 1: Import and initialize ProjectGraphBuilder in `TableAgent.__init__`**

In `table_agent.py`, add import (around line 20):

```python
from core.project_graph_builder import ProjectGraphBuilder
```

In `__init__`, update GraphRAG engine initialization (around line 54-55):

```python
# Initialize project graph builder (shared with GraphRAG)
self.project_graph_builder = ProjectGraphBuilder()

# Initialize GraphRAG engine with project graph
self.graphrag_engine = GraphRAGEngine(
    self.rag_engine, self.llm_integration,
    project_graph_builder=self.project_graph_builder
)
```

- [ ] **Step 2: Update `process_query` to auto-route based on LLM decision**

Replace lines 133-134 (the `use_graphrag` determination) with auto-routing logic:

```python
# Auto-route based on LLM query understanding (NOT manual toggle)
# The route decision comes from _understand_query, which runs inside
# map_query_to_columns below. We need to extract it from the mapping result.
# For now, initialize as None — will be set after mapping.
query_route = None
```

Then, after getting the mapping result (around line 163), extract the route:

```python
# Extract auto-routing decision from LLM query understanding
query_route = mapping.get("route", "qa")
entity_count = mapping.get("entity_count", 0)
logger.info(f"✓ Auto-route: {query_route} (entities={entity_count}, mode={query_mode})")
```

Then replace the method selection block (lines 314-330):

```python
# Auto-route: GraphRAG for single-entity exploration, QA for multi-entity
if query_route == "graphrag":
    # Determine anchor: guest molecule or zeolite?
    anchor_type = None
    anchor_value = None
    
    if detected_molecules:
        anchor_type = "guest"
        anchor_value = detected_molecules[0]
    elif mapping.get("specific_zeolite"):
        anchor_type = "zeolite"
        anchor_value = mapping.get("specific_zeolite")
    elif mapping.get("zeolite_column"):
        # Try to extract zeolite from query via LLM's specific_zeolite
        anchor_type = "zeolite"
        anchor_value = mapping.get("specific_zeolite") or "MFI"  # fallback
    
    if anchor_type and anchor_value:
        logger.info(f"Using GraphRAG with project graph: anchor={anchor_type}:{anchor_value}")
        graph_result = self.graphrag_engine.analyze_with_project_graph(
            query, anchor_type, anchor_value
        )
        if "error" in graph_result:
            # Fall back to standard LLM if graph analysis fails
            logger.warning(f"GraphRAG failed: {graph_result['error']}, falling back to standard QA")
            response = self.llm_integration.generate_response(
                query, enhanced_context, "analysis",
                is_comparison=False
            )
            graph_info = {"error": graph_result["error"]}
        else:
            response = graph_result["response"]
            graph_info = graph_result.get("graph_info", {})
    else:
        # Can't determine anchor, fall back to standard QA
        logger.warning("GraphRAG route but no anchor found, falling back to standard QA")
        response = self.llm_integration.generate_response(
            query, enhanced_context, "analysis", is_comparison=False
        )
        graph_info = {}
else:
    # Standard Q&A pipeline for multi-entity / comparison / ranking queries
    logger.info("Using standard Q&A pipeline")
    response = self.llm_integration.generate_response(
        query, enhanced_context, "analysis",
        is_comparison=comparison_info.get("is_comparison", False) or comparison_info.get("is_separation", False),
        molecule_a=detected_molecules[0] if len(detected_molecules) > 0 else None,
        molecule_b=detected_molecules[1] if len(detected_molecules) > 1 else None
    )
    graph_info = {}
```

- [ ] **Step 3: Remove `use_graphrag` instance variable and `set_method`**

Remove line 73: `self.use_graphrag = False`

Remove the entire `set_method` method (lines 1692-1695):

```python
# REMOVED: set_method — no longer needed with auto-routing
```

Update `get_system_status` (around line 1697-1712) to remove `current_method` and `graphrag_engine` references:

```python
def get_system_status(self) -> Dict[str, Any]:
    """Get system status"""
    return {
        "data_loaded": self.current_data is not None,
        "data_shape": self.current_data.shape if self.current_data is not None else None,
        "model": self.llm_integration.model,
        "rag_engine": "ready" if self.rag_engine else "not initialized",
        "project_graph": {
            "ready": self.project_graph_builder._built if self.project_graph_builder else False,
            "guests": len(self.project_graph_builder._guest_index) if self.project_graph_builder else 0,
            "zeolites": len(self.project_graph_builder._zeolite_index) if self.project_graph_builder else 0,
        } if self.project_graph_builder else "not available",
    }
```

- [ ] **Step 4: Verify syntax**

Run: `py -3.13 -m py_compile table_agent.py`
Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add table_agent.py
git commit -m "feat: auto-route queries between GraphRAG (single-entity) and QA (multi-entity) based on LLM understanding"
```

---

### Task 5: Remove GraphRAG button from UI and clean up backend

**Files:**
- Modify: `templates/index.html:97-106` (remove toggle HTML)
- Modify: `templates/index.html:125` (remove `useGR` variable)
- Modify: `templates/index.html:141` (remove toggle event listener)
- Modify: `templates/index.html:191` (remove `use_graphrag` from form data)
- Modify: `app.py:86-112` (remove `use_graphrag` parameter from `/api/query`)
- Modify: `app.py:261-276` (remove `/api/set-method` endpoint)

**Interfaces:**
- Consumes: Updated `table_agent.process_query` (no longer accepts `use_graphrag`)
- Produces: Clean UI without GraphRAG toggle, clean API without method switching

- [ ] **Step 1: Remove GraphRAG toggle from HTML sidebar**

In `templates/index.html`, remove the entire "Analysis Mode" section (lines 97-106):

```html
<!-- REMOVED: Analysis Mode toggle — now auto-routed by AI -->
```

- [ ] **Step 2: Remove JS variable and toggle handler**

Remove line 125: `var useGR=false;` → delete this line.

Remove line 141 (toggle event listener):
```javascript
// REMOVED: GraphRAG toggle handler — now auto-routed
```

- [ ] **Step 3: Remove `use_graphrag` from form data in `sendQ()`**

In `sendQ()` (line 191), change:
```javascript
var fd=new FormData();fd.append('query',q);fd.append('use_graphrag',useGR);
```
To:
```javascript
var fd=new FormData();fd.append('query',q);
```

- [ ] **Step 4: Update `/api/query` in `app.py`**

Remove `use_graphrag` parameter from the route:

```python
@app.post("/api/query")
async def process_query(
    query: str = Form(...)
):
    """Process user query"""
    try:
        if table_agent.current_data is None:
            raise HTTPException(status_code=400, detail="Please upload a table file first")
        
        result = table_agent.process_query(query)  # No more use_graphrag param
        
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        import traceback
        logger.error(f"Query processing failed: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 5: Remove `/api/set-method` endpoint from `app.py`**

Remove lines 261-276 (the entire `/api/set-method` route).

- [ ] **Step 6: Update `process_query` signature in `table_agent.py`**

Change the method signature:
```python
def process_query(self, query: str) -> Dict[str, Any]:
    """Process user query (auto-routes between GraphRAG and QA)"""
```

Remove the `use_graphrag` parameter documentation.

- [ ] **Step 7: Verify no remaining references**

Run: `grep -rn "use_graphrag\|useGR\|set.method\|setMethod\|graphrag_engine\|GraphRAG" --include="*.py" --include="*.html" --include="*.js" .`
Expected: Only legitimate references in `core/graphrag_engine.py`, `core/project_graph_builder.py`, and `table_agent.py` (for the auto-routing logic). No remaining toggle/button references.

- [ ] **Step 8: Verify syntax**

Run: `py -3.13 -m py_compile app.py`
Run: `py -3.13 -m py_compile table_agent.py`
Expected: No errors.

- [ ] **Step 9: Commit**

```bash
git add templates/index.html app.py table_agent.py
git commit -m "refactor: remove manual GraphRAG toggle, implement auto-routing"
```

---

### Task 6: End-to-end verification

**Files:**
- Test: Manual testing via the web UI

- [ ] **Step 1: Start the server**

Run: `py -3.13 app.py`
Expected: Server starts without errors, project graph builds successfully.

- [ ] **Step 2: Test single-entity → GraphRAG**

Query: "Tell me about CO2 diffusion in zeolites"
Expected: Response from GraphRAG showing CO2 diffusion stats across zeolites, topology effects, ion effects. Response mentions specific mean_logD values.

- [ ] **Step 3: Test single-entity zeolite → GraphRAG**

Query: "What are the diffusion properties of MFI zeolite?"
Expected: Response from GraphRAG showing guest molecules that diffuse through MFI, ranked by diffusion speed.

- [ ] **Step 4: Test multi-entity → Q&A pipeline**

Query: "Which zeolite is best for separating CO2 and CH4?"
Expected: Response from Q&A pipeline with ranking table, direct evidence with DOI, and Tier 2 predictions.

- [ ] **Step 5: Test Chinese pattern analysis → GraphRAG**

Query: "分析二氧化碳在不同分子筛中的扩散规律"
Expected: GraphRAG response with pattern analysis of CO2 diffusion.

- [ ] **Step 6: Test multi-entity zeolite+guest → Q&A**

Query: "How does CO2 diffuse in MFI at 300K?"
Expected: Q&A pipeline response with specific data from CSV.

- [ ] **Step 7: Verify no "method_used" field in response**

Check JSON response: should not contain `"method_used": "graphrag"` or `"method_used": "original"`. Instead, the auto-routing is transparent to the user.

- [ ] **Step 8: Commit final verification notes**

```bash
git add -A
git commit -m "test: end-to-end verification of GraphRAG auto-routing"
```

---

## Self-Review

### 1. Spec coverage
- ✅ Remove GraphRAG button → Task 5
- ✅ Auto-route single-entity → GraphRAG → Tasks 1, 4
- ✅ Auto-route multi-entity → Q&A → Tasks 1, 4
- ✅ Integrate project1_guest into graph → Task 2
- ✅ Integrate project2_zeolite into graph → Task 2
- ✅ Subgraph anchoring on guest or zeolite → Tasks 2, 3
- ✅ Pattern analysis calls GraphRAG → Tasks 3, 4

### 2. Placeholder scan
- No TBD/TODO/fill-in-later found
- All code blocks are complete
- All commands have expected output

### 3. Type consistency
- `_understand_query` returns `Dict[str, Any]` with `route` key → consumed by `map_query_to_columns` → consumed by `process_query`
- `ProjectGraphBuilder.get_subgraph(anchor_type, anchor_value)` returns `Dict[str, Any]` → consumed by `GraphRAGEngine.analyze_with_project_graph`
- All function signatures match across tasks
