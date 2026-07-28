"""
Project Graph Builder - Build knowledge graph from project1_guest and project2_zeolite data.

Graph Structure:
  Nodes:
    - Guest: guest molecule (e.g., CO2, CH4, H2O)
    - Zeolite: specific zeolite (e.g., MFI, H-ZSM-5, DD3R)
    - Topology: framework topology (e.g., MFI, FAU, LTA, DDR)
    - Ion: exchange ion (e.g., H+, Na+, K+, Cu+)

  Edges:
    - Guest --[diffuses_in]--> Zeolite : {mean_logD, median_logD, std_logD, count, D_range_orders}
    - Guest --[diffuses_in]--> Topology : {mean_logD, median_logD, std_logD, count}
    - Guest --[affected_by]--> Ion : {mean_logD, count}
    - Zeolite --[has_topology]--> Topology : {}
    - Zeolite --[hosts]--> Guest : {mean_logD, median_logD, std_logD, count, kinetic_diameter_A, Ea, R2}
"""
import json
import os
import base64
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from loguru import logger
import networkx as nx


class ProjectGraphBuilder:
    """Build and query knowledge graph from project1_guest and project2_zeolite statistics."""

    def __init__(self, project1_dir: str = None, project2_dir: str = None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project1_stats_dir = project1_dir or os.path.join(base, "project1_guest", "output", "stats")
        self.project1_figures_dir = os.path.join(base, "project1_guest", "output", "figures")
        self.project1_tables_dir = os.path.join(base, "project1_guest", "output", "tables")
        self.project2_stats_dir = project2_dir or os.path.join(base, "project2_zeolite", "output", "stats")
        self.project2_figures_dir = os.path.join(base, "project2_zeolite", "output", "figures")
        self.project2_tables_dir = os.path.join(base, "project2_zeolite", "output", "tables")
        self.graph = nx.DiGraph()
        self._guest_index = defaultdict(list)    # guest_name_lower → [node_ids]
        self._zeolite_index = defaultdict(list)   # zeolite_name_lower → [node_ids]
        self._topology_index = defaultdict(list)  # topology_lower → [node_ids]
        self._guest_figures = {}   # guest_name_lower → {"zeolites": path, "arrhenius": path, "si_al": path}
        self._zeolite_figures = {} # zeolite_name_lower → {"guests": path, "size_vs_D": path, "arrhenius": path}
        self._built = False

    def build(self) -> nx.DiGraph:
        """Build the complete knowledge graph. Idempotent — clears and rebuilds."""
        logger.info("Building project knowledge graph from project1 + project2 data...")
        self.graph.clear()
        self._guest_index.clear()
        self._zeolite_index.clear()
        self._topology_index.clear()
        self._guest_figures.clear()
        self._zeolite_figures.clear()

        self._index_figures()
        self._load_project1_guest()
        self._load_project2_zeolite()
        self._merge_similar_guest_nodes()

        self._built = True
        logger.info(f"Graph built: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        logger.info(f"  Guests: {len(self._guest_index)}, Zeolites: {len(self._zeolite_index)}, "
                    f"Topologies: {len(self._topology_index)}")
        logger.info(f"  Figures: {len(self._guest_figures)} guests, {len(self._zeolite_figures)} zeolites")
        return self.graph

    def _index_figures(self):
        """Index available figures from project1 and project2 output directories."""
        # Project1 guest figures: {guest}_zeolites.png, {guest}_arrhenius.png, {guest}_si_al.png
        if os.path.isdir(self.project1_figures_dir):
            for fname in os.listdir(self.project1_figures_dir):
                if not fname.endswith(".png"):
                    continue
                fpath = os.path.join(self.project1_figures_dir, fname)
                name_no_ext = fname[:-4]  # Remove .png
                # Parse: {guest}_{chart_type}
                for suffix in ["_zeolites", "_arrhenius", "_si_al"]:
                    if name_no_ext.endswith(suffix):
                        guest_name = name_no_ext[:-len(suffix)]
                        guest_key = guest_name.lower().strip()
                        if guest_key not in self._guest_figures:
                            self._guest_figures[guest_key] = {}
                        chart_key = suffix.lstrip("_")  # zeolites, arrhenius, si_al
                        self._guest_figures[guest_key][chart_key] = fpath
                        break

        # Project2 zeolite figures: {zeolite}_guests.png, {zeolite}_size_vs_D.png, {zeolite}_arrhenius.png
        if os.path.isdir(self.project2_figures_dir):
            for fname in os.listdir(self.project2_figures_dir):
                if not fname.endswith(".png"):
                    continue
                fpath = os.path.join(self.project2_figures_dir, fname)
                name_no_ext = fname[:-4]
                for suffix in ["_guests", "_size_vs_D", "_arrhenius"]:
                    if name_no_ext.endswith(suffix):
                        zeolite_name = name_no_ext[:-len(suffix)]
                        zeolite_key = zeolite_name.lower().strip()
                        if zeolite_key not in self._zeolite_figures:
                            self._zeolite_figures[zeolite_key] = {}
                        chart_key = suffix.lstrip("_")  # guests, size_vs_D, arrhenius
                        self._zeolite_figures[zeolite_key][chart_key] = fpath
                        break

    def _load_project1_guest(self):
        """Load project1 data: guest-centric view.
        For each guest molecule, add edges to zeolites, topologies, and ions.
        """
        if not os.path.isdir(self.project1_stats_dir):
            logger.warning(f"project1_guest stats dir not found: {self.project1_stats_dir}")
            return

        for fname in sorted(os.listdir(self.project1_stats_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self.project1_stats_dir, fname)
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
                z_name = z_entry.get("zeolite_name", "")
                if not z_name:
                    continue
                z_node = f"zeolite:{z_name}"
                if z_node not in self.graph.nodes:
                    self.graph.add_node(z_node, type="zeolite", name=z_name)
                self._zeolite_index[z_name.lower()].append(z_node)
                self.graph.add_edge(guest_node, z_node, relation="diffuses_in",
                    mean_logD=z_entry.get("mean_logD"),
                    median_logD=z_entry.get("median_logD"),
                    std_logD=z_entry.get("std_logD"),
                    count=z_entry.get("count"),
                    D_range_orders=z_entry.get("D_range_orders"))

            # Guest → Topology edges
            for t_entry in data.get("by_topology", []):
                t_name = t_entry.get("topology", t_entry.get("拓扑", ""))
                if not t_name or t_name == "Other":
                    continue
                t_node = f"topology:{t_name}"
                if t_node not in self.graph.nodes:
                    self.graph.add_node(t_node, type="topology", name=t_name)
                self._topology_index[t_name.lower()].append(t_node)
                self.graph.add_edge(guest_node, t_node, relation="diffuses_in_topology",
                    mean_logD=t_entry.get("mean_logD"),
                    median_logD=t_entry.get("median_logD"),
                    std_logD=t_entry.get("std_logD"),
                    count=t_entry.get("n", t_entry.get("count")))

            # Guest → Ion edges
            for i_entry in data.get("by_ion", []):
                i_name = i_entry.get("ion", i_entry.get("离子", ""))
                if not i_name or i_name == "(none)":
                    continue
                i_node = f"ion:{i_name}"
                if i_node not in self.graph.nodes:
                    self.graph.add_node(i_node, type="ion", name=i_name)
                self.graph.add_edge(guest_node, i_node, relation="affected_by_ion",
                    mean_logD=i_entry.get("mean_logD"),
                    count=i_entry.get("n", i_entry.get("count")))

            # Arrhenius per-zeolite for this guest
            for a_entry in data.get("arrhenius", []):
                z_name = a_entry.get("zeolite_name", a_entry.get("分子筛", ""))
                if z_name:
                    z_node = f"zeolite:{z_name}"
                    if z_node in self.graph.nodes and self.graph.has_edge(guest_node, z_node):
                        self.graph.edges[guest_node, z_node]["Ea_kJ_mol"] = a_entry.get("Ea_kJ_mol")
                        self.graph.edges[guest_node, z_node]["Ea_R2"] = a_entry.get("R2")
                        self.graph.edges[guest_node, z_node]["Ea_n"] = a_entry.get("n")

            # Store overall guest Ea if available
            if "Ea_kJ_mol" in data:
                self.graph.nodes[guest_node]["Ea_kJ_mol"] = data["Ea_kJ_mol"]
                self.graph.nodes[guest_node]["Ea_R2"] = data.get("Ea_R2")

        logger.info(f"Loaded project1_guest: {len(self._guest_index)} guests")

    def _load_project2_zeolite(self):
        """Load project2 data: zeolite-centric view.
        For each zeolite, add edges to guest molecules with kinetic diameters.
        """
        if not os.path.isdir(self.project2_stats_dir):
            logger.warning(f"project2_zeolite stats dir not found: {self.project2_stats_dir}")
            return

        for fname in sorted(os.listdir(self.project2_stats_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self.project2_stats_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")
                continue

            z_name = data.get("zeolite_name", fname.replace(".json", ""))
            z_node = f"zeolite:{z_name}"
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
                self._topology_index[topology.lower()].append(t_node)

            self.graph.nodes[z_node]["n_total"] = data.get("n_total", 0)

            # Zeolite → Guest edges (with kinetic diameter)
            for g_entry in data.get("by_guest", []):
                g_name = g_entry.get("guest_molecule", "")
                if not g_name:
                    continue
                g_node = f"guest:{g_name}"
                if g_node not in self.graph.nodes:
                    self.graph.add_node(g_node, type="guest", name=g_name)
                self._guest_index[g_name.lower()].append(g_node)

                kd = g_entry.get("kinetic_diameter_A")
                self.graph.add_edge(z_node, g_node, relation="hosts",
                    mean_logD=g_entry.get("mean_logD"),
                    median_logD=g_entry.get("median_logD"),
                    std_logD=g_entry.get("std_logD"),
                    count=g_entry.get("count"),
                    D_range_orders=g_entry.get("D_range_orders"),
                    kinetic_diameter_A=kd)

                if kd is not None:
                    self.graph.nodes[g_node]["kinetic_diameter_A"] = kd

            # Arrhenius per guest
            for a_entry in data.get("arrhenius", []):
                g_name = a_entry.get("guest_molecule", a_entry.get("客体分子", ""))
                if g_name:
                    g_node = f"guest:{g_name}"
                    if g_node in self.graph.nodes and self.graph.has_edge(z_node, g_node):
                        self.graph.edges[z_node, g_node]["Ea_kJ_mol"] = a_entry.get("Ea_kJ_mol")
                        self.graph.edges[z_node, g_node]["Ea_R2"] = a_entry.get("R2")
                        self.graph.edges[z_node, g_node]["Ea_n"] = a_entry.get("n")

        logger.info(f"Loaded project2_zeolite: {len(self._zeolite_index)} zeolites")

    def _merge_similar_guest_nodes(self):
        """Merge guest nodes that represent the same molecule with different name formats.

        Examples: 'lead' + 'lead(2+)' → 'lead', 'CO2' + 'carbon dioxide' → 'CO2'
        Only merges when one name is clearly a sub-form of the other (charge suffix, etc.)
        """
        import re
        # Normalize: strip charge suffixes like (2+), (II), (3+), etc.
        def normalize(name: str) -> str:
            return re.sub(r'\s*\([IV\d+]+\)\s*$', '', name.strip()).strip().lower()

        # Group guests by normalized name
        groups = defaultdict(list)
        for guest_key in list(self._guest_index.keys()):
            norm = normalize(guest_key)
            groups[norm].append(guest_key)

        merged = 0
        for norm, names in groups.items():
            if len(names) <= 1:
                continue
            # Merge: keep the shortest name as canonical, move edges from others
            canonical = min(names, key=len)
            canonical_node = f"guest:{canonical}"
            for other in names:
                if other == canonical:
                    continue
                other_node = f"guest:{other}"
                if other_node not in self.graph.nodes:
                    continue
                # Move all edges from other_node to canonical_node
                for src, tgt, data in list(self.graph.in_edges(other_node, data=True)):
                    new_tgt = canonical_node if tgt == other_node else tgt
                    new_src = canonical_node if src == other_node else src
                    if not self.graph.has_edge(new_src, new_tgt):
                        self.graph.add_edge(new_src, new_tgt, **data)
                for src, tgt, data in list(self.graph.out_edges(other_node, data=True)):
                    new_tgt = canonical_node if tgt == other_node else tgt
                    new_src = canonical_node if src == other_node else src
                    if not self.graph.has_edge(new_src, new_tgt):
                        self.graph.add_edge(new_src, new_tgt, **data)
                # Update node properties on canonical
                other_data = dict(self.graph.nodes[other_node])
                for k, v in other_data.items():
                    if k not in ('type', 'name') and v is not None:
                        if k not in self.graph.nodes[canonical_node] or self.graph.nodes[canonical_node][k] is None:
                            self.graph.nodes[canonical_node][k] = v
                # Remove old node
                self.graph.remove_node(other_node)
                # Update index
                if other in self._guest_index:
                    del self._guest_index[other]
                merged += 1
                logger.info(f"  Merged guest '{other}' → '{canonical}'")

        if merged:
            logger.info(f"Merged {merged} duplicate guest nodes")

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
                "nodes": [...], "edges": [...],
                "summary": str,
                "figures": [{"chart_type": str, "base64": str}],
                "tables": [{"name": str, "csv_text": str}]
            }
        """
        if not self._built:
            self.build()

        anchor_node = self._find_anchor_node(anchor_type, anchor_value)
        if not anchor_node:
            return {"anchor": None, "nodes": [], "edges": [], "summary": "",
                    "figures": [], "tables": [],
                    "error": f"No {anchor_type} matching '{anchor_value}' found"}

        # BFS to extract subgraph
        visited = set()
        subgraph_nodes = []
        subgraph_edges = []

        queue = deque([(anchor_node, 0)])
        visited.add(anchor_node)

        while queue:
            current, depth = queue.popleft()
            node_data = dict(self.graph.nodes[current])
            subgraph_nodes.append({
                "id": current,
                "type": node_data.get("type", "unknown"),
                "name": node_data.get("name", current),
                "properties": {k: v for k, v in node_data.items()
                              if k not in ("type", "name")}
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

        summary = self._summarize_subgraph(anchor_type, anchor_value, subgraph_nodes, subgraph_edges)
        figures = self._get_figures_for_anchor(anchor_type, anchor_value)
        tables = self._get_tables_for_anchor(anchor_type, anchor_value)

        return {
            "anchor": {"type": anchor_type, "name": anchor_value, "node_id": anchor_node},
            "nodes": subgraph_nodes,
            "edges": subgraph_edges,
            "summary": summary,
            "figures": figures,
            "tables": tables,
        }

    def _find_anchor_node(self, anchor_type: str, anchor_value: str) -> Optional[str]:
        """Find the best-matching node for an anchor value."""
        anchor_lower = anchor_value.lower().strip()

        if anchor_type == "guest":
            candidates = self._guest_index.get(anchor_lower, [])
        elif anchor_type == "zeolite":
            candidates = self._zeolite_index.get(anchor_lower, [])
        else:
            return None

        # Exact match: prefer the canonical name
        for node_id in candidates:
            node_data = self.graph.nodes.get(node_id, {})
            if node_data.get("name", "").lower() == anchor_lower:
                return node_id

        if candidates:
            return candidates[0]

        # Fuzzy match across all nodes of this type
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == anchor_type:
                name = data.get("name", "").lower()
                if anchor_lower in name or name in anchor_lower:
                    return node_id

        return None

    def _summarize_subgraph(self, anchor_type: str, anchor_value: str,
                            nodes: List[Dict], edges: List[Dict]) -> str:
        """Generate a text summary of the subgraph for LLM context."""
        lines = [f"=== Knowledge Graph: {anchor_type} '{anchor_value}' ===\n"]

        by_relation = defaultdict(list)
        for edge in edges:
            by_relation[edge["relation"]].append(edge)

        if anchor_type == "guest":
            lines.append(f"## Diffusion across zeolites for {anchor_value}\n")
            lines.append(f"| Zeolite | mean_logD | D_range(orders) | n | Ea(kJ/mol) |")
            lines.append(f"|---------|-----------|-----------------|---|------------|")
            # Collect edges from both directions:
            #   "diffuses_in": guest -> zeolite (zeolite = target)
            #   "hosts":       zeolite -> guest (zeolite = source)
            diff_edges = by_relation.get("diffuses_in", []) + by_relation.get("hosts", [])
            seen = set()
            zeolite_stats = []
            for e in diff_edges:
                # Determine which side is the zeolite
                is_source_guest = e["source"].startswith("guest:")
                is_target_guest = e["target"].startswith("guest:")
                if is_source_guest and not is_target_guest:
                    z_name = e["target"].replace("zeolite:", "")
                elif is_target_guest and not is_source_guest:
                    z_name = e["source"].replace("zeolite:", "")
                else:
                    continue  # Skip guest-guest or zeolite-zeolite edges
                if z_name in seen:
                    continue
                seen.add(z_name)
                props = e["properties"]
                zeolite_stats.append({
                    "zeolite": z_name,
                    "mean_logD": props.get("mean_logD"),
                    "D_range": props.get("D_range_orders"),
                    "count": props.get("count"),
                    "Ea": props.get("Ea_kJ_mol"),
                })
            zeolite_stats.sort(key=lambda x: (x["mean_logD"] is not None, x["mean_logD"] or -999), reverse=True)
            for zs in zeolite_stats[:20]:
                ea_str = f"{zs['Ea']:.1f}" if zs.get("Ea") is not None else "N/A"
                mean_str = f"{zs['mean_logD']:.2f}" if zs.get("mean_logD") is not None else "N/A"
                range_str = f"{zs['D_range']:.1f}" if zs.get("D_range") is not None else "N/A"
                lines.append(f"| {zs['zeolite']} | {mean_str} | {range_str} | {zs['count']} | {ea_str} |")

            topo_edges = by_relation.get("diffuses_in_topology", [])
            if topo_edges:
                lines.append(f"\n## Topology influence\n")
                lines.append(f"| Topology | mean_logD | n |")
                lines.append(f"|----------|-----------|----|")
                for e in topo_edges[:10]:
                    t_name = e["target"].replace("topology:", "")
                    props = e["properties"]
                    mean_str = f"{props.get('mean_logD', 'N/A'):.2f}" if isinstance(props.get('mean_logD'), (int, float)) else str(props.get('mean_logD', 'N/A'))
                    lines.append(f"| {t_name} | {mean_str} | {props.get('count', 'N/A')} |")

            ion_edges = by_relation.get("affected_by_ion", [])
            if ion_edges:
                lines.append(f"\n## Ion effects\n")
                lines.append(f"| Ion | mean_logD | n |")
                lines.append(f"|-----|-----------|----|")
                for e in ion_edges[:10]:
                    i_name = e["target"].replace("ion:", "")
                    props = e["properties"]
                    mean_str = f"{props.get('mean_logD', 'N/A'):.2f}" if isinstance(props.get('mean_logD'), (int, float)) else str(props.get('mean_logD', 'N/A'))
                    lines.append(f"| {i_name} | {mean_str} | {props.get('count', 'N/A')} |")

        elif anchor_type == "zeolite":
            lines.append(f"## Guest molecules diffusing through {anchor_value}\n")
            topo_edges = by_relation.get("has_topology", [])
            if topo_edges:
                t_name = topo_edges[0]["target"].replace("topology:", "")
                lines.append(f"Topology: {t_name}\n")

            lines.append(f"| Guest | mean_logD | D_range(orders) | n | d_kin(Å) | Ea(kJ/mol) |")
            lines.append(f"|-------|-----------|-----------------|---|----------|------------|")
            guest_edges = by_relation.get("hosts", []) + by_relation.get("diffuses_in", [])
            seen = set()
            guest_stats = []
            for e in guest_edges:
                # Determine which side is the guest
                is_source_guest = e["source"].startswith("guest:")
                is_target_guest = e["target"].startswith("guest:")
                guest_node = e["source"] if is_source_guest else (e["target"] if is_target_guest else None)
                if not guest_node:
                    continue
                if guest_node in seen:
                    continue
                seen.add(guest_node)
                props = e["properties"]
                guest_stats.append({
                    "guest": guest_node.replace("guest:", ""),
                    "mean_logD": props.get("mean_logD"),
                    "D_range": props.get("D_range_orders"),
                    "count": props.get("count"),
                    "kd_A": props.get("kinetic_diameter_A"),
                    "Ea": props.get("Ea_kJ_mol"),
                })
            guest_stats.sort(key=lambda x: (x["mean_logD"] is not None, x["mean_logD"] or -999), reverse=True)
            for gs in guest_stats[:20]:
                kd_str = f"{gs['kd_A']:.1f}" if gs.get("kd_A") is not None else "N/A"
                ea_str = f"{gs['Ea']:.1f}" if gs.get("Ea") is not None else "N/A"
                mean_str = f"{gs['mean_logD']:.2f}" if gs.get("mean_logD") is not None else "N/A"
                range_str = f"{gs['D_range']:.1f}" if gs.get("D_range") is not None else "N/A"
                lines.append(f"| {gs['guest']} | {mean_str} | {range_str} | {gs['count']} | {kd_str} | {ea_str} |")

        lines.append(f"\nTotal: {len(nodes)} nodes, {len(edges)} edges in subgraph")
        return "\n".join(lines)

    def _get_figures_for_anchor(self, anchor_type: str, anchor_value: str) -> List[Dict[str, str]]:
        """Find and base64-encode figures for the anchor entity."""
        figures = []
        anchor_lower = anchor_value.lower().strip()

        if anchor_type == "guest":
            fig_dict = self._guest_figures.get(anchor_lower, {})
            # Also try fuzzy match
            if not fig_dict:
                for key, val in self._guest_figures.items():
                    if anchor_lower in key or key in anchor_lower:
                        fig_dict = val
                        break
            for chart_type in ["zeolites", "arrhenius", "si_al"]:
                if chart_type in fig_dict:
                    b64 = self._encode_image(fig_dict[chart_type])
                    if b64:
                        figures.append({"chart_type": chart_type, "base64": b64, "format": "png"})

        elif anchor_type == "zeolite":
            fig_dict = self._zeolite_figures.get(anchor_lower, {})
            if not fig_dict:
                for key, val in self._zeolite_figures.items():
                    if anchor_lower in key or key in anchor_lower:
                        fig_dict = val
                        break
            for chart_type in ["guests", "size_vs_D", "arrhenius"]:
                if chart_type in fig_dict:
                    b64 = self._encode_image(fig_dict[chart_type])
                    if b64:
                        figures.append({"chart_type": chart_type, "base64": b64, "format": "png"})

        return figures

    def _get_tables_for_anchor(self, anchor_type: str, anchor_value: str) -> List[Dict[str, str]]:
        """Find CSV tables for the anchor entity."""
        tables = []
        anchor_lower = anchor_value.lower().strip()

        if anchor_type == "guest":
            table_dir = self.project1_tables_dir
            if os.path.isdir(table_dir):
                for fname in sorted(os.listdir(table_dir)):
                    if not fname.endswith(".csv"):
                        continue
                    # File pattern: {guest}_by_{dimension}.csv
                    fname_lower = fname.lower()
                    # Match: starts with guest name
                    if fname_lower.startswith(anchor_lower + "_") or fname_lower.startswith(anchor_lower.replace(" ", "_") + "_"):
                        fpath = os.path.join(table_dir, fname)
                        try:
                            with open(fpath, "r", encoding="utf-8") as f:
                                csv_text = f.read()
                            tables.append({"name": fname, "csv_text": csv_text[:5000]})  # Limit to 5KB
                        except Exception:
                            pass

        elif anchor_type == "zeolite":
            table_dir = self.project2_tables_dir
            if os.path.isdir(table_dir):
                for fname in sorted(os.listdir(table_dir)):
                    if not fname.endswith(".csv"):
                        continue
                    fname_lower = fname.lower()
                    if fname_lower.startswith(anchor_lower + "_") or fname_lower.startswith(anchor_lower.replace(" ", "_") + "_"):
                        fpath = os.path.join(table_dir, fname)
                        try:
                            with open(fpath, "r", encoding="utf-8") as f:
                                csv_text = f.read()
                            tables.append({"name": fname, "csv_text": csv_text[:5000]})
                        except Exception:
                            pass

        return tables

    def _encode_image(self, path: str) -> Optional[str]:
        """Base64-encode an image file."""
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            logger.warning(f"Failed to encode image {path}: {e}")
            return None

    def get_available_entities(self) -> Dict[str, Any]:
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
