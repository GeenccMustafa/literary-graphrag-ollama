import networkx as nx
import pickle
import os
from dotenv import load_dotenv
from pyvis.network import Network
from typing import Optional, Dict, Any, List
import html

load_dotenv()

GRAPH_INPUT_PATH: str = os.getenv("GRAPH_PATH", "data/literary_graph.gpickle")
COMMUNITY_SUMMARIES_PATH: str = os.getenv("COMMUNITY_SUMMARIES_PATH", "data/community_summaries.pkl")
VISUALIZATION_OUTPUT_PATH: str = os.getenv("VISUALIZATION_OUTPUT_PATH", "data/literary_graph_visualization.html")
PYVIS_HEIGHT: str = os.getenv("PYVIS_HEIGHT", "800px")
PYVIS_WIDTH: str = os.getenv("PYVIS_WIDTH", "100%")
PYVIS_CDN_RESOURCES: str = os.getenv("PYVIS_CDN_RESOURCES", "remote").lower()
PYVIS_HEADING: str = os.getenv("PYVIS_HEADING", "Literary Character Network")

if PYVIS_CDN_RESOURCES not in ["local", "in_line", "remote"]:
    print(f"Warning: Invalid PYVIS_CDN_RESOURCES. Defaulting to 'remote'.")
    PYVIS_CDN_RESOURCES = "remote"

COLOR_PALETTE = [
    "#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6",
    "#dd4477", "#66aa00", "#b82e2e", "#316395", "#994499", "#22aa99",
    "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#329262", "#5574a6",
    "#3b3eac"
]

def _load_graph(file_path: str) -> Optional[nx.Graph]:
    if not os.path.exists(file_path):
        print(f"ERROR: Graph file not found at {file_path}.")
        return None
    try:
        with open(file_path, "rb") as gf: graph = pickle.load(gf)
        print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        return graph
    except Exception as e: print(f"ERROR loading graph: {e}"); return None

def _load_community_data(file_path: str) -> Dict[str, str]:
    if not os.path.exists(file_path):
        print(f"Warning: Community summaries not found: {file_path}.")
        return {}
    try:
        with open(file_path, "rb") as sf: summaries = pickle.load(sf)
        node_to_community = {node_name: comm_id for comm_id, data in summaries.items() for node_name in data.get("nodes", [])}
        print(f"Loaded {len(summaries)} communities, mapping {len(node_to_community)} nodes.")
        return node_to_community
    except Exception as e: print(f"Error loading community data: {e}"); return {}

def _create_pyvis_network(
    graph: nx.Graph, node_to_community: Dict[str, str], height: str, width: str,
    cdn_resources: str, heading_text: str
) -> Network:
    print("Creating Pyvis network visualization...")
    net = Network(notebook=False, height=height, width=width, cdn_resources=cdn_resources,
                  heading=f"<h1>{html.escape(heading_text)}</h1>", directed=nx.is_directed(graph),
                  select_menu=True, filter_menu=True)

    community_id_to_int_map: Dict[str, int] = {}
    next_community_int_id = 1

    for node_str, attrs in graph.nodes(data=True):
        node_id_for_pyvis = str(node_str)
        node_label_display = str(node_str)
        
        # --- PLAIN TEXT Node Tooltip Construction ---
        plain_tooltip_parts: List[str] = [f"Name: {html.escape(node_label_display)}"] # Start with name

        community_id_str = node_to_community.get(node_str)
        group_val: int = 0
        node_color: Optional[str] = None
        if community_id_str:
            if community_id_str not in community_id_to_int_map:
                community_id_to_int_map[community_id_str] = next_community_int_id
                next_community_int_id += 1
            group_val = community_id_to_int_map[community_id_str]
            node_color = COLOR_PALETTE[(group_val - 1) % len(COLOR_PALETTE)]
            plain_tooltip_parts.append(f"Community: {html.escape(community_id_str)}")

        node_type_attr = attrs.get('type', 'N/A')
        plain_tooltip_parts.append(f"Type: {html.escape(node_type_attr)}")

        skip_attrs = {'type', 'label', 'id', 'community_id', 'size', 'color', 'group', 'x', 'y', 'fx', 'fy', 'title', 'value', 'font'}
        for key, value in attrs.items():
            if key not in skip_attrs:
                plain_tooltip_parts.append(f"{html.escape(key.replace('_', ' ').title())}: {html.escape(str(value))}")
        
        node_tooltip_text = "\n".join(plain_tooltip_parts)
        # --- End PLAIN TEXT Node Tooltip Construction ---

        degree = graph.degree(node_id_for_pyvis)
        node_size = 10 + degree * 2

        net.add_node(node_id_for_pyvis, label=html.escape(node_label_display), title=node_tooltip_text,
                     size=node_size, group=group_val, color=node_color)

    for source_str, target_str, attrs_edge in graph.edges(data=True):
        source_id = str(source_str)
        target_id = str(target_str)
        edge_weight = attrs_edge.get('weight', 1.0)
        
        # --- PLAIN TEXT Edge Tooltip Construction ---
        plain_tooltip_parts: List[str] = [f"From: {html.escape(source_id)}", f"To: {html.escape(target_id)}"]
        rel_type = attrs_edge.get('type', 'related')
        plain_tooltip_parts.append(f"Relationship: {html.escape(rel_type)}")
        plain_tooltip_parts.append(f"Weight: {edge_weight:.2f}")

        for key, value in attrs_edge.items():
            if key not in skip_attrs: # Use same skip_attrs for edges for simplicity
                plain_tooltip_parts.append(f"{html.escape(key.replace('_', ' ').title())}: {html.escape(str(value))}")
        edge_tooltip_text = "\n".join(plain_tooltip_parts)
        # --- End PLAIN TEXT Edge Tooltip Construction ---

        net.add_edge(source_id, target_id, title=edge_tooltip_text, value=edge_weight,
                     width=max(0.5, edge_weight / 3.0 if edge_weight > 0 else 0.5), color="#cccccc")

    options_str = """
    {
      "nodes": {"borderWidth": 2, "borderWidthSelected": 4, "font": { "size": 14, "face": "Arial", "color": "#333333" }, "shadow": { "enabled": true, "size": 8, "x": 4, "y": 4, "color": "rgba(0,0,0,0.2)"}, "shapeProperties": { "interpolation": false }},
      "edges": {"smooth": { "enabled": true, "type": "dynamic", "roundness": 0.5 }, "arrows": { "to": { "enabled": false, "scaleFactor": 0.6 }}, "color": { "inherit": false, "highlight": "#ff0000", "hover": "#ff4500" }, "widthConstraint": { "maximum": 20 }, "hoverWidth": 1.5 },
      "interaction": {"hover": true, "tooltipDelay": 250, "navigationButtons": true, "keyboard": { "enabled": true }, "hideEdgesOnDrag": false, "hideNodesOnDrag": false, "multiselect": true, "hoverConnectedEdges": true},
      "layout": { "improvedLayout": true, "hierarchical": false },
      "physics": {"enabled": true, "barnesHut": {"gravitationalConstant": -20000, "centralGravity": 0.1, "springLength": 150, "springConstant": 0.02, "damping": 0.15, "avoidOverlap": 0.3}, "forceAtlas2Based": {"gravitationalConstant": -100, "centralGravity": 0.01, "springLength": 150, "springConstant": 0.05, "damping": 0.4, "avoidOverlap": 0.5}, "solver": "barnesHut", "minVelocity": 0.5, "maxVelocity": 40, "stabilization": {"enabled": true, "iterations": 2000, "updateInterval": 50, "onlyDynamicEdges": false, "fit": true}, "adaptiveTimestep": true},
      "manipulation": { "enabled": false }
    }"""
    net.set_options(options_str)
    return net

def _save_pyvis_network(pyvis_net: Network, output_path: str) -> None:
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        pyvis_net.save_graph(output_path)
        print(f"Graph visualization saved: {os.path.abspath(output_path)}")
    except Exception as e: print(f"ERROR saving graph: {e}"); import traceback; traceback.print_exc()

def main_visualize_graph() -> None:
    print("--- Starting Graph Visualization (Plain Text Tooltips Version) ---")
    graph = _load_graph(GRAPH_INPUT_PATH)
    if not graph or graph.number_of_nodes() == 0:
        print("Graph empty or not loaded. Aborting.")
        return
    node_to_community_map = _load_community_data(COMMUNITY_SUMMARIES_PATH)
    pyvis_network = _create_pyvis_network(graph, node_to_community_map, PYVIS_HEIGHT, PYVIS_WIDTH, PYVIS_CDN_RESOURCES, PYVIS_HEADING)
    _save_pyvis_network(pyvis_network, VISUALIZATION_OUTPUT_PATH)
    print("--- Graph Visualization Finished ---")

if __name__ == "__main__":
    main_visualize_graph()