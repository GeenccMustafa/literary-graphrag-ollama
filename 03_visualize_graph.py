import networkx as nx
import pickle
import os
from dotenv import load_dotenv
from pyvis.network import Network
from typing import Optional, Dict, Any

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants ---
GRAPH_INPUT_PATH: str = os.getenv("GRAPH_PATH", "data/literary_graph.gpickle")
VISUALIZATION_OUTPUT_PATH: str = os.getenv(
    "VISUALIZATION_OUTPUT_PATH", "data/literary_graph_visualization.html"
)
PYVIS_HEIGHT: str = os.getenv("PYVIS_HEIGHT", "750px")
PYVIS_WIDTH: str = os.getenv("PYVIS_WIDTH", "100%")
PYVIS_CDN_RESOURCES: str = os.getenv("PYVIS_CDN_RESOURCES", "remote")
PYVIS_HEADING: str = os.getenv("PYVIS_HEADING", "Literary Knowledge Graph")


def _load_graph(file_path: str) -> Optional[nx.Graph]:
    """
    Loads a NetworkX graph from a Gpickle file.

    Args:
        file_path: The path to the Gpickle file.

    Returns:
        The loaded NetworkX graph, or None if loading fails or file not found.
    """
    print(f"Loading graph from {file_path}...")
    if not os.path.exists(file_path):
        print(
            f"ERROR: Graph file not found at {file_path}. "
            "Please run the knowledge graph building script first."
        )
        return None
    try:
        with open(file_path, "rb") as gf:
            graph = pickle.load(gf)
        print(
            f"Graph loaded successfully with {graph.number_of_nodes()} nodes "
            f"and {graph.number_of_edges()} edges."
        )
        return graph
    except Exception as e:
        print(f"ERROR: Could not load graph from {file_path}. Error: {e}")
        return None


def _create_pyvis_network(
    graph: nx.Graph, height: str, width: str, cdn_resources: str, heading: str
) -> Network:
    """
    Creates a Pyvis Network object from a NetworkX graph.

    Args:
        graph: The NetworkX graph.
        height: The height of the visualization canvas.
        width: The width of the visualization canvas.
        cdn_resources: How to load JS/CSS ('remote' or 'local').
        heading: The main heading for the visualization.

    Returns:
        A Pyvis Network object populated with nodes and edges.
    """
    print("Creating Pyvis network visualization...")
    net = Network(
        notebook=False,  # False for standalone HTML
        height=height,
        width=width,
        cdn_resources=cdn_resources,
        heading=heading,
    )

    for node, attrs in graph.nodes(data=True):
        node_title = f"Character: {node}<br>Type: {attrs.get('type', 'N/A')}"
        degree = graph.degree(node)
        node_size = 10 + degree * 2
        
        community_id = attrs.get('community_id', 1)
        if isinstance(community_id, str) and community_id.startswith("Community_"):
            try:
                community_group = int(community_id.split("_")[-1])
            except ValueError:
                community_group = 1 
        elif isinstance(community_id, int):
            community_group = community_id
        else:
            community_group = 1 

        net.add_node(
            node,
            label=node,
            title=node_title,
            size=node_size,
            group=community_group
        )

    for source, target, attrs in graph.edges(data=True):
        edge_title = (
            f"Relationship: {attrs.get('type', 'related')}<br>"
            f"Weight: {attrs.get('weight', 1)}"
        )
        edge_width = attrs.get("weight", 1)
        net.add_edge(source, target, title=edge_title, value=edge_width)

    options = """
    var options = {
      "nodes": {
        "font": {
          "size": 12,
          "face": "Tahoma"
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        },
        "arrows": {
          "to": { "enabled": false }
        } 
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "navigationButtons": true,
        "keyboard": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "solver": "barnesHut",
        "minVelocity": 0.75,
        "stabilization": {
          "iterations": 1000
        }
      }
    }
    """
    net.set_options(options)

    return net


def _save_pyvis_network(pyvis_net: Network, output_path: str) -> None:
    """
    Saves the Pyvis network visualization to an HTML file.

    Args:
        pyvis_net: The Pyvis Network object.
        output_path: The path to save the HTML file.
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: 
            os.makedirs(output_dir, exist_ok=True)
        pyvis_net.save_graph(output_path)
        print(f"Graph visualization saved to: {output_path}")
        print("You can open this HTML file in your web browser.")
    except Exception as e:
        print(f"ERROR: Could not save graph visualization to {output_path}. Error: {e}")


def main_visualize_graph() -> None:
    """
    Main function to load a graph and generate an interactive HTML visualization.
    """
    print("--- Starting Graph Visualization ---")

    graph = _load_graph(GRAPH_INPUT_PATH)
    if graph is None:
        print("Process aborted due to graph loading failure.")
        return

    if graph.number_of_nodes() == 0:
        print("Graph is empty. Nothing to visualize.")
        print("--- Graph Visualization Finished (No Graph) ---")
        return

    pyvis_network = _create_pyvis_network(
        graph, PYVIS_HEIGHT, PYVIS_WIDTH, PYVIS_CDN_RESOURCES, PYVIS_HEADING
    )

    _save_pyvis_network(pyvis_network, VISUALIZATION_OUTPUT_PATH)

    print("--- Graph Visualization Finished ---")


if __name__ == "__main__":
    main_visualize_graph()