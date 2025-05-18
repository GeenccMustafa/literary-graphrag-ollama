import networkx as nx
import ollama
import pickle
from networkx.algorithms.community import louvain_communities
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Set, Optional, Generator

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants ---
GRAPH_INPUT_PATH: str = os.getenv("GRAPH_PATH", "data/literary_graph.gpickle")
COMMUNITY_SUMMARIES_PATH: str = os.getenv(
    "COMMUNITY_SUMMARIES_PATH", "data/community_summaries.pkl"
)
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL", "llama3:8b") 
MIN_COMMUNITY_SIZE_FOR_SUMMARY: int = int(os.getenv("MIN_COMMUNITY_SIZE_FOR_SUMMARY", "2"))
LOUVAIN_SEED: int = 42


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


def _detect_communities(graph: nx.Graph) -> List[List[str]]:
    """
    Performs community detection on the graph using the Louvain algorithm.

    Args:
        graph: The NetworkX graph.

    Returns:
        A list of communities, where each community is a list of node names (strings).
        Returns an empty list if detection fails or the graph is unsuitable.
    """
    if graph.number_of_nodes() == 0:
        print("Graph is empty. Skipping community detection.")
        return []

    print("Performing community detection using Louvain algorithm...")
    try:
        communities_generator: Generator[Set[str], None, None] = louvain_communities(
            graph, weight="weight", seed=LOUVAIN_SEED
        )
        detected_communities: List[List[str]] = [
            sorted(list(c)) for c in communities_generator
        ] # Convert sets to sorted lists
        print(f"Detected {len(detected_communities)} communities (weighted).")
    except Exception as e:
        print(
            f"Error during weighted community detection: {e}. "
            "Attempting unweighted detection."
        )
        try:
            communities_generator = louvain_communities(graph, seed=LOUVAIN_SEED)
            detected_communities = [
                sorted(list(c)) for c in communities_generator
            ]
            print(f"Detected {len(detected_communities)} communities (unweighted).")
        except Exception as e2:
            print(f"ERROR: Unweighted community detection also failed: {e2}")
            return []
    
    return detected_communities


def _is_ollama_available() -> bool:
    """Checks if the Ollama service is available and reachable."""
    try:
        ollama.list()  
        print("Ollama service is available.")
        return True
    except Exception as e:
        print(
            f"ERROR: Could not connect to Ollama. Is the service running? Error: {e}"
        )
        return False


def _generate_community_summary_with_llm(
    community_nodes: List[str], llm_model: str
) -> str:
    """
    Generates a summary for a single community using an LLM.

    Args:
        community_nodes: A list of node names (characters) in the community.
        llm_model: The name of the LLM model to use via Ollama.

    Returns:
        A string summary of the community, or an error message if generation fails.
    """
    node_list_str = ", ".join(community_nodes)
    prompt = (
        "The following characters from a literary work form a distinct community or "
        f"closely related group: {node_list_str}. "
        "Based on common literary themes and typical character interactions in narratives, "
        "provide a concise one or two sentence summary that captures the potential essence "
        "or primary characteristic of this group. For example, you might describe them as "
        "'This group appears to be central to the main plot's conflict' or "
        "'These characters likely represent the protagonist's main allies and support system.' "
        "Avoid making definitive statements about the specific plot if you don't know it, "
        "focus on the archetypal role such a grouping might play."
    )

    try:
        response = ollama.generate(model=llm_model, prompt=prompt)
        summary = response["response"].strip()
        return summary
    except Exception as e:
        print(f"Error generating LLM summary for community [{node_list_str}]: {e}")
        return "Error: Could not generate summary for this community."


def _summarize_communities(
    communities: List[List[str]],
    llm_model: str,
    min_community_size: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Summarizes detected communities using an LLM, filtering out small communities.

    Args:
        communities: A list of communities (each a list of node names).
        llm_model: The name of the LLM model to use.
        min_community_size: The minimum number of nodes a community must have to be summarized.

    Returns:
        A dictionary where keys are community IDs (e.g., "Community_1") and
        values are dictionaries containing "nodes" and "summary".
    """
    if not communities:
        print("No communities to summarize.")
        return {}

    if not _is_ollama_available():
        print("Ollama not available. Skipping community summarization.")
        return {}

    print(f"Summarizing communities using LLM ({llm_model}). This may take time...")
    community_summaries: Dict[str, Dict[str, Any]] = {}
    valid_communities_count = 0

    for i, community_nodes in enumerate(communities):
        if len(community_nodes) < min_community_size:
            print(
                f"Skipping Community {i+1} (Nodes: {', '.join(community_nodes)}) - "
                f"size {len(community_nodes)} is less than minimum {min_community_size}."
            )
            continue
        
        valid_communities_count +=1
        community_id = f"Community_{valid_communities_count}"

        print(
            f"\n--- Summarizing {community_id}/{len(communities)} (Original Index {i+1}) ---"
        )
        print(f"Nodes: {', '.join(community_nodes)}")

        summary = _generate_community_summary_with_llm(community_nodes, llm_model)
        
        community_summaries[community_id] = {
            "nodes": community_nodes, 
            "summary": summary,
        }
        print(f"LLM Summary for {community_id}: {summary}")

    if not community_summaries:
        print("No communities met the size criteria for summarization.")
    return community_summaries


def _save_community_summaries(
    summaries: Dict[str, Dict[str, Any]], output_path: str
) -> None:
    """
    Saves the community summaries to a pickle file.

    Args:
        summaries: The dictionary of community summaries.
        output_path: The path to save the pickle file.
    """
    if not summaries:
        print("No summaries to save.")
        return

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as sf:
            pickle.dump(summaries, sf)
        print(f"\nCommunity summaries saved to {output_path}")
    except Exception as e:
        print(f"ERROR: Could not save community summaries to {output_path}. Error: {e}")


def main() -> None:
    """
    Main orchestrator for graph analysis and community summarization.
    Loads a graph, detects communities, and uses an LLM to summarize them.
    """
    print("--- Starting Graph Analysis and Community Summarization ---")

    graph = _load_graph(GRAPH_INPUT_PATH)
    if graph is None:
        print("Process aborted due to graph loading failure.")
        return

    communities = _detect_communities(graph)
    if not communities:
        print("No communities were detected or an error occurred. Skipping summarization.")
        print("--- Graph Analysis and Community Summarization Finished (No Summaries) ---")
        return
    
    community_summaries = _summarize_communities(
        communities, LLM_MODEL_NAME, MIN_COMMUNITY_SIZE_FOR_SUMMARY
    )

    if community_summaries:
        _save_community_summaries(community_summaries, COMMUNITY_SUMMARIES_PATH)
    else:
        print("No community summaries were generated.")

    print("--- Graph Analysis and Community Summarization Finished ---")


if __name__ == "__main__":
    main()