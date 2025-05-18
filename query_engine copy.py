# graphrag_literary/query_engine.py
import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from qdrant_client.http.exceptions import UnexpectedResponse
import ollama
import pickle
import os
from dotenv import load_dotenv
from typing import List, Set, Optional, Dict, Any, Tuple

# Load environment variables from .env file
# It's good practice for each script/module that directly uses .env vars to load it.
load_dotenv()

# --- Configuration Constants ---
GRAPH_PATH_ENV: str = os.getenv("GRAPH_PATH", "data/literary_graph.gpickle")
QDRANT_HOST_ENV: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT_STR_ENV: Optional[str] = os.getenv("QDRANT_PORT", "6333")
QDRANT_COLLECTION_NAME_ENV: str = os.getenv(
    "QDRANT_COLLECTION_NAME", "literary_chunks_v1"
)
LLM_MODEL_NAME_ENV: str = os.getenv("LLM_MODEL", "llama3:8b")
EMBEDDING_MODEL_NAME_ENV: str = os.getenv(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
)
SPACY_MODEL_NAME_ENV: str = os.getenv("SPACY_MODEL_NAME", "en_core_web_sm")

MAX_CONTEXT_CHUNKS_ENV_STR: str = os.getenv("MAX_CONTEXT_CHUNKS", "5")
NEIGHBORHOOD_DEPTH_ENV_STR: str = os.getenv("NEIGHBORHOOD_DEPTH", "1")
QDRANT_CLIENT_TIMEOUT_ENV_STR: str = os.getenv("QDRANT_CLIENT_TIMEOUT", "20")
LLM_TEMPERATURE_ENV_STR: str = os.getenv("LLM_TEMPERATURE", "0.2")
LLM_NUM_CTX_ENV_STR: str = os.getenv("LLM_NUM_CTX", "4096") # Get as string first

# --- Parse Integer and Float Environment Variables Safely ---
def _parse_int_env(val_str: Optional[str], default: int, var_name: str) -> int:
    """Safely parses an integer from a string environment variable."""
    if val_str is None:
        print(f"Warning: Environment variable {var_name} not found. Using default: {default}")
        return default
    try:
        # Clean the string: remove comments and surrounding quotes
        cleaned_val = val_str.split('#')[0].strip().strip('"').strip("'")
        return int(cleaned_val)
    except ValueError:
        print(
            f"ERROR: Invalid value for {var_name}: '{val_str}'. "
            f"Expected an integer. Using default: {default}."
        )
        return default

def _parse_float_env(val_str: Optional[str], default: float, var_name: str) -> float:
    """Safely parses a float from a string environment variable."""
    if val_str is None:
        print(f"Warning: Environment variable {var_name} not found. Using default: {default}")
        return default
    try:
        cleaned_val = val_str.split('#')[0].strip().strip('"').strip("'")
        return float(cleaned_val)
    except ValueError:
        print(
            f"ERROR: Invalid value for {var_name}: '{val_str}'. "
            f"Expected a float. Using default: {default}."
        )
        return default

MAX_CONTEXT_CHUNKS_ENV: int = _parse_int_env(MAX_CONTEXT_CHUNKS_ENV_STR, 5, "MAX_CONTEXT_CHUNKS")
NEIGHBORHOOD_DEPTH_ENV: int = _parse_int_env(NEIGHBORHOOD_DEPTH_ENV_STR, 1, "NEIGHBORHOOD_DEPTH")
QDRANT_CLIENT_TIMEOUT_ENV: int = _parse_int_env(QDRANT_CLIENT_TIMEOUT_ENV_STR, 20, "QDRANT_CLIENT_TIMEOUT")
LLM_TEMPERATURE_ENV: float = _parse_float_env(LLM_TEMPERATURE_ENV_STR, 0.2, "LLM_TEMPERATURE")

# Specifically for LLM_NUM_CTX with debug
print(f"DEBUG query_engine.py: Raw value for LLM_NUM_CTX from os.getenv: '{LLM_NUM_CTX_ENV_STR}' (Type: {type(LLM_NUM_CTX_ENV_STR)})")
LLM_NUM_CTX_ENV: int = _parse_int_env(LLM_NUM_CTX_ENV_STR, 4096, "LLM_NUM_CTX")
print(f"DEBUG query_engine.py: Parsed value for LLM_NUM_CTX_ENV: {LLM_NUM_CTX_ENV}")


# Determine QDRANT_PORT based on whether it's in-memory or remote
if QDRANT_HOST_ENV == ":memory:" or not QDRANT_PORT_STR_ENV:
    QDRANT_PORT_ENV: Optional[int] = None
else:
    try:
        # QDRANT_PORT_STR_ENV can also have comments or quotes
        cleaned_port_str = QDRANT_PORT_STR_ENV.split('#')[0].strip().strip('"').strip("'")
        QDRANT_PORT_ENV = int(cleaned_port_str)
    except (ValueError, AttributeError): # AttributeError if QDRANT_PORT_STR_ENV is None
        print(
            f"ERROR: Invalid QDRANT_PORT value '{QDRANT_PORT_STR_ENV}'. "
            "Must be an integer. Defaulting to None or erroring if critical."
        )
        # Decide on fallback or raise error
        if QDRANT_HOST_ENV != ":memory:": # Only critical if not in-memory
             raise ValueError(f"Invalid QDRANT_PORT: {QDRANT_PORT_STR_ENV} for non-memory Qdrant.")
        QDRANT_PORT_ENV = None


class QueryEngineInitializationError(Exception):
    """Custom exception for errors during QueryEngine initialization."""
    pass


class LiteraryQueryEngine:
    """
    A query engine for literary texts, combining knowledge graph traversal,
    vector search, and LLM-based answer generation.
    """

    def __init__(
        self,
        graph_path: str = GRAPH_PATH_ENV,
        qdrant_host: str = QDRANT_HOST_ENV,
        qdrant_port: Optional[int] = QDRANT_PORT_ENV,
        qdrant_collection_name: str = QDRANT_COLLECTION_NAME_ENV,
        spacy_model_name: str = SPACY_MODEL_NAME_ENV,
        embedding_model_name: str = EMBEDDING_MODEL_NAME_ENV,
        llm_model_name: str = LLM_MODEL_NAME_ENV,
        max_context_chunks: int = MAX_CONTEXT_CHUNKS_ENV,
        neighborhood_depth: int = NEIGHBORHOOD_DEPTH_ENV,
        qdrant_client_timeout: int = QDRANT_CLIENT_TIMEOUT_ENV,
        llm_temperature: float = LLM_TEMPERATURE_ENV,
        llm_num_ctx: int = LLM_NUM_CTX_ENV,

    ) -> None:
        """
        Initializes the LiteraryQueryEngine by loading all necessary models and data.
        Args are passed to instance attributes.
        """
        print("Initializing Literary Query Engine...")
        self.graph_path = graph_path
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_collection_name = qdrant_collection_name
        self.spacy_model_name = spacy_model_name
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.max_context_chunks = max_context_chunks
        self.neighborhood_depth = neighborhood_depth
        self.qdrant_client_timeout = qdrant_client_timeout
        self.llm_temperature = llm_temperature
        self.llm_num_ctx = llm_num_ctx # This uses the globally parsed LLM_NUM_CTX_ENV

        try:
            self._load_spacy_model()
            self._load_embedding_model()
            self._load_graph()
            self._connect_qdrant()
            self._check_ollama()
            self._precompute_graph_nodes()

            print("Literary Query Engine initialized successfully.")
        except Exception as e:
            # Log the original error for debugging
            print(f"Detailed initialization error: {e}")
            raise QueryEngineInitializationError(
                f"Failed to initialize QueryEngine: {e}"
            )

    def _load_spacy_model(self) -> None:
        """Loads the spaCy model."""
        print(f"Loading spaCy model ({self.spacy_model_name})...")
        try:
            self.nlp_model: spacy.language.Language = spacy.load(self.spacy_model_name)
        except OSError:
            raise QueryEngineInitializationError(
                f"SpaCy model '{self.spacy_model_name}' not found. "
                f"Please run: python -m spacy download {self.spacy_model_name}"
            )

    def _load_embedding_model(self) -> None:
        """Loads the SentenceTransformer embedding model."""
        print(f"Loading sentence-transformer model ({self.embedding_model_name})...")
        self.embedding_model: SentenceTransformer = SentenceTransformer(
            self.embedding_model_name
        )

    def _load_graph(self) -> None:
        """Loads the knowledge graph."""
        print(f"Loading graph from {self.graph_path}...")
        if not os.path.exists(self.graph_path):
            raise QueryEngineInitializationError(
                f"Graph file not found at {self.graph_path}. "
                "Ensure 01_build_kg_and_vector_db.py has run successfully."
            )
        with open(self.graph_path, "rb") as gf:
            self.graph: nx.Graph = pickle.load(gf)
        if not self.graph.nodes:
            print("Warning: Loaded graph has no nodes.")


    def _precompute_graph_nodes(self) -> None:
        """Precomputes a set of graph nodes for efficient lookup."""
        if hasattr(self, 'graph') and self.graph and self.graph.nodes: # Check if graph has nodes
            self.graph_nodes_set: Set[str] = set(self.graph.nodes())
        else:
            self.graph_nodes_set = set()
            print("Warning: Graph has no nodes, so graph_nodes_set is empty.")


    def _connect_qdrant(self) -> None:
        """Connects to the Qdrant vector database and verifies collection."""
        print(
            f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}..."
        )
        if self.qdrant_host == ":memory:":
            print(
                "Warning: Attempting to use in-memory Qdrant for query engine. "
                "This is only valid if data was populated in the same session and process. "
                "For persistent data, use a non-:memory: Qdrant instance."
            )
            self.qdrant_client = QdrantClient(host=self.qdrant_host)
        else:
            if self.qdrant_port is None: # Should have been caught by QDRANT_PORT_ENV logic
                raise QueryEngineInitializationError(
                    "Qdrant port is None for a non-memory Qdrant instance. Check .env QDRANT_PORT."
                )
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                timeout=self.qdrant_client_timeout,
            )
        try:
            self.qdrant_client.get_collection(
                collection_name=self.qdrant_collection_name
            )
            print(
                f"Successfully connected to Qdrant and collection "
                f"'{self.qdrant_collection_name}' found."
            )
        except UnexpectedResponse as e:
            if e.status_code == 404:
                raise QueryEngineInitializationError(
                    f"Qdrant collection '{self.qdrant_collection_name}' not found. "
                    "Ensure 01_build_kg_and_vector_db.py ran successfully and "
                    "used the same collection name and Qdrant instance."
                )
            raise QueryEngineInitializationError(
                f"An unexpected Qdrant API error occurred: {e}"
            )
        except Exception as e:
            raise QueryEngineInitializationError(
                f"Could not connect to Qdrant at {self.qdrant_host}:{self.qdrant_port}. "
                f"Is it running? Error: {e}"
            )

    def _check_ollama(self) -> None:
        """Verifies that the Ollama service is reachable."""
        print("Verifying Ollama connection...")
        try:
            ollama.list()
            print("Ollama service is responsive.")
        except Exception as e:
            raise QueryEngineInitializationError(
                f"Could not connect to Ollama. Is the service running? Error: {e}"
            )

    def _normalize_entity_text(self, text: str) -> str:
        """
        Normalizes entity text by collapsing spaces, title-casing, and removing leading articles.
        """
        normalized = " ".join(text.strip().split()).title()
        articles = ["The ", "A ", "An "]
        for article in articles:
            if normalized.startswith(article):
                normalized = normalized[len(article) :].strip()
                break
        return normalized

    def get_entities_from_query(self, query_text: str) -> List[str]:
        """
        Extracts PERSON entities from the query text that are present in the graph.
        """
        doc = self.nlp_model(query_text)
        query_entities = [
            self._normalize_entity_text(ent.text)
            for ent in doc.ents
            if ent.label_ == "PERSON"
        ]
        valid_query_entities = [
            entity for entity in query_entities if entity and entity in self.graph_nodes_set
        ]
        return list(set(valid_query_entities))

    def get_graph_context_entities(self, entities: List[str]) -> List[str]:
        """
        Expands a list of entities by including their neighbors in the graph.
        """
        if not self.graph or not self.graph.nodes: # Check if graph has nodes
            print("Warning: Graph not loaded or has no nodes. Cannot get graph context.")
            return entities

        graph_context_nodes: Set[str] = set(entities)
        for entity_name in entities:
            if entity_name in self.graph:
                paths = nx.single_source_shortest_path_length(
                    self.graph, entity_name, cutoff=self.neighborhood_depth
                )
                graph_context_nodes.update(paths.keys())
        return list(graph_context_nodes)

    def get_relevant_chunks(
        self, query_text: str, filter_entities: Optional[List[str]] = None
    ) -> List[str]:
        """
        Retrieves relevant text chunks from Qdrant.
        """
        query_vector = self.embedding_model.encode(query_text).tolist()
        
        qdrant_filter_payload: Optional[Filter] = None
        if filter_entities:
            valid_filter_values = [
                str(e) for e in filter_entities if e and isinstance(e, str)
            ]
            if valid_filter_values:
                qdrant_filter_payload = Filter(
                    should=[
                        FieldCondition(
                            key="payload.entities_in_chunk",
                            match=MatchAny(any=valid_filter_values),
                        )
                    ]
                )
            else:
                print("  Note: No valid entities provided for Qdrant filtering.")
        
        try:
            search_results = self.qdrant_client.search(
                collection_name=self.qdrant_collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter_payload,
                limit=self.max_context_chunks,
            )
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []

        return [
            point.payload["text"]
            for point in search_results
            if point.payload and "text" in point.payload
        ]

    def _build_llm_prompt(self, user_query: str, context_str: str) -> str:
        """Constructs the prompt for the LLM."""
        return f"""You are a helpful assistant knowledgeable about a specific literary work.
Your knowledge is based *only* on the "Provided Context" below.
Answer the "User Question" truthfully and based *solely* on the information within the "Provided Context".
If the "Provided Context" does not contain the answer, you MUST state that the information is not available in the provided context.
Do not make up information or use any external knowledge beyond what is given in the context.

Provided Context:
{context_str}

User Question: {user_query}

Answer:"""

    def answer_query(self, user_query: str) -> str:
        """
        Answers a user's query using the RAG pipeline.
        """
        print(f"\nProcessing Query: '{user_query}'")

        query_entities = self.get_entities_from_query(user_query)
        print(f"  Identified entities in query (and in graph): {query_entities}")

        entities_for_qdrant_filter: List[str] = []
        if query_entities:
            expanded_entities = self.get_graph_context_entities(query_entities)
            print(
                f"  Expanded entities via graph (depth {self.neighborhood_depth}): "
                f"{expanded_entities}"
            )
            entities_for_qdrant_filter = expanded_entities
        else:
            print("  No specific entities from query found in graph to expand.")

        context_chunks = self.get_relevant_chunks(
            user_query, filter_entities=entities_for_qdrant_filter
        )

        if not context_chunks and entities_for_qdrant_filter:
            print(
                "  No relevant chunks found with entity filtering. "
                "Retrying with pure semantic search..."
            )
            context_chunks = self.get_relevant_chunks(user_query, filter_entities=None)

        if not context_chunks:
            print("  No relevant chunks found in Qdrant even after fallback.")
            return (
                "I could not find relevant information in the knowledge base to "
                "answer your question."
            )

        context_str = "\n\n---\n\n".join(context_chunks)
        prompt = self._build_llm_prompt(user_query, context_str)
        
        print(f"  Asking LLM ({self.llm_model_name})... (Temp: {self.llm_temperature}, Ctx: {self.llm_num_ctx})")
        try:
            response = ollama.generate(
                model=self.llm_model_name,
                prompt=prompt,
                options={
                    "temperature": self.llm_temperature,
                    "num_ctx": self.llm_num_ctx # Use the instance attribute
                },
            )
            answer = response["response"].strip()
            print(f"  LLM Answer: {answer}")
            return answer
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return (
                "Sorry, I encountered an error trying to generate an answer "
                f"with the LLM: {e}"
            )

# --- Self-test / Example Usage ---
if __name__ == "__main__":
    print("Running Literary Query Engine self-test...")
    query_engine_instance: Optional[LiteraryQueryEngine] = None
    try:
        query_engine_instance = LiteraryQueryEngine()
        test_queries = [
            "Who is Raskolnikov?",
            "What is the relationship between Raskolnikov and Sonia?",
        ]
        for t_query in test_queries:
            llm_answer = query_engine_instance.answer_query(t_query)
    except QueryEngineInitializationError as e:
        print(f"FATAL: Query engine self-test failed during initialization: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during self-test: {e}")
    finally:
        if hasattr(query_engine_instance, 'qdrant_client') and query_engine_instance.qdrant_client:
             print("Query engine operations finished.")