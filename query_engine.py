# graphrag_literary/query_engine.py
import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, ScoredPoint
from qdrant_client.http.exceptions import UnexpectedResponse
import ollama
import pickle
import os
from dotenv import load_dotenv
from typing import List, Set, Optional, Dict, Any, Tuple

# Load environment variables from .env file
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
LLM_NUM_CTX_ENV_STR: str = os.getenv("LLM_NUM_CTX", "4096")
NUM_CHAT_HISTORY_TURNS_ENV_STR: str = os.getenv("NUM_CHAT_HISTORY_TURNS", "2")


# --- Parse Integer and Float Environment Variables Safely ---
def _parse_int_env(val_str: Optional[str], default: int, var_name: str) -> int:
    if val_str is None:
        print(f"Warning: Environment variable {var_name} not found. Using default: {default}")
        return default
    try:
        cleaned_val = val_str.split('#')[0].strip().strip('"').strip("'")
        return int(cleaned_val)
    except ValueError:
        print(
            f"ERROR: Invalid value for {var_name}: '{val_str}'. "
            f"Expected an integer. Using default: {default}."
        )
        return default

def _parse_float_env(val_str: Optional[str], default: float, var_name: str) -> float:
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

print(f"DEBUG query_engine.py: Raw value for LLM_NUM_CTX from os.getenv: '{LLM_NUM_CTX_ENV_STR}' (Type: {type(LLM_NUM_CTX_ENV_STR)})")
LLM_NUM_CTX_ENV: int = _parse_int_env(LLM_NUM_CTX_ENV_STR, 4096, "LLM_NUM_CTX")
print(f"DEBUG query_engine.py: Parsed value for LLM_NUM_CTX_ENV: {LLM_NUM_CTX_ENV}")

NUM_CHAT_HISTORY_TURNS_ENV: int = _parse_int_env(NUM_CHAT_HISTORY_TURNS_ENV_STR, 2, "NUM_CHAT_HISTORY_TURNS")


if QDRANT_HOST_ENV == ":memory:" or not QDRANT_PORT_STR_ENV:
    QDRANT_PORT_ENV: Optional[int] = None
else:
    try:
        cleaned_port_str = QDRANT_PORT_STR_ENV.split('#')[0].strip().strip('"').strip("'")
        QDRANT_PORT_ENV = int(cleaned_port_str)
    except (ValueError, AttributeError):
        print(
            f"ERROR: Invalid QDRANT_PORT value '{QDRANT_PORT_STR_ENV}'. Must be an integer."
        )
        if QDRANT_HOST_ENV != ":memory:":
             raise ValueError(f"Invalid QDRANT_PORT: {QDRANT_PORT_STR_ENV} for non-memory Qdrant.")
        QDRANT_PORT_ENV = None


class QueryEngineInitializationError(Exception):
    pass


class LiteraryQueryEngine:
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
        num_chat_history_turns: int = NUM_CHAT_HISTORY_TURNS_ENV,
    ) -> None:
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
        self.llm_num_ctx = llm_num_ctx
        self.num_chat_history_turns = num_chat_history_turns
        print(f"Query Engine Config: Using num_chat_history_turns = {self.num_chat_history_turns}")

        try:
            self._load_spacy_model()
            self._load_embedding_model()
            self._load_graph()
            self._connect_qdrant()
            self._check_ollama()
            self._precompute_graph_nodes()
            print("Literary Query Engine initialized successfully.")
        except Exception as e:
            print(f"Detailed initialization error: {e}")
            raise QueryEngineInitializationError(f"Failed to initialize QueryEngine: {e}")

    def _load_spacy_model(self) -> None:
        print(f"Loading spaCy model ({self.spacy_model_name})...")
        try:
            self.nlp_model: spacy.language.Language = spacy.load(self.spacy_model_name)
        except OSError:
            raise QueryEngineInitializationError(
                f"SpaCy model '{self.spacy_model_name}' not found. Run: python -m spacy download {self.spacy_model_name}"
            )

    def _load_embedding_model(self) -> None:
        print(f"Loading sentence-transformer model ({self.embedding_model_name})...")
        self.embedding_model: SentenceTransformer = SentenceTransformer(self.embedding_model_name)

    def _load_graph(self) -> None:
        print(f"Loading graph from {self.graph_path}...")
        if not os.path.exists(self.graph_path):
            raise QueryEngineInitializationError(f"Graph file not found: {self.graph_path}.")
        with open(self.graph_path, "rb") as gf:
            self.graph: nx.Graph = pickle.load(gf)
        if not self.graph.nodes:
            print("Warning: Loaded graph has no nodes.")

    def _precompute_graph_nodes(self) -> None:
        if hasattr(self, 'graph') and self.graph and self.graph.nodes:
            self.graph_nodes_set: Set[str] = set(self.graph.nodes())
        else:
            self.graph_nodes_set = set()
            print("Warning: Graph has no nodes or not loaded; graph_nodes_set is empty.")

    def _connect_qdrant(self) -> None:
        print(f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}...")
        if self.qdrant_host == ":memory:":
            print("Warning: Using in-memory Qdrant. Data persistence requires a server.")
            self.qdrant_client = QdrantClient(host=self.qdrant_host)
        else:
            if self.qdrant_port is None:
                raise QueryEngineInitializationError("Qdrant port is None for non-memory Qdrant.")
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host, port=self.qdrant_port, timeout=self.qdrant_client_timeout
            )
        try:
            self.qdrant_client.get_collection(collection_name=self.qdrant_collection_name)
            print(f"Successfully connected to Qdrant and collection '{self.qdrant_collection_name}' found.")
        except UnexpectedResponse as e:
            if e.status_code == 404:
                raise QueryEngineInitializationError(f"Qdrant collection '{self.qdrant_collection_name}' not found.")
            raise QueryEngineInitializationError(f"Unexpected Qdrant API error: {e}")
        except Exception as e:
            raise QueryEngineInitializationError(f"Could not connect to Qdrant: {e}")

    def _check_ollama(self) -> None:
        print("Verifying Ollama connection...")
        try:
            ollama.list()
            print("Ollama service is responsive.")
        except Exception as e:
            raise QueryEngineInitializationError(f"Could not connect to Ollama: {e}")

    def _normalize_entity_text(self, text: str) -> str:
        normalized = " ".join(text.strip().split()).title()
        articles = ["The ", "A ", "An "]
        for article in articles:
            if normalized.startswith(article):
                normalized = normalized[len(article):].strip()
                break
        return normalized

    def get_entities_from_query(self, query_text: str) -> List[str]:
        doc = self.nlp_model(query_text)
        query_entities = [
            self._normalize_entity_text(ent.text)
            for ent in doc.ents if ent.label_ == "PERSON"
        ]
        return list(set(entity for entity in query_entities if entity and entity in self.graph_nodes_set))

    def get_graph_context_entities(self, entities: List[str]) -> List[str]:
        if not self.graph or not self.graph.nodes:
            print("Warning: Graph not loaded/empty. Cannot get graph context.")
            return entities
        graph_context_nodes: Set[str] = set(entities)
        for entity_name in entities:
            if entity_name in self.graph:
                paths = nx.single_source_shortest_path_length(
                    self.graph, entity_name, cutoff=self.neighborhood_depth
                )
                graph_context_nodes.update(paths.keys())
        return list(graph_context_nodes)

    def get_relevant_chunks_with_indices(
        self, query_text: str, filter_entities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves relevant text chunks from Qdrant, returning them with their indices."""
        query_vector = self.embedding_model.encode(query_text).tolist()
        qdrant_filter_payload: Optional[Filter] = None
        if filter_entities:
            valid_filter_values = [str(e) for e in filter_entities if e and isinstance(e, str)]
            if valid_filter_values:
                qdrant_filter_payload = Filter(
                    should=[FieldCondition(key="payload.entities_in_chunk", match=MatchAny(any=valid_filter_values))]
                )
            else:
                print("  Note: No valid entities for Qdrant filtering.")
        
        try:
            search_results: List[ScoredPoint] = self.qdrant_client.search(
                collection_name=self.qdrant_collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter_payload,
                limit=self.max_context_chunks,
                with_payload=True
            )
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []

        retrieved_points_data: List[Dict[str, Any]] = []
        for point in search_results:
            if point.payload and "text" in point.payload and "chunk_idx" in point.payload:
                retrieved_points_data.append(
                    {"text": point.payload["text"], "chunk_idx": point.payload["chunk_idx"]}
                )
        return retrieved_points_data

    def _build_llm_prompt(
        self, user_query: str, context_str: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        history_str_segment = ""
        if chat_history and self.num_chat_history_turns > 0:
            num_messages_to_include = self.num_chat_history_turns * 2
            history_up_to_current_query = chat_history[:-1]
            actual_history_to_format = history_up_to_current_query[-num_messages_to_include:]
            if actual_history_to_format:
                formatted_history_lines = ["Previous conversation:"]
                for message in actual_history_to_format:
                    role_display = "User" if message["role"] == "user" else "Assistant"
                    formatted_history_lines.append(f"{role_display}: {message['content']}")
                history_str_segment = "\n".join(formatted_history_lines) + "\n\n---\n\n"

        return f"""You are a helpful assistant knowledgeable about a specific literary work.
Your knowledge is based *only* on the "Provided Context" below. The text segments in "Provided Context" are presented in their original chronological order from the novel.
{history_str_segment}If the conversation history is provided (as "Previous conversation"), use it to understand the context of the "Current User Question".
Answer the "Current User Question" truthfully and based *solely* on the information within the "Provided Context" and relevant "Previous conversation".
If the "Provided Context" or "Previous conversation" does not contain the answer, you MUST state that the information is not available.
Do not make up information or use any external knowledge.

Provided Context:
{context_str}

Current User Question: {user_query}

Answer:"""

    def answer_query(
        self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        print(f"\nProcessing Query: '{user_query}'")
        if chat_history and self.num_chat_history_turns > 0:
            history_for_log = chat_history[:-1]
            num_messages_to_log = min(len(history_for_log), self.num_chat_history_turns * 2)
            if num_messages_to_log > 0:
                print(f"  With chat history (last {num_messages_to_log} prior messages considered):")

        query_for_rag = user_query
        query_entities = self.get_entities_from_query(query_for_rag)
        print(f"  Identified entities for RAG (from '{query_for_rag[:50]}...'): {query_entities}")

        entities_for_qdrant_filter: List[str] = []
        if query_entities:
            expanded_entities = self.get_graph_context_entities(query_entities)
            print(f"  Expanded entities via graph (depth {self.neighborhood_depth}): {expanded_entities}")
            entities_for_qdrant_filter = expanded_entities
        else:
            print("  No specific entities from query found in graph for RAG.")

        retrieved_points_data = self.get_relevant_chunks_with_indices(
            query_for_rag, filter_entities=entities_for_qdrant_filter
        )

        if not retrieved_points_data and entities_for_qdrant_filter:
            print("  No RAG chunks with entity filtering. Retrying with pure semantic search...")
            retrieved_points_data = self.get_relevant_chunks_with_indices(query_for_rag, filter_entities=None)

        context_str: str
        if not retrieved_points_data:
            print("  No relevant RAG chunks found even after fallback.")
            context_str = "No specific context was retrieved from the text for this question. Answer based on previous conversation if relevant, or state that information is not available from the provided text."
        else:
            retrieved_points_data.sort(key=lambda item: item["chunk_idx"])
            print(f"  Retrieved {len(retrieved_points_data)} RAG chunks, sorted by original index.")
            chunk_texts = [p["text"] for p in retrieved_points_data]
            context_str = "\n\n---\n\n".join(chunk_texts)
        
        prompt = self._build_llm_prompt(user_query, context_str, chat_history)
        
        print(f"  Asking LLM ({self.llm_model_name})... (Temp: {self.llm_temperature}, Ctx: {self.llm_num_ctx})")
        try:
            response = ollama.generate(
                model=self.llm_model_name,
                prompt=prompt,
                options={"temperature": self.llm_temperature, "num_ctx": self.llm_num_ctx},
            )
            answer = response["response"].strip()
            print(f"  LLM Answer: {answer}")
            return answer
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return f"Sorry, I encountered an error trying to generate an answer: {e}"

# --- Self-test / Example Usage ---
if __name__ == "__main__":
    print("Running Literary Query Engine self-test...")
    query_engine_instance: Optional[LiteraryQueryEngine] = None
    try:
        query_engine_instance = LiteraryQueryEngine()
        
        print("\n--- Test Query 1 (No History) ---")
        query1 = "Who is Raskolnikov?"
        answer1 = query_engine_instance.answer_query(query1)
        print(f"Query: {query1}\nAnswer: {answer1}")

        history_for_test2 = [
            {"role": "user", "content": query1},
            {"role": "assistant", "content": answer1},
            {"role": "user", "content": "What are his main problems?"}
        ]
        print("\n--- Test Query 2 (With History) ---")
        query2 = history_for_test2[-1]["content"]
        answer2 = query_engine_instance.answer_query(query2, chat_history=history_for_test2)
        print(f"Query: {query2}\nAnswer: {answer2}")

    except QueryEngineInitializationError as e:
        print(f"FATAL: Query engine self-test failed during initialization: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during self-test: {e}")
    finally:
        if hasattr(query_engine_instance, 'qdrant_client') and query_engine_instance.qdrant_client:
             print("\nQuery engine operations finished.")