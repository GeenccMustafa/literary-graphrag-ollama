import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
import re
from collections import Counter
import pickle
import uuid
import os
from dotenv import load_dotenv
from itertools import combinations
from typing import List, Tuple, Set, Optional, Any

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants ---
NOVEL_PATH: str = os.getenv("NOVEL_PATH", "data/Crime_and_Punishment.txt")
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT_STR: Optional[str] = os.getenv("QDRANT_PORT", "6333")
QDRANT_COLLECTION_NAME: str = os.getenv(
    "QDRANT_COLLECTION_NAME", "literary_chunks_v1"
)
GRAPH_OUTPUT_PATH: str = os.getenv("GRAPH_PATH", "data/literary_graph.gpickle")
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "10"))
MIN_ENTITY_OCCURRENCE: int = int(os.getenv("MIN_ENTITY_OCCURRENCE", "5"))
QDRANT_BATCH_SIZE: int = int(os.getenv("QDRANT_BATCH_SIZE", "100"))
SPACY_MODEL_NAME: str = os.getenv("SPACY_MODEL_NAME", "en_core_web_sm")
SENTENCE_BATCH_SIZE_FOR_GLOBAL_NER: int = int(os.getenv("SENTENCE_BATCH_SIZE_FOR_GLOBAL_NER", "500"))


# Determine QDRANT_PORT based on whether it's in-memory or remote
if QDRANT_HOST == ":memory:" or not QDRANT_PORT_STR:
    QDRANT_PORT: Optional[int] = None
else:
    try:
        QDRANT_PORT = int(QDRANT_PORT_STR)
    except ValueError:
        print(
            f"ERROR: Invalid QDRANT_PORT value '{QDRANT_PORT_STR}'. "
            "Must be an integer. Exiting."
        )
        exit(1)


def _load_text(file_path: str) -> str:
    """Loads text content from the specified file."""
    print(f"Loading text from {file_path}...")
    if not os.path.exists(file_path):
        print(
            f"ERROR: Novel file not found at {file_path}. "
            "Please check your .env settings and file location."
        )
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
    except Exception as e:
        print(f"ERROR: Could not read file at {file_path}. Error: {e}")
        raise
    
    if not text_content.strip():
        print(f"ERROR: The file {file_path} is empty or contains only whitespace.")
        raise ValueError(f"File {file_path} is empty.")
        
    print(f"Loaded text with {len(text_content)} characters.")
    return text_content


def _load_spacy_model(
    model_name: str, text_length: int
) -> spacy.language.Language:
    """Loads the spaCy model and adjusts max_length if necessary."""
    print(f"Loading spaCy model ({model_name})...")
    try:
        nlp = spacy.load(model_name)
        if text_length >= nlp.max_length:
            nlp.max_length = text_length + 100 
            print(
                f"Increased spaCy nlp.max_length to {nlp.max_length} "
                "characters."
            )
    except OSError:
        print(
            f"ERROR: spaCy model '{model_name}' not found. "
            f"Please run: python -m spacy download {model_name}"
        )
        raise
    except Exception as e:
        print(f"ERROR: Could not load spaCy model. Error: {e}")
        raise
    return nlp


def _load_embedding_model(
    model_name: str,
) -> Tuple[SentenceTransformer, int]:
    """Loads the sentence-transformer model."""
    print(f"Loading sentence-transformer model: {model_name}...")
    try:
        embedding_model = SentenceTransformer(model_name)
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        if embedding_dim is None:
            raise ValueError(
                "Could not determine embedding dimension from model."
            )
    except Exception as e:
        print(
            f"ERROR: Could not load sentence-transformer model "
            f"'{model_name}'. Error: {e}"
        )
        raise
    return embedding_model, embedding_dim


def _setup_qdrant(
    host: str,
    port: Optional[int],
    collection_name: str,
    embedding_dim: int,
) -> QdrantClient:
    """Sets up the Qdrant client and ensures the collection exists."""
    print("Setting up Qdrant client...")
    if host == ":memory:":
        print("Using in-memory Qdrant.")
        qdrant_client = QdrantClient(":memory:")
    else:
        print(f"Connecting to Qdrant at {host}:{port}")
        try:
            qdrant_client = QdrantClient(host=host, port=port, timeout=30)
            qdrant_client.get_collections() 
            print("Successfully connected to Qdrant.")
        except Exception as e:
            print(
                f"ERROR: Could not connect to Qdrant at {host}:{port}. "
                f"Is it running? Error: {e}"
            )
            raise

    try:
        qdrant_client.get_collection(collection_name=collection_name)
        print(
            f"Collection '{collection_name}' already exists. "
            "Recreating it to ensure freshness."
        )
        qdrant_client.delete_collection(collection_name=collection_name)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim, distance=Distance.COSINE
            ),
        )
        print(f"Collection '{collection_name}' recreated.")
    except UnexpectedResponse as e:
        if e.status_code == 404:  
            print(f"Collection '{collection_name}' not found. Creating...")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim, distance=Distance.COSINE
                ),
            )
            print(f"Collection '{collection_name}' created.")
        else:
            print(
                f"ERROR: An unexpected error occurred with Qdrant collection "
                f"'{collection_name}'. Error: {e}"
            )
            raise
    except Exception as e:
        print(
            f"ERROR: Could not ensure Qdrant collection "
            f"'{collection_name}' exists. Error: {e}"
        )
        raise
    return qdrant_client


def _preprocess_text_and_create_spacy_doc(
    text_content: str, nlp_model: spacy.language.Language
) -> spacy.tokens.Doc:
    """Cleans text and processes it with spaCy."""
    cleaned_text = re.sub(r"\s+", " ", text_content).strip()
    print(
        "Processing full text with spaCy... "
        "(This may take a while for long texts)"
    )
    try:
        doc = nlp_model(cleaned_text)
    except Exception as e:
        print(
            "ERROR: spaCy failed to process the full text even after "
            "increasing max_length."
        )
        print(
            "This could be due to extreme text length or unusual characters. "
            "Consider further text preprocessing or segmenting the input "
            "to spaCy if issues persist."
        )
        print(f"Specific error: {e}")
        raise
    return doc


def _normalize_entity_text(text: str) -> str:
    """Normalizes entity text by collapsing spaces, title-casing, and removing leading articles."""
    normalized = " ".join(text.strip().split())
    normalized = normalized.title()

    articles = ["The ", "A ", "An "]
    for article in articles:
        if normalized.startswith(article):
            normalized = normalized[len(article):].strip()
            break 
    return normalized


def _create_text_chunks_and_extract_entities(
    doc: spacy.tokens.Doc, chunk_size: int, nlp_model: spacy.language.Language
) -> Tuple[List[str], List[List[str]]]:
    """
    Creates text chunks from sentences and extracts entities for each chunk.
    Uses the pre-processed spaCy Doc object for efficiency.
    """
    spacy_sents = [sent for sent in doc.sents if sent.text.strip()]
    if not spacy_sents:
        print(f"ERROR: No sentences extracted from the provided text.")
        raise ValueError("No sentences found in document.")
    print(f"Extracted {len(spacy_sents)} sentences.")

    print(f"Chunking text into groups of {chunk_size} sentences...")
    text_chunks: List[str] = []
    chunk_entities_map: List[List[str]] = []

    for i in range(0, len(spacy_sents), chunk_size):
        chunk_sents = spacy_sents[i : i + chunk_size]
        if not chunk_sents:
            continue

        chunk_text = " ".join(sent.text for sent in chunk_sents)
        text_chunks.append(chunk_text)

        # Efficient entity extraction using the existing Doc object
        # Get character start and end of the current chunk relative to the original doc
        chunk_start_char = chunk_sents[0].start_char
        chunk_end_char = chunk_sents[-1].end_char
        
        current_chunk_entities: Set[str] = set()
        for ent in doc.ents:
            if (
                ent.start_char >= chunk_start_char
                and ent.end_char <= chunk_end_char
            ):
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                    normalized_name = _normalize_entity_text(ent.text)
                    if normalized_name: # Ensure not empty after normalization
                        current_chunk_entities.add(normalized_name)
        chunk_entities_map.append(list(current_chunk_entities))

    print(f"Created {len(text_chunks)} text chunks.")
    return text_chunks, chunk_entities_map


def _identify_main_characters(
    doc: spacy.tokens.Doc, min_occurrence: int
) -> Set[str]:
    """Identifies main characters based on PERSON entity occurrences in the doc."""
    print("Extracting global PERSON entities for graph nodes...")
    global_person_entity_counts: Counter = Counter()

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = _normalize_entity_text(ent.text)
            if name and len(name) > 1:  # Basic filter for valid names
                global_person_entity_counts[name] += 1
    
    main_characters = {
        entity
        for entity, count in global_person_entity_counts.items()
        if count >= min_occurrence
    }
    print(
        f"Identified {len(main_characters)} main characters "
        f"(occurring >= {min_occurrence} times): "
        f"{list(main_characters)[:30]}"
        f"{'...' if len(main_characters) > 30 else ''}"
    )
    return main_characters


def _build_knowledge_graph(
    main_characters: Set[str], chunk_entities_map: List[List[str]]
) -> nx.Graph:
    """Builds a knowledge graph based on co-occurrence of main characters in chunks."""
    print("Building knowledge graph...")
    graph = nx.Graph()
    for char_node in main_characters:
        graph.add_node(char_node, type="character")

    for entities_in_chunk in chunk_entities_map:
        # Filter entities in chunk to include only main characters
        present_main_chars = {
            entity for entity in entities_in_chunk if entity in main_characters
        }

        if len(present_main_chars) > 1:
            for char1, char2 in combinations(sorted(list(present_main_chars)), 2): # sorted for consistent edge representation if needed later
                if graph.has_edge(char1, char2):
                    graph[char1][char2]["weight"] += 1
                else:
                    graph.add_edge(char1, char2, weight=1, type="co-occurs_with")
    
    print(
        f"Graph built with {graph.number_of_nodes()} nodes and "
        f"{graph.number_of_edges()} edges."
    )
    return graph


def _save_graph(graph: nx.Graph, output_path: str) -> None:
    """Saves the graph to a Gpickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as gf:
        pickle.dump(graph, gf)
    print(f"Graph saved to {output_path}")


def _embed_and_store_chunks(
    text_chunks: List[str],
    chunk_entities_map: List[List[str]],
    embedding_model: SentenceTransformer,
    qdrant_client: QdrantClient,
    collection_name: str,
    batch_size: int,
) -> None:
    """Generates embeddings for text chunks and stores them in Qdrant."""
    print(
        f"Generating embeddings for {len(text_chunks)} chunks and "
        "storing in Qdrant..."
    )
    points_to_upsert: List[PointStruct] = []

    for i, chunk_text in enumerate(text_chunks):
        if (i + 1) % (max(1, batch_size // 2)) == 0 or i == 0:
            print(f"  Embedding chunk {i+1}/{len(text_chunks)}...")

        embedding = embedding_model.encode(chunk_text).tolist()
        payload = {
            "text": chunk_text,
            "chunk_idx": i,
            "entities_in_chunk": chunk_entities_map[i],
        }
        points_to_upsert.append(
            PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)
        )

        if len(points_to_upsert) >= batch_size:
            qdrant_client.upsert(
                collection_name=collection_name, points=points_to_upsert
            )
            print(f"  Upserted {len(points_to_upsert)} points to Qdrant...")
            points_to_upsert = []

    if points_to_upsert:
        qdrant_client.upsert(
            collection_name=collection_name, points=points_to_upsert
        )
        print(
            f"  Upserted final {len(points_to_upsert)} points to Qdrant."
        )


def main() -> None:
    """
    Main orchestrator for building the knowledge graph and vector database.
    Loads a novel, processes it to extract entities and relationships,
    builds a graph of character co-occurrences, and stores text chunks
    with their embeddings and associated entities in a Qdrant vector database.
    """
    print("--- Starting Knowledge Graph and Vector DB Construction ---")

    try:
        # --- 1. Load Text and Models ---
        text_content = _load_text(NOVEL_PATH)
        nlp = _load_spacy_model(SPACY_MODEL_NAME, len(text_content))
        embedding_model, embedding_dim = _load_embedding_model(
            EMBEDDING_MODEL_NAME
        )

        # --- 2. Qdrant Setup ---
        qdrant_client = _setup_qdrant(
            QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, embedding_dim
        )

        # --- 3. Process Text and Extract Chunks/Entities ---
        doc = _preprocess_text_and_create_spacy_doc(text_content, nlp)
        text_chunks, chunk_entities_map = (
            _create_text_chunks_and_extract_entities(
                doc, CHUNK_SIZE, nlp
            )
        )
        
        if not text_chunks:
            print("No text chunks were created. Exiting.")
            return

        # --- 4. Identify Main Characters and Build Graph ---
        main_characters = _identify_main_characters(doc, MIN_ENTITY_OCCURRENCE)
        knowledge_graph = _build_knowledge_graph(
            main_characters, chunk_entities_map
        )
        _save_graph(knowledge_graph, GRAPH_OUTPUT_PATH)

        # --- 5. Generate Embeddings & Store in Qdrant ---
        _embed_and_store_chunks(
            text_chunks,
            chunk_entities_map,
            embedding_model,
            qdrant_client,
            QDRANT_COLLECTION_NAME,
            QDRANT_BATCH_SIZE,
        )

        print("--- Knowledge Graph and Vector DB Construction Finished ---")
        collection_info = qdrant_client.get_collection(
            collection_name=QDRANT_COLLECTION_NAME
        )
        print(
            f"Total points in Qdrant collection "
            f"'{QDRANT_COLLECTION_NAME}': {collection_info.points_count}"
        )

    except (FileNotFoundError, ValueError, ConnectionError, OSError, Exception) as e:
        print(f"An critical error occurred during the process: {e}")
        print("Process aborted.")


if __name__ == "__main__":
    main()