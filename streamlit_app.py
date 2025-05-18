# graphrag_literary/streamlit_app.py
import streamlit as st
from query_engine import (
    LiteraryQueryEngine,
    QueryEngineInitializationError,
    # Import the globally parsed ENV variables from query_engine.py
    # These are already processed and type-cast correctly there.
    GRAPH_PATH_ENV, QDRANT_HOST_ENV, QDRANT_PORT_ENV, QDRANT_COLLECTION_NAME_ENV,
    SPACY_MODEL_NAME_ENV, EMBEDDING_MODEL_NAME_ENV, LLM_MODEL_NAME_ENV,
    MAX_CONTEXT_CHUNKS_ENV, NEIGHBORHOOD_DEPTH_ENV, QDRANT_CLIENT_TIMEOUT_ENV,
    LLM_TEMPERATURE_ENV, LLM_NUM_CTX_ENV, NUM_CHAT_HISTORY_TURNS_ENV # Added this
)
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# Load .env variables (good practice at the top of the main app script)
load_dotenv()

# --- Page Configuration (run only once at the start of the script) ---
st.set_page_config(
    page_title="Literary GraphRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Application State Initialization ---
if "app_state" not in st.session_state:
    st.session_state.app_state: Dict[str, Any] = {
        "init_error_message": None,
        "query_engine_initialized": False,
        "chat_messages": [
            {
                "role": "assistant",
                "content": "Hello! How can I help you explore the novel today?",
            }
        ],
    }

# --- Resource Initialization ---
@st.cache_resource  # Caches the *instance* of LiteraryQueryEngine
def get_query_engine_instance() -> Optional[LiteraryQueryEngine]:
    """
    Initializes and returns an instance of the LiteraryQueryEngine.
    This function is cached by Streamlit, so the engine is initialized only once.
    It uses the globally parsed and type-casted _ENV variables from query_engine.py.
    """
    print("Attempting to initialize LiteraryQueryEngine via Streamlit...")
    try:
        engine = LiteraryQueryEngine(
            graph_path=GRAPH_PATH_ENV,
            qdrant_host=QDRANT_HOST_ENV,
            qdrant_port=QDRANT_PORT_ENV,
            qdrant_collection_name=QDRANT_COLLECTION_NAME_ENV,
            spacy_model_name=SPACY_MODEL_NAME_ENV,
            embedding_model_name=EMBEDDING_MODEL_NAME_ENV,
            llm_model_name=LLM_MODEL_NAME_ENV,
            max_context_chunks=MAX_CONTEXT_CHUNKS_ENV,
            neighborhood_depth=NEIGHBORHOOD_DEPTH_ENV,
            qdrant_client_timeout=QDRANT_CLIENT_TIMEOUT_ENV,
            llm_temperature=LLM_TEMPERATURE_ENV,
            llm_num_ctx=LLM_NUM_CTX_ENV,
            num_chat_history_turns=NUM_CHAT_HISTORY_TURNS_ENV # Pass the new parameter
        )
        st.session_state.app_state["query_engine_initialized"] = True
        st.session_state.app_state["init_error_message"] = None
        print("LiteraryQueryEngine initialized successfully via Streamlit.")
        return engine
    except QueryEngineInitializationError as e:
        error_msg = f"Query Engine Initialization Failed: {e}"
        st.session_state.app_state["init_error_message"] = error_msg
        st.session_state.app_state["query_engine_initialized"] = False
        print(f"Streamlit: {error_msg}")
        return None
    except Exception as e:
        error_msg = f"An unexpected error occurred during engine initialization in Streamlit: {e}"
        st.session_state.app_state["init_error_message"] = error_msg
        st.session_state.app_state["query_engine_initialized"] = False
        print(f"Streamlit: {error_msg} (Type: {type(e)})") # Added type of exception
        return None


# --- Main Application Logic ---
def run_app():
    """Main function to render the Streamlit application."""
    st.title("📚 Literary GraphRAG")
    st.caption(
        f"Querying '{os.path.basename(os.getenv('NOVEL_PATH', 'your novel'))}' "
        f"using GraphRAG with LLM: {os.getenv('LLM_MODEL', 'N/A')}"
    )

    query_engine: Optional[LiteraryQueryEngine] = get_query_engine_instance()

    if st.session_state.app_state["init_error_message"]:
        st.error(
            f"**Application Initialization Failed:**\n\n"
            f"{st.session_state.app_state['init_error_message']}"
        )
        st.warning(
            "The application may not function correctly. Please check your setup "
            "(Qdrant, Ollama, file paths in .env) and console logs, then refresh the page."
        )
        return

    if not st.session_state.app_state["query_engine_initialized"] or query_engine is None:
        st.warning(
            "Resources are still loading or failed to load. Please wait or refresh."
        )
        with st.spinner("Initializing Literary GraphRAG Engine... This may take a moment."):
            pass # Actual loading is handled by the @st.cache_resource function call above
        
        # Re-check state after the cached function has run at least once
        if not st.session_state.app_state["query_engine_initialized"]:
             if not st.session_state.app_state["init_error_message"]:
                st.session_state.app_state["init_error_message"] = "Engine initialization did not complete successfully after first attempt."
             st.error(
                f"**Application Initialization Failed (Post-Spinner):**\n\n"
                f"{st.session_state.app_state['init_error_message']}"
             )
             return
    else:
        st.success("Literary GraphRAG Engine is ready!")

        # Display chat messages from history
        for message in st.session_state.app_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question about the novel..."):
            # Add user message to app's chat history
            st.session_state.app_state["chat_messages"].append(
                {"role": "user", "content": prompt}
            )
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        # Pass the current full chat history to the query engine
                        # The engine will decide how much of it to use based on its config
                        full_response = query_engine.answer_query(
                            prompt,
                            chat_history=st.session_state.app_state["chat_messages"]
                        )
                        message_placeholder.markdown(full_response)
                    except Exception as e:
                        err_msg = (
                            f"An unexpected error occurred while processing your query: {e}"
                        )
                        st.error(err_msg) # Display error in the main chat area
                        message_placeholder.markdown(err_msg) # And in the placeholder
                        full_response = "Error: Could not process request." # Fallback for history

            # Add assistant response to app's chat history
            st.session_state.app_state["chat_messages"].append(
                {"role": "assistant", "content": full_response}
            )

# --- Sidebar Content ---
def render_sidebar():
    """Renders the sidebar content."""
    st.sidebar.header("About GraphRAG")
    st.sidebar.info(
        "This application uses Retrieval Augmented Generation (RAG) enhanced by a "
        "Knowledge Graph (KG). Key steps:\n"
        "1. A KG of characters and their co-occurrences is built from the novel.\n"
        "2. Text chunks are stored in a Qdrant vector database.\n"
        "3. When you ask a question, relevant entities are identified.\n"
        "4. The KG helps find related entities and concepts.\n"
        "5. Relevant text chunks are retrieved from Qdrant using this "
        "graph-enhanced context.\n"
        "6. An LLM (via Ollama) generates an answer based on your query and the "
        "retrieved context, potentially considering recent chat history." # Added history mention
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Configuration")
    
    # Display configuration values (raw values from .env for info purposes)
    config_details = {
        "Novel": os.path.basename(os.getenv("NOVEL_PATH", "N/A")),
        "LLM": os.getenv("LLM_MODEL", "N/A"),
        "Embedding Model": os.getenv("EMBEDDING_MODEL_NAME", "N/A"),
        "SpaCy Model": os.getenv("SPACY_MODEL_NAME", "N/A"),
        "Qdrant Collection": os.getenv("QDRANT_COLLECTION_NAME", "N/A"),
        "Qdrant Host": f"{os.getenv('QDRANT_HOST', 'N/A')}:{os.getenv('QDRANT_PORT', 'N/A') if os.getenv('QDRANT_HOST') != ':memory:' else ''}",
        "Graph File": os.path.basename(os.getenv("GRAPH_PATH", "N/A")),
        "Max Context Chunks": os.getenv("MAX_CONTEXT_CHUNKS", "N/A"),
        "Graph Neighborhood Depth": os.getenv("NEIGHBORHOOD_DEPTH", "N/A"),
        "LLM Num Ctx": os.getenv("LLM_NUM_CTX", "N/A"),
        "LLM Temp": os.getenv("LLM_TEMPERATURE", "N/A"),
        "Chat History Turns (LLM)": os.getenv("NUM_CHAT_HISTORY_TURNS", "N/A") # Added this
    }
    for key, value in config_details.items():
        st.sidebar.text(f"{key}: {value}")

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History", key="clear_chat_hist_btn"): # Changed key
        st.session_state.app_state["chat_messages"] = [
            {"role": "assistant", "content": "Chat history cleared. Ask me something new!"}
        ]
        st.rerun()

    if st.sidebar.button("Reload Application & Resources", key="reload_app_res_btn"): # Changed key
        st.cache_resource.clear()
        # Reset relevant parts of the app_state to trigger re-initialization sequence
        st.session_state.app_state["query_engine_initialized"] = False
        st.session_state.app_state["init_error_message"] = None
        # Optionally reset chat messages too, or keep them
        st.session_state.app_state["chat_messages"] = [
             {"role": "assistant", "content": "Application reloading... How can I help you explore the novel today?"}
        ]
        print("Streamlit: Reload Application & Resources button clicked. All @st.cache_resource caches cleared.")
        st.rerun()


# Run the app
if __name__ == "__main__":
    render_sidebar()
    run_app()