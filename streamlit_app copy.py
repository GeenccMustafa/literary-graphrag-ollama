# graphrag_literary/streamlit_app.py
import streamlit as st
from query_engine import (
    LiteraryQueryEngine,
    QueryEngineInitializationError,
    # Import the globally parsed ENV variables if you want to pass them explicitly
    # Or just rely on the constructor defaults in LiteraryQueryEngine
    GRAPH_PATH_ENV, QDRANT_HOST_ENV, QDRANT_PORT_ENV, QDRANT_COLLECTION_NAME_ENV,
    SPACY_MODEL_NAME_ENV, EMBEDDING_MODEL_NAME_ENV, LLM_MODEL_NAME_ENV,
    MAX_CONTEXT_CHUNKS_ENV, NEIGHBORHOOD_DEPTH_ENV, QDRANT_CLIENT_TIMEOUT_ENV,
    LLM_TEMPERATURE_ENV, LLM_NUM_CTX_ENV # Make sure this is imported
)
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

load_dotenv()

st.set_page_config(
    page_title="Literary GraphRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

@st.cache_resource
def get_query_engine_instance() -> Optional[LiteraryQueryEngine]:
    print("Attempting to initialize LiteraryQueryEngine via Streamlit...")
    try:
        # OPTION A: Rely on defaults defined in LiteraryQueryEngine constructor
        # engine = LiteraryQueryEngine()

        # OPTION B: Explicitly pass the already parsed ENV variables
        # This is more verbose but makes it clear what streamlit_app.py thinks it's using.
        engine = LiteraryQueryEngine(
            graph_path=GRAPH_PATH_ENV, # from query_engine
            qdrant_host=QDRANT_HOST_ENV, # from query_engine
            qdrant_port=QDRANT_PORT_ENV, # from query_engine
            qdrant_collection_name=QDRANT_COLLECTION_NAME_ENV, # from query_engine
            spacy_model_name=SPACY_MODEL_NAME_ENV, # from query_engine
            embedding_model_name=EMBEDDING_MODEL_NAME_ENV, # from query_engine
            llm_model_name=LLM_MODEL_NAME_ENV, # from query_engine
            max_context_chunks=MAX_CONTEXT_CHUNKS_ENV, # from query_engine (already int)
            neighborhood_depth=NEIGHBORHOOD_DEPTH_ENV, # from query_engine (already int)
            qdrant_client_timeout=QDRANT_CLIENT_TIMEOUT_ENV, # from query_engine (already int)
            llm_temperature=LLM_TEMPERATURE_ENV, # from query_engine (already float)
            llm_num_ctx=LLM_NUM_CTX_ENV # from query_engine (already int)
        )
        # NO MORE os.getenv or int() calls here for these variables!

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
        print(f"Streamlit: {error_msg}")
        return None

# ... (rest of streamlit_app.py: run_app, render_sidebar, etc. can remain the same) ...

def run_app():
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
        # This case should ideally be covered by the spinner/error above,
        # but as a fallback or if caching behaves unexpectedly.
        st.warning(
            "Resources are still loading or failed to load. Please wait or refresh."
        )
        # Show a spinner while waiting for the first run of get_query_engine_instance
        with st.spinner("Initializing Literary GraphRAG Engine... This may take a moment."):
            pass # The actual loading happens in the first call to get_query_engine_instance
        
        # Re-check after get_query_engine_instance has definitely run once (due to @st.cache_resource)
        if not st.session_state.app_state["query_engine_initialized"]:
            # If still not initialized after the first attempt, show error (it would have been set by the cached function)
             if not st.session_state.app_state["init_error_message"]: # If no specific error, provide a generic one.
                st.session_state.app_state["init_error_message"] = "Engine initialization did not complete successfully."
             st.error(
                f"**Application Initialization Failed:**\n\n"
                f"{st.session_state.app_state['init_error_message']}"
             )
             return
    else:
        st.success("Literary GraphRAG Engine is ready!")
        for message in st.session_state.app_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the novel..."):
            st.session_state.app_state["chat_messages"].append(
                {"role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        full_response = query_engine.answer_query(prompt) # query_engine should not be None here
                        message_placeholder.markdown(full_response)
                    except Exception as e:
                        err_msg = (
                            f"An unexpected error occurred while processing your query: {e}"
                        )
                        st.error(err_msg)
                        message_placeholder.markdown(err_msg)
                        full_response = "Error processing request." 

            st.session_state.app_state["chat_messages"].append(
                {"role": "assistant", "content": full_response}
            )

def render_sidebar():
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
        "retrieved context."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Configuration")
    
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
        "LLM Num Ctx": os.getenv("LLM_NUM_CTX", "N/A"), # Display raw from .env for info
        "LLM Temp": os.getenv("LLM_TEMPERATURE", "N/A")
    }
    for key, value in config_details.items():
        st.sidebar.text(f"{key}: {value}")

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History", key="clear_chat"):
        st.session_state.app_state["chat_messages"] = [
            {"role": "assistant", "content": "Chat history cleared. Ask me something new!"}
        ]
        st.rerun() 

    if st.sidebar.button("Reload Application & Resources", key="reload_app_resources"):
        st.cache_resource.clear() 
        st.session_state.app_state["query_engine_initialized"] = False
        st.session_state.app_state["init_error_message"] = None
        st.session_state.app_state["chat_messages"] = [
             {"role": "assistant", "content": "Application reloading... How can I help you explore the novel today?"}
        ]
        print("Streamlit: Reload Application & Resources button clicked. Cache cleared.")
        st.rerun()

if __name__ == "__main__":
    render_sidebar()
    run_app()