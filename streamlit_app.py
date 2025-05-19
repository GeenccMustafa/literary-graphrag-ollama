# graphrag_literary/streamlit_app.py
import streamlit as st
from query_engine import (
    LiteraryQueryEngine,
    QueryEngineInitializationError,
    GRAPH_PATH_ENV, QDRANT_HOST_ENV, QDRANT_PORT_ENV, QDRANT_COLLECTION_NAME_ENV,
    SPACY_MODEL_NAME_ENV, EMBEDDING_MODEL_NAME_ENV, LLM_MODEL_NAME_ENV,
    MAX_CONTEXT_CHUNKS_ENV, NEIGHBORHOOD_DEPTH_ENV, QDRANT_CLIENT_TIMEOUT_ENV,
    LLM_TEMPERATURE_ENV, LLM_NUM_CTX_ENV, NUM_CHAT_HISTORY_TURNS_ENV,
    COMMUNITY_SUMMARIES_PATH_ENV # New import
)
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# Load .env variables
load_dotenv()

# --- Page Configuration ---
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
                "content": "Hello! How can I help you explore the novel today? Tip: Ask about specific characters or their relationships for richer insights!",
                "rag_context": None # Add rag_context field
            }
        ],
    }

# --- Resource Initialization ---
@st.cache_resource
def get_query_engine_instance() -> Optional[LiteraryQueryEngine]:
    print("Attempting to initialize LiteraryQueryEngine via Streamlit...")
    try:
        engine = LiteraryQueryEngine(
            graph_path=GRAPH_PATH_ENV,
            community_summaries_path=COMMUNITY_SUMMARIES_PATH_ENV, # New argument
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
            num_chat_history_turns=NUM_CHAT_HISTORY_TURNS_ENV
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
        print(f"Streamlit: {error_msg} (Type: {type(e)})")
        return None

# --- UI Helper to Display RAG Context ---
def display_rag_context(rag_context_data: Optional[Dict[str, Any]]):
    if not rag_context_data:
        return

    with st.expander("How this answer was generated (RAG Context):", expanded=False):
        if rag_context_data.get("query_entities_extracted"):
            st.markdown(f"**Entities found in your query:** `{', '.join(rag_context_data['query_entities_extracted'])}`")
        else:
            st.markdown("**Entities found in your query:** None explicitly identified in graph.")

        if rag_context_data.get("graph_expanded_entities"):
            st.markdown(f"**Related entities from Knowledge Graph (neighborhood):** `{', '.join(rag_context_data['graph_expanded_entities'])}`")
        else:
            st.markdown("**Related entities from Knowledge Graph (neighborhood):** No graph expansion or no initial entities.")

        if rag_context_data.get("community_info"):
            st.markdown("**Relevant Character Communities:**")
            for comm_info in rag_context_data["community_info"]:
                nodes_str = ", ".join(comm_info.get('nodes', []))
                st.markdown(f"- **{comm_info.get('id', 'Unknown Community')}**: _{comm_info.get('summary', 'N/A')}_ (Members: {nodes_str})")
        else:
            st.markdown("**Relevant Character Communities:** None identified for this query.")
        
        st.markdown("**Retrieved context from the novel (chronological order):**")
        retrieved_chunks = rag_context_data.get("retrieved_chunks", [])
        if retrieved_chunks:
            for i, chunk_data in enumerate(retrieved_chunks):
                with st.container(border=True):
                    st.caption(f"Chunk (Original Index: {chunk_data.get('chunk_idx', 'N/A')}, Score: {chunk_data.get('score', 'N/A'):.4f})")
                    st.markdown(f"> {chunk_data.get('text', 'N/A')}")
                    if chunk_data.get('entities_in_chunk'):
                        st.caption(f"Entities in this chunk: {', '.join(chunk_data['entities_in_chunk'])}")
        else:
            st.markdown("No specific text chunks were retrieved to answer this question.")
        
        # For debugging or advanced users, show the exact context string fed to LLM
        # with st.expander("LLM Context String (Debug)", expanded=False):
        #     st.text(rag_context_data.get("final_context_str_for_llm", "Not available."))


# --- Main Application Logic ---
def run_app():
    st.title("📚 Literary GraphRAG")
    st.caption(
        f"Querying '{os.path.basename(os.getenv('NOVEL_PATH', 'your novel'))}' "
        f"using GraphRAG with LLM: {LLM_MODEL_NAME_ENV}"
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
        # Spinner will show while the cached function runs for the first time
        with st.spinner("Initializing Literary GraphRAG Engine... This may take a moment."):
            pass 
        
        if not st.session_state.app_state["query_engine_initialized"]:
             if not st.session_state.app_state["init_error_message"]:
                st.session_state.app_state["init_error_message"] = "Engine initialization did not complete successfully after first attempt."
             st.error(
                f"**Application Initialization Failed (Post-Spinner):**\n\n"
                f"{st.session_state.app_state['init_error_message']}"
             )
             return
    else:
        if not st.session_state.app_state["init_error_message"]: # only show success if no error during init
             st.success("Literary GraphRAG Engine is ready!")

    # Display chat messages
    for message in st.session_state.app_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("rag_context"):
                display_rag_context(message["rag_context"])


    if prompt := st.chat_input("Ask a question about the novel..."):
        if not query_engine:
            st.error("Query engine is not available. Cannot process the query.")
            return

        st.session_state.app_state["chat_messages"].append(
            {"role": "user", "content": prompt, "rag_context": None}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_data: Optional[Dict[str, Any]] = None
            with st.spinner("Thinking..."):
                try:
                    # Prepare chat history for the engine
                    # Engine expects a list of dicts: {"role": "user/assistant", "content": "..."}
                    # The engine itself will take the last N turns based on its config.
                    # The history should include the current user prompt as the last item if the engine
                    # is designed to process it as part of history (my _build_llm_prompt excludes it)
                    # so we pass the history *up to* the current prompt for the history segment,
                    # and the current prompt separately.
                    
                    # The `answer_query` method expects chat_history to include the current query
                    # as the last item IF it's meant to use it for context building before its own prompt logic.
                    # My `_build_llm_prompt` takes the current user_query separately and builds history from `chat_history[:-1]`.
                    # So, we should pass the current `st.session_state.app_state["chat_messages"]` as is.
                    
                    full_response_data = query_engine.answer_query(
                        prompt,
                        chat_history=st.session_state.app_state["chat_messages"] # Pass current history
                    )
                    
                    if full_response_data and "llm_answer" in full_response_data:
                        message_placeholder.markdown(full_response_data["llm_answer"])
                        if full_response_data: # Display RAG context for this assistant message
                             display_rag_context(full_response_data)
                    else:
                        message_placeholder.markdown("Error: No response received from the query engine.")
                
                except Exception as e:
                    err_msg = f"An unexpected error occurred while processing your query: {e}"
                    st.error(err_msg)
                    message_placeholder.markdown(err_msg)
                    # Log error for debugging
                    import traceback
                    traceback.print_exc()


            # Add assistant response (and its RAG context) to app's chat history
            assistant_response_content = "Error: Could not process request."
            rag_context_for_history = None

            if full_response_data and "llm_answer" in full_response_data:
                assistant_response_content = full_response_data["llm_answer"]
                rag_context_for_history = full_response_data # Store the whole dict
            
            st.session_state.app_state["chat_messages"].append(
                {"role": "assistant", "content": assistant_response_content, "rag_context": rag_context_for_history}
            )

# --- Sidebar Content ---
def render_sidebar():
    st.sidebar.header("About GraphRAG")
    st.sidebar.info(
        "This application uses Retrieval Augmented Generation (RAG) enhanced by a "
        "Knowledge Graph (KG). Key steps:\n"
        "1. A KG of characters and their co-occurrences is built from the novel.\n"
        "2. Text chunks are stored in a Qdrant vector database.\n"
        "3. When you ask a question, relevant entities are identified.\n"
        "4. The KG helps find related entities and concepts (neighborhood lookup).\n"
        "5. Character communities (if any) related to these entities are identified.\n" # New
        "6. Relevant text chunks are retrieved from Qdrant using this "
        "graph-enhanced context.\n"
        "7. An LLM (via Ollama) generates an answer based on your query and the "
        "retrieved context, potentially considering recent chat history."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tips for Better Results")
    st.sidebar.markdown("- Ask about specific **characters** (e.g., 'What is Raskolnikov thinking?').")
    st.sidebar.markdown("- Inquire about **relationships** (e.g., 'Describe the interactions between Sonia and Raskolnikov.').")
    st.sidebar.markdown("- Use **pronouns** in follow-up questions; the chat history helps maintain context.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Configuration")
    
    config_details = {
        "Novel": os.path.basename(os.getenv("NOVEL_PATH", "N/A")),
        "LLM": LLM_MODEL_NAME_ENV,
        "Embedding Model": EMBEDDING_MODEL_NAME_ENV,
        "SpaCy Model": SPACY_MODEL_NAME_ENV,
        "Qdrant Collection": QDRANT_COLLECTION_NAME_ENV,
        "Qdrant Host": f"{QDRANT_HOST_ENV}:{QDRANT_PORT_ENV if QDRANT_HOST_ENV != ':memory:' and QDRANT_PORT_ENV else ''}",
        "Graph File": os.path.basename(GRAPH_PATH_ENV),
        "Community Summaries": os.path.basename(COMMUNITY_SUMMARIES_PATH_ENV), # New
        "Max Context Chunks": MAX_CONTEXT_CHUNKS_ENV,
        "Graph Neighborhood Depth": NEIGHBORHOOD_DEPTH_ENV,
        "LLM Num Ctx": LLM_NUM_CTX_ENV,
        "LLM Temp": LLM_TEMPERATURE_ENV,
        "Chat History Turns (LLM)": NUM_CHAT_HISTORY_TURNS_ENV
    }
    for key, value in config_details.items():
        st.sidebar.text(f"{key}: {value}")

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History", key="clear_chat_hist_btn"):
        st.session_state.app_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": "Chat history cleared. Ask me something new! Tip: Ask about specific characters or their relationships for richer insights!",
                "rag_context": None
            }
        ]
        st.rerun()

    if st.sidebar.button("Reload Application & Resources", key="reload_app_res_btn"):
        st.cache_resource.clear()
        st.session_state.app_state["query_engine_initialized"] = False
        st.session_state.app_state["init_error_message"] = None
        st.session_state.app_state["chat_messages"] = [
             {
                 "role": "assistant",
                 "content": "Application reloading... How can I help you explore the novel today? Tip: Ask about specific characters or their relationships for richer insights!",
                 "rag_context": None
             }
        ]
        print("Streamlit: Reload Application & Resources button clicked. All @st.cache_resource caches cleared.")
        st.rerun()

# Run the app
if __name__ == "__main__":
    render_sidebar()
    run_app()