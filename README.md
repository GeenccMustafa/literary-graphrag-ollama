# Literary GraphRAG Ollama Project

This project demonstrates a Retrieval Augmented Generation (RAG) system enhanced by a Knowledge Graph (KG) to answer questions about a literary work (default: "Crime and Punishment"). It utilizes:

*   **spaCy:** For Natural Language Processing (NLP) tasks like sentence segmentation and Named Entity Recognition (NER).
*   **NetworkX:** To build and analyze the character co-occurrence knowledge graph.
*   **Sentence-Transformers:** For generating text embeddings.
*   **Qdrant:** As a vector database to store and search text chunk embeddings.
*   **Ollama:** For running local Large Language Models (LLMs) like Llama 3 for answer generation and community summarization.
*   **Streamlit:** To create an interactive web application for querying the novel.
*   **Pyvis:** For generating interactive visualizations of the knowledge graph.

## Features

*   Constructs a knowledge graph of characters based on their co-occurrence in text chunks.
*   Stores text chunks from the novel, along with their embeddings and extracted entities, in a Qdrant vector database.
*   Performs community detection (Louvain algorithm) on the character graph.
*   Summarizes detected character communities using an LLM via Ollama.
*   Generates an interactive HTML visualization of the knowledge graph.
*   Provides a Streamlit web application for users to ask questions about the novel.
*   Implements a RAG pipeline that:
    *   Identifies entities in the user's query.
    *   Expands context using the knowledge graph.
    *   Retrieves relevant, chronologically sorted text chunks from Qdrant.
    *   Considers recent chat history for follow-up questions.
    *   Generates answers using an Ollama-hosted LLM based on the retrieved context and chat history.

## Project Structure
```
GRAPHRAG/
├── .env.example # Example environment file (users should copy to .env)
├── .gitignore # Specifies intentionally untracked files
├── 01_build_kg_and_vector_db.py # Script to build KG and Qdrant DB
├── 02_graph_analysis_and_community.py # Script for graph analysis & community summarization
├── 03_visualize_graph.py # Script to generate graph visualization HTML
├── main.py # Main orchestrator script to run the pipeline
├── query_engine.py # Core RAG query engine logic
├── requirements.txt # Python dependencies
├── streamlit_app.py # Streamlit UI application
└── data/
      └── Crime_and_Punishment.txt # Default novel (or place your novel here)
```
# Generated files like .gpickle, .pkl, .html will appear here after running scripts

## Setup Instructions

### 1. Prerequisites
*   **Python:** Version 3.9 or higher.
*   **Git:** For cloning the repository.
*   **Docker:** For running the Qdrant vector database. Install from [Docker's website](https://www.docker.com/products/docker-desktop/).
*   **Ollama:** For running local LLMs. Install from [ollama.com](https://ollama.com/).

### 2. Clone the Repository
```
git clone [git@github.com:GeenccMustafa/literary-graphrag-ollama.git](https://github.com/GeenccMustafa/literary-graphrag-ollama.git)
cd literary-graphrag-ollam
```
### 3. Set Up Python Environment

It's highly recommended to use a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
### 5. Configure Ollama

After installing Ollama, pull the LLM model specified in your .env file (default is llama3:8b):

```
ollama pull llama3:8b
```
Ensure the Ollama application/service is running in the background.

### 6. Set Up Environment Variables
Copy the example environment file and customize it:
```
cp .env.example .env
```
- Open the .env file and review/edit the variables:
- NOVEL_PATH: Path to your input novel text file (e.g., data/Crime_and_Punishment.txt).
- LLM_MODEL: The Ollama model tag to use (e.g., llama3:8b).
- Other variables usually have sensible defaults.

### 7. Prepare Novel Data
- Place your novel's text file (e.g., Crime_and_Punishment.txt) into the data/ directory if it's not already there.
- Ensure the NOVEL_PATH in your .env file correctly points to this file.

## Running the Application

The main.py script orchestrates the data processing and application launch.

1. Start Qdrant Docker Container:
Open a new terminal window/tab and run:
```
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant_literary qdrant/qdrant
```
