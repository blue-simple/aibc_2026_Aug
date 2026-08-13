# A comprehensive explanation of the data flows and implementation details.
# A flowchart illustrating the process flow for each use case in the application. Each use case should have its own flowchart.

import streamlit as st
from PIL import Image
import io

# -------------------------------
# Streamlit App: Autism Chatbot Documentation
# -------------------------------

st.set_page_config(page_title="Autism Chatbot Documentation", layout="wide")

st.title("📘 Autism Chatbot Documentation")
st.write("Documents the **implementation details**, **data flows** and **process flowcharts** for the Autism Chatbot.")


# -------------------------------
# Implementation Details
# -------------------------------

# 1. Architecture
st.header("🔧 1. Architecture")
st.markdown("""
   - **Framework**: Built with **Streamlit** for the user interface.  
   - **LLM Integration**: Uses **OpenAI’s GPT models** (`gpt-4o-mini`, `gpt-4o`) for generating responses.  
   - **Vector Database**: Employs **Chroma** for persistent or session-only storage of document embeddings.  
   - **Helper Functions**: Encapsulated in a separate module (`helper_functions.utility`) for crawling, splitting, embedding, retrieval, tone/safety checks, and fact-checking.  
""")


# 2. Data Flow
st.header("📚 2. Data Flow")
data_flow_1 = """
   1. **User Input Layer**
      - The user either pastes URLs into the text area or uploads a PDF. These are the two entry points for knowledge that feeds the RAG system.
"""
data_flow_2 = """
   2. **Ingestion & Chunking**
      - For URLs, `build_url_chunks()` calls `crawl_internal_pages()`, which uses BeautifulSoup to scrape text from each page and follows internal links up to `max_pages`.  
      - The scraped text (with source URL metadata) is split into overlapping chunks by `RecursiveCharacterTextSplitter`.  
      - For PDFs, `load_and_split()` extracts text via `PdfReader` and similarly splits it into chunks.
"""
data_flow_3 = """
   3. **Vector Store (Chroma)**
      - Chunks flow into `extend_resource_store()`, which either creates a new Chroma vector store via `build_vectorstore()` or appends to the existing one via `.add_documents()`.  
      - Depending on the `persist_mode` toggle, Chroma either saves to disk (`chroma_db/`) or stays in memory for the session only.  
      - On app startup, the code attempts to reload an existing Chroma index from disk.
"""
data_flow_4 = """
   4. **Query & Retrieval**
      - When the user submits a chat message, `retrieve_context()` converts the query into an embedding.  
      - It retrieves the top *k* chunks from the vector store that are most semantically similar to the query.  
      - These chunks become the context injected into the RAG system prompt via `build_rag_system_prompt()`.
"""
data_flow_5 = """
   5. **LLM Response**
      - The system prompt (with injected context) plus the full conversation history is sent to OpenAI's chat completions API.  
      - The response streams back chunk by chunk and is rendered incrementally in the chat UI.
"""
data_flow_6 = """
   6. **Validation Layer**
      - After the response is generated, two checks run in parallel:  
      - `check_tone_and_safety()` does a quick keyword scan on the reply.  
      - `validate_against_trusted_sources()` retrieves relevant excerpts from a trusted vectorstore (built once at startup from `trusted_sources.txt` via `build_trusted_vectorstore()`) and sends both the answer and excerpts to the LLM as a judge.  
      - The judge returns a structured verdict, which is displayed in the **Validation Results** expander.
"""
st.markdown(data_flow_1)
col1, col2 = st.columns(2)
with col1:
   st.image("data/usecase1_crawlingURL.png", caption="URL Input & Crawling", width=300)
with col2:
   st.image("data/usecase2_uploadPDF.png", caption="PDF Upload", width=300)

st.markdown(data_flow_2)
st.markdown(data_flow_3)
st.markdown(data_flow_4)
st.image("data/usecase3_queryingChatbot.png", caption="Querying the Chatbot", width=600)
st.markdown(data_flow_5)
st.markdown(data_flow_6)
st.image("data/usecase4_factchecking.png", caption="Fact-Checking", width=200)


# 3. Intelligence Layer
st.header("🧠 3. Intelligence Layer")
st.markdown("""
   - **Embedding Model**: `text-embedding-3-small` for vector representation of text.  
   - **Retriever**: Configured with `k` (number of chunks to retrieve).  
   - **System Prompt**: Tailored for a **social worker assistant in Singapore**, ensuring supportive and professional tone.  
   - **Streaming Responses**: Uses OpenAI’s streaming API to display answers progressively.  
""")


# 4. Validation & Safety
st.header("✅ 4. Validation & Safety")
st.markdown("""
   - **Tone Check**: Flags unprofessional or unsafe language (`lol`, `haha`, `dumb`, `kill`, `hate`).  
   - **Fact-Checking**:  
      - Retrieves excerpts from trusted sources.  
      - Compares assistant’s answer against them using a secondary LLM.  
      - Verdict categories: supported, partially supported, unsupported, contradicted.  
   - **User Feedback**: Displays verdict with icons (✅, 🟡, ⚪, 🔴) and citations.  
""")


# 5. User Interface
st.header("🎨 5. User Interface")
st.markdown("""
   - **Main Page**: Title, welcome message, chat history, chat input.  
   - **Sidebar Controls**:  
      - Mode & Persistence toggle.  
      - Fact-check sources list.  
      - Resource input (URLs, PDFs).  
      - Settings (system prompt, model, temperature, retrieval size).  
      - Conversation tools (download chat, clear chat).  
   - **Expandable Sections**:  
      - Sources retrieved from user documents.  
      - Validation results (fact-check + tone/safety).  
""")


# 6. Persistence & State
st.header("🗄️ 6. Persistence & State")
st.markdown("""
   - **Session State Variables**:  
      - `messages`: Conversation history.  
      - `authenticated`: Login status.  
      - `valid_username`: User identity.  
      - `system_prompt`, `selected_model`, `temperature`: Model settings.  
      - `vectorstore`: Chroma index (persistent or session-only).  
      - `resource_documents`: Loaded chunks.  
      - `loaded_urls`: URLs added by user.  
   - **Persistence Mode**:  
      - Session-only (reset on restart).  
      - Persistent (saved to disk in `chroma_db`).  
""")


# 7. Deployment
st.header("🚀 7. Deployment")
st.markdown("""
   - **Streamlit App**: Runs as a web application.  
   - **Chroma Index**: Reloaded at startup if persistence is enabled.  
   - **Scalability**: Designed to handle multiple sessions with isolated state management.  
""")