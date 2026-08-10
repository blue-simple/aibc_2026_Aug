# A comprehensive explanation of the data flows and implementation details.
# A flowchart illustrating the process flow for each use case in the application. Each use case should have its own flowchart.
# Refer to the sample here for examples of flowcharts and methodology (Slides 13, 14, and 15)

import streamlit as st
from PIL import Image
from pathlib import Path
import graphviz

st.set_page_config(page_title="Application Methodology", layout="wide")

st.title("📊 Application Methodology and Flowcharts")


st.header("User Interaction Flow")

img_dir = Path("/mount/src/aibc_2026_aug/data")
img_path = img_dir / "method.jpg"
img = Image.open(img_path)
st.image(img, use_column_width=True)

# --- Diagram for the RAG Application ---
st.header("3️Master Lifecycle Diagram")
st.markdown("""
This diagram summarizes the entire Retrieval‑Augmented Generation (RAG) workflow — from data ingestion to final response display.
It visually connects all five use cases into one continuous lifecycle.
""")

img_path = img_dir / master_lifecycle_rag.png
master_img = Image.open(img_path)
st.image(master_img, caption="Master Lifecycle Diagram for RAG Application", use_column_width=True)
with open(img_path, "rb") as f:
    st.download_button("⬇️ Download Master Lifecycle Diagram", f, file_name="master_lifecycle_rag.png", mime="image/png")

# -------------------------------
# (1) Comprehensive Explanation
# -------------------------------
st.header("1️⃣ Comprehensive Explanation of Data Flows & Implementation Details")

st.markdown("""
### 🔹 Overview
This application integrates document ingestion, retrieval‑augmented generation (RAG), and validation workflows to support autism‑related queries.  
It combines **PDF and URL ingestion**, **vectorstore indexing**, **contextual prompt generation**, and **fact‑checking** against trusted sources.

### 🔹 Data Flow Components
| Component | Description |
|------------|-------------|
| **Document ingestion** | PDFs are uploaded and processed by `load_and_split()`. URLs are crawled using `crawl_internal_pages()` and chunked via `build_url_chunks()`. |
| **Vectorstore (Chroma)** | Built using `build_vectorstore()` with optional persistence. Documents are added through `extend_resource_store()`. |
| **Context retrieval** | `retrieve_context()` fetches relevant chunks for a user query. |
| **Prompt construction** | `build_rag_system_prompt()` integrates retrieved context into the system prompt. |
| **Chat interaction** | User messages are stored in `st.session_state["messages"]`. The OpenAI API streams responses. |
| **Validation & safety** | `check_tone_and_safety()` flags tone issues; `validate_against_trusted_sources()` fact‑checks answers. |
| **Sidebar controls** | Allow toggling persistence, uploading documents, adjusting model settings, and managing conversations. |

### 🔹 Data Flow Summary
1. User provides input (URL, PDF, or chat prompt).  
2. Data ingestion → text extracted, chunked, stored in vectorstore.  
3. Session state updated.  
4. Query execution → context retrieved.  
5. System prompt built.  
6. OpenAI API called.  
7. Validation checks.  
8. Response displayed.
""")

# Master flowchart
master_flow = graphviz.Digraph()
master_flow.attr(rankdir="TB", size="8,8")
master_flow.node("A", "User Input: URL / PDF / Chat", shape="box", style="filled", fillcolor="#FFD966")
master_flow.node("B", "Data Ingestion", shape="box", style="filled", fillcolor="#9FC5E8")
master_flow.node("C", "Text Extraction & Chunking", shape="box", style="filled", fillcolor="#9FC5E8")
master_flow.node("D", "Vectorstore Build/Update", shape="box", style="filled", fillcolor="#F9CB9C")
master_flow.node("E", "Session State Updated", shape="box", style="filled", fillcolor="#F9CB9C")
master_flow.node("F", "Chat Prompt?", shape="diamond", style="filled", fillcolor="#CFE2F3")
master_flow.node("G", "Retrieve Context from Vectorstore", shape="box", style="filled", fillcolor="#9FC5E8")
master_flow.node("H", "Fallback to system_prompt_no_doc", shape="box", style="filled", fillcolor="#9FC5E8")
master_flow.node("I", "Build RAG System Prompt", shape="box", style="filled", fillcolor="#F9CB9C")
master_flow.node("J", "OpenAI API Call", shape="box", style="filled", fillcolor="#F9CB9C")
master_flow.node("K", "Stream Response to UI", shape="box", style="filled", fillcolor="#F9CB9C")
master_flow.node("L", "Run Tone & Safety Checks", shape="box", style="filled", fillcolor="#9FC5E8")
master_flow.node("M", "Validate Against Trusted Sources", shape="box", style="filled", fillcolor="#9FC5E8")
master_flow.node("N", "Warnings if needed", shape="box", style="filled", fillcolor="#F9CB9C")
master_flow.node("O", "Final Assistant Response with Sources", shape="box", style="filled", fillcolor="#93C47D")

master_flow.edges(["AB", "BC", "CD", "DE", "EF"])
master_flow.edge("F", "G", label="Yes")
master_flow.edge("F", "H", label="No")
master_flow.edges(["GI", "HI", "IJ", "JK", "KL", "KM", "LN", "MN", "NO"])

st.graphviz_chart(master_flow)
st.download_button("⬇️ Download Master Flowchart (PNG)", data=master_flow.source, file_name="master_flowchart.dot", mime="text/plain")

# -------------------------------
# (2) Flowcharts for Each Use Case
# -------------------------------
st.header("2️⃣ Flowcharts for Each Use Case")

def show_flowchart(title, dot_source, filename):
    st.subheader(title)
    st.graphviz_chart(dot_source)
    st.download_button(f"⬇️ Download {title} (PNG)", data=dot_source, file_name=filename, mime="text/plain")

# Use Case 1
pdf_flow = """
flowchart TD
    A[User uploads PDF] --> B[load_and_split()]
    B --> C[Extract text with PdfReader]
    C --> D[Split text into chunks]
    D --> E[extend_resource_store()]
    E --> F[Add chunks to vectorstore]
    F --> G[Session state updated]
    G --> H[Ready for RAG queries]
"""
show_flowchart("📄 Use Case 1: PDF Upload & Indexing", pdf_flow, "usecase1_pdf_upload.dot")

# Use Case 2
url_flow = """
flowchart TD
    A[User enters URLs] --> B[crawl_internal_pages()]
    B --> C[Scrape text & links]
    C --> D[build_url_chunks()]
    D --> E[Split text into chunks]
    E --> F[extend_resource_store()]
    F --> G[Add chunks to vectorstore]
    G --> H[Session state updated]
    H --> I[Ready for RAG queries]
"""
show_flowchart("🌐 Use Case 2: URL Input & Crawling", url_flow, "usecase2_url_crawling.dot")

# Use Case 3
chat_flow = """
flowchart TD
    A[User enters chat prompt] --> B[Save to session_state["messages"]]
    B --> C{Vectorstore exists?}
    C -->|Yes| D[retrieve_context()]
    D --> E[build_rag_system_prompt(context)]
    C -->|No| F[system_prompt_no_doc]
    E --> G[Build api_messages]
    F --> G
    G --> H[OpenAI API call]
    H --> I[Stream response to UI]
    I --> J[Run tone/safety checks]
    I --> K[Validate against trusted sources]
    J --> L[Display warnings if needed]
    K --> L
    L --> M[Assistant response shown with sources]
"""
show_flowchart("💬 Use Case 3: Chat Query with RAG", chat_flow, "usecase3_chat_query.dot")

# Use Case 4
validation_flow = """
flowchart TD
    A[Assistant generates response] --> B[check_tone_and_safety()]
    A --> C[validate_against_trusted_sources()]
    B --> D[Flag tone/safety issues]
    C --> E[Compare answer vs trusted excerpts]
    E --> F[Return verdict: supported/unsupported/contradicted]
    D --> G[Display warnings/errors]
    F --> G[Show validation results in UI]
"""
show_flowchart("✅ Use Case 4: Validation & Fact‑Checking", validation_flow, "usecase4_validation.dot")

# Use Case 5
sidebar_flow = """
flowchart TD
    A[User interacts with sidebar] --> B[Persistence toggle]
    A --> C[Enter URLs]
    A --> D[Upload PDF]
    A --> E[Adjust settings: model, temperature, k]
    A --> F[Conversation tools: download, clear]
    B --> G[Vectorstore built persistent/session-only]
    C --> G
    D --> G
    E --> H[Update session_state]
    F --> I[Update conversation state]
"""
show_flowchart("⚙️ Use Case 5: Sidebar Controls", sidebar_flow, "usecase5_sidebar.dot")

st.success("✅ Comprehensive explanation, all flowcharts, and download options are ready. Run with `streamlit run methodology_with_downloads.py` to view interactively.")
