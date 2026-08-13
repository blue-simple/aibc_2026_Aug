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
st.write("This Streamlit app documents the **data flows**, **implementation details**, and **process flowcharts** for the Autism Chatbot RAG system.")

# -------------------------------
# Tabs for Navigation
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Data Flows", "⚙️ Implementation Details", "🔄 Flowcharts"])

# -------------------------------
# Data Flows
# -------------------------------
with tab1:
    st.header("📊 Data Flow Overview")
    st.markdown("### Use Case A: Crawling URLs")
    # use case 1: URL Input and Crawling
    file_path="data/dataflow.png"
    img = Image.open(file_path)
    st.image(img, caption="Data Flow", width=700)
    data_flows_text = """
    1. **Input Sources**
       - 🌐 User-provided URLs (crawled internally)
       - 📑 Uploaded PDFs (extracted text)
       - 🔗 Trusted sources (predefined autism-related sites)

    2. **Processing**
       - 🧹 Clean HTML (remove scripts/styles)
       - ✂️ Split text into chunks
       - 🧠 Embed chunks with OpenAIEmbeddings

    3. **Storage**
       - 💾 Chroma vectorstore (persistent or session-only)
       - 📦 InMemoryVectorStore for trusted sources

    4. **Retrieval**
       - 🔍 Query matched against embeddings
       - 📜 Relevant chunks retrieved

    5. **Response Generation**
       - 📝 Build system prompt
       - 🤖 OpenAI model generates response
       - ✅ Fact-check against trusted sources
       - ⚠️ Tone/safety checks applied

    6. **Output**
       - 💬 Assistant reply
       - 📌 Sources displayed
       - 📊 Validation results shown
    """
    st.markdown(data_flows_text)

# -------------------------------
# Implementation Details
# -------------------------------
with tab2:
    st.header("⚙️ Implementation Details")
    implementation_text = """
    - **Crawling**: Uses `requests` + `BeautifulSoup` to fetch internal links.
    - **Vectorstore Management**:
      - `build_vectorstore()` creates Chroma store.
      - `extend_resource_store()` appends chunks.
      - `build_trusted_vectorstore()` caches trusted sources.
    - **PDF Handling**: `load_and_split()` extracts text with `PdfReader`, splits into chunks.
    - **Retrieval**: `retrieve_context()` queries vectorstore for top-k chunks.
    - **Prompt Engineering**: `build_rag_system_prompt()` ensures Singapore ASD context.
    - **Validation**:
      - `check_tone_and_safety()` flags unsafe language.
      - `validate_against_trusted_sources()` compares answers with trusted excerpts.
    - **Session State**: Streamlit `st.session_state` manages authentication, resources, persistence, and chat history.
    """
    st.markdown(implementation_text)

# -------------------------------
# Flowcharts
# -------------------------------
with tab3:
   st.header("🔄 Flowcharts")

   st.markdown("### Use Case 1: Crawling URLs")
   st.image("data/usecase1_crawlingURL.png", caption="🌐 Use Case 1: URL Input & Crawling", width=500)
   st.markdown("---")  # Optional divider between images
   st.markdown("### Use Case 2: Uploading PDFs")
   st.image("data/usecase2_uploadPDF.png", caption="📑 Use Case 2: PDF Upload", width=500)
   st.markdown("---")  # Optional divider between images
   st.markdown("### Use Case 3: Querying the Chatbot")
   st.image("data/usecase3_queryingChatbot.png", caption="💬 Use Case 3: Querying the Chatbot", use_column_width=True)
   st.markdown("---")  # Optional divider between images   
   st.markdown("### Use Case 4: Fact-Checking")
   st.image("data/usecase4_factchecking.png", caption="✅ Use Case 4: Fact-Checking", width=300)

#    st.markdown("### Use Case 1: Crawling URLs")
#    st.image("data/usecase1_crawlingURL.png", caption="🌐 Use Case 1: URL Input & Crawling", use_column_width=True)
#    st.markdown("### Use Case 2: Uploading PDFs")
#    st.image("data/usecase2_uploadPDF.png", caption="📑 Use Case 2: PDF Upload", use_column_width=True)
#    st.markdown("### Use Case 3: Querying the Chatbot")
#    st.image("data/usecase3_queryingChatbot.png", caption="💬 Use Case 3: Querying the Chatbot", use_column_width=True)
#    st.markdown("### Use Case 4: Fact-Checking")
#    st.image("data/usecase4_factchecking.png", caption="✅ Use Case 4: Fact-Checking", use_column_width=True)