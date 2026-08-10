# A comprehensive explanation of the data flows and implementation details.
# A flowchart illustrating the process flow for each use case in the application. Each use case should have its own flowchart.
# Refer to the sample here for examples of flowcharts and methodology (Slides 13, 14, and 15)

#import streamlit as st
#from PIL import Image
#from pathlib import Path

#st.title("Methodology")

#img_path = Path("/mount/src/aibc_2026_aug/data/method.jpg")
#img = Image.open(img_path)
#st.image(img, use_column_width=True)

import streamlit as st

st.set_page_config(page_title="Methodology", layout="wide")

st.title("📊 Application Methodology")

# -------------------------------
# (1) Comprehensive Explanation
# -------------------------------
st.header("1️⃣ Comprehensive Explanation of Data Flows & Implementation Details")

st.markdown("""
### 🔹 Core Components
- **Document ingestion**
  - PDFs → `load_and_split()` extracts text → split into chunks → stored in vectorstore.
  - URLs → `crawl_internal_pages()` scrapes → `build_url_chunks()` chunks text → stored in vectorstore.

- **Vectorstore (Chroma)**
  - Built via `build_vectorstore()` with persistence toggle.
  - New chunks added via `extend_resource_store()`.
  - Retrieval via `retrieve_context()`.

- **System prompt construction**
  - If vectorstore exists → `build_rag_system_prompt(context)`.
  - Else → fallback to `system_prompt_no_doc`.

- **Chat interaction**
  - User input saved in `st.session_state["messages"]`.
  - Context retrieved if available.
  - OpenAI API called with system + user messages.
  - Response streamed back to UI.

- **Validation & safety**
  - `check_tone_and_safety()` keyword checks.
  - `validate_against_trusted_sources()` fact-checks against Enabling Guide SG, NIMH, Autism Association SG.

- **Sidebar controls**
  - Mode indicator (Persistent vs Session-only).
  - Fact-check sources status.
  - Document input (URLs, PDFs).
  - Settings (model, temperature, retrieval k).
  - Conversation tools (download, clear chat).
""")

st.markdown("""
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

# -------------------------------
# (2) Flowcharts
# -------------------------------
st.header("2️⃣ Flowcharts for Each Use Case")

st.subheader("📄 Use Case 1: PDF Upload & Indexing")
st.graphviz_chart("""
flowchart TD
    A[User uploads PDF] --> B[load_and_split()]
    B --> C[Extract text with PdfReader]
    C --> D[Split text into chunks]
    D --> E[extend_resource_store()]
    E --> F[Add chunks to vectorstore]
    F --> G[Session state updated]
    G --> H[Ready for RAG queries]
""")

st.subheader("🌐 Use Case 2: URL Input & Crawling")
st.graphviz_chart("""
flowchart TD
    A[User enters URLs] --> B[crawl_internal_pages()]
    B --> C[Scrape text & links]
    C --> D[build_url_chunks()]
    D --> E[Split text into chunks]
    E --> F[extend_resource_store()]
    F --> G[Add chunks to vectorstore]
    G --> H[Session state updated]
    H --> I[Ready for RAG queries]
""")

st.subheader("💬 Use Case 3: Chat Query with RAG")
st.graphviz_chart("""
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
""")

st.subheader("✅ Use Case 4: Validation & Fact-Checking")
st.graphviz_chart("""
flowchart TD
    A[Assistant generates response] --> B[check_tone_and_safety()]
    A --> C[validate_against_trusted_sources()]
    B --> D[Flag tone/safety issues]
    C --> E[Compare answer vs trusted excerpts]
    E --> F[Return verdict: supported/unsupported/contradicted]
    D --> G[Display warnings/errors]
    F --> G[Show validation results in UI]
""")

st.subheader("⚙️ Use Case 5: Sidebar Controls")
st.graphviz_chart("""
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
""")

st.success("✅ Documentation and flowcharts are ready. Run with `streamlit run methodology.py` to view interactively.")