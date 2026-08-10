from multiprocessing import context
import json
import tempfile
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_core.vectorstores import InMemoryVectorStore
# change out InMemoryVectorStore to Chroma for better performance and persistence
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pypdf import PdfReader
from helper_functions.utility import check_password

import os
# Disable Chroma telemetry
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# -------------------------------
# Utility Functions
# -------------------------------
def is_valid_http_url(candidate_url):
    """Check if a string is a valid HTTP/HTTPS URL."""
    parsed = urlparse(candidate_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def crawl_internal_pages(start_urls, max_pages):
    """Crawl internal links from multiple websites and return page records with their source URL."""
    if isinstance(start_urls, str):
        start_urls = [start_urls]

    #all_page_texts = []
    all_page_records = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for start_url in start_urls:
        if not is_valid_http_url(start_url):
            continue

        target_domain = urlparse(start_url).netloc
        queue = deque([start_url])
        visited = set()
        #page_texts = []
        page_records = []

        while queue and len(visited) < max_pages:
            current_url = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)

            try:
                response = requests.get(current_url, headers=headers, timeout=15)
                response.raise_for_status()
            except requests.RequestException:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            page_text = soup.get_text(separator="\n", strip=True)
            if page_text.strip():
                #page_texts.append(page_text)
                page_records.append({"url": current_url, "text": page_text})

            for anchor in soup.find_all("a", href=True):
                href = anchor["href"]
                next_url = urljoin(current_url, href)
                parsed_next = urlparse(next_url)
                if parsed_next.scheme not in {"http", "https"}:
                    continue
                if parsed_next.netloc != target_domain:
                    continue
                next_url = next_url.split("#", 1)[0]
                if next_url not in visited and next_url not in queue:
                    queue.append(next_url)

        #all_page_texts.extend(page_texts)
        all_page_records.extend(page_records)

    #return all_page_texts
    return all_page_records


# InMemoryVectorStore version
#@st.cache_resource
#def build_vectorstore(_chunks):
#    """Create a per-session in-memory vector store from document chunks."""
#    if not _chunks:
#        return None
#    embeddings = OpenAIEmbeddings(
#        model="text-embedding-3-small",
#        api_key=st.secrets["OPENAI_API_KEY"],
#    )
#    return InMemoryVectorStore.from_documents(documents=_chunks, embedding=embeddings)

# Chroma version to switch between persistent mode (saved across sessions) and session-only mode (ephemeral)
def build_vectorstore(_chunks, persist_mode=False):
    """Create a Chroma vector store, persistent or session-only based on toggle."""
    if not _chunks:
        return None
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    if persist_mode:
        return Chroma.from_documents(
            documents=_chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
    else:
        return Chroma.from_documents(
            documents=_chunks,
            embedding=embeddings
        )

 
# Pass the toggle value when building or adding documents
def extend_resource_store(chunks, persist_mode=False):
    """Append newly loaded resource chunks into Chroma, persistent or session-only."""
    if not chunks:
        return
    current_docs = st.session_state.get("resource_documents", [])
    current_docs.extend(chunks)
    st.session_state["resource_documents"] = current_docs

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"],
    )

    if "vectorstore" in st.session_state and st.session_state["vectorstore"] is not None:
        st.session_state["vectorstore"].add_documents(chunks)
    else:
        st.session_state["vectorstore"] = build_vectorstore(current_docs, persist_mode=persist_mode)


def load_and_split(uploaded_file, chunk_size=800, chunk_overlap=100):
    """Load a PDF from an uploaded file and split it into text chunks."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        reader = PdfReader(temp_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.create_documents([text])
    finally:
        Path(temp_path).unlink(missing_ok=True)


def retrieve_context(vectorstore, query, k=4):
    """Retrieve relevant document chunks from the vector store."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    joined_context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return joined_context, docs


#def build_rag_system_prompt(context):
def build_rag_system_prompt(context):
    """Build a system prompt that prioritizes answering from the provided URLs/PDFs,
    and falls back to general Singapore-relevant guidance only when the context is insufficient."""
    return (
        "You are a social worker assistant based in Singapore. "
        "Your role is to provide clear, practical, and supportive answers about Autism Spectrum Disorder (ASD) "
        "and related services, supports, and resources in Singapore.\n\n"

        "Guidelines:\n"
        "- Use the combined context from all loaded Singapore websites and uploaded documents as your primary source.\n"
        "- Synthesize information across sources; do not rely on a single page or file when multiple sources are available.\n"
        "- If the context does not support a detail, respond with practical, general information relevant to Singapore.\n"
        "- When appropriate, include links to the resources mentioned.\n"
        "- If location or navigation information is needed, reference OneMap where relevant.\n"
        "- Do not invent facts, infer unsupported details, or speculate beyond the provided context.\n"
        "- Keep the tone clear, supportive, and professional.\n\n"

        f"Context:\n{context}"
    )


def build_url_chunks(start_urls, chunk_size=800, chunk_overlap=100, max_pages=20):
    """
    Crawl the websites in start_urls and split the gathered text into retrievable chunks.
    start_urls: str or list of URLs
    chunk_size: size of each text chunk
    chunk_overlap: overlap between chunks
    max_pages: maximum pages to crawl per domain
    """
    # Allow single string input or list
    if isinstance(start_urls, str):
        start_urls = [start_urls]

    scraped_pages = crawl_internal_pages(start_urls, max_pages=max_pages)

    if not scraped_pages:
        return []

    # Extract text and metadata
    texts = [record["text"] for record in scraped_pages]
    metadatas = [{"source": record["url"]} for record in scraped_pages]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # Pass texts and metadata separately
    return splitter.create_documents(texts, metadatas=metadatas)



# -------------------------------
# Output validation against trusted sources
# -------------------------------
TRUSTED_SOURCE_URLS = [
    "https://www.enablingguide.sg/",
    "https://www.nimh.nih.gov/",
    "https://www.autism.org.sg/",
]


@st.cache_resource(show_spinner="Indexing trusted reference sources for fact-checking...")
def build_trusted_vectorstore():
    """Crawl the trusted ASD reference sites once and cache the resulting vector store for the app's lifetime."""
    chunks = build_url_chunks(TRUSTED_SOURCE_URLS, max_pages=15)
    if not chunks:
        return None
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    return InMemoryVectorStore.from_documents(documents=chunks, embedding=embeddings)


def check_tone_and_safety(output: str) -> dict:
    """Cheap keyword-based checks for tone and safety issues in an agent's output."""
    results = {"tone": "pass", "safety": "pass", "issues": []}

    if any(word in output.lower() for word in ["lol", "haha", "dumb"]):
        results["tone"] = "fail"
        results["issues"].append("Unprofessional tone detected.")

    if any(word in output.lower() for word in ["kill", "hate"]):
        results["safety"] = "fail"
        results["issues"].append("Unsafe language detected.")

    return results


def validate_against_trusted_sources(client, model, question, answer, vectorstore, k=4):
    """
    Fact-check an assistant answer against excerpts retrieved from trusted ASD reference sites,
    using a second LLM call as a judge. Returns a verdict, explanation, and cited excerpts.
    """
    if vectorstore is None:
        return {
            "verdict": "sources_unavailable",
            "explanation": "Trusted reference sources could not be loaded, so the answer could not be fact-checked.",
            "citations": [],
        }

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(f"{question}\n\n{answer}")
    if not docs:
        return {
            "verdict": "no_relevant_source",
            "explanation": "No relevant passages were found in the trusted sources for this topic.",
            "citations": [],
        }

    excerpt_block = "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in docs
    )

    judge_prompt = (
        "You are a fact-checking assistant. Compare the ASSISTANT ANSWER to the SOURCE EXCERPTS "
        "from trusted autism-related reference sites (Enabling Guide SG, NIMH, Autism Association Singapore). "
        "Judge whether the answer's factual claims are supported by the excerpts.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ASSISTANT ANSWER:\n{answer}\n\n"
        f"SOURCE EXCERPTS:\n{excerpt_block}\n\n"
        "Respond with strict JSON only, no markdown fences, in this exact shape:\n"
        '{"verdict": "supported" | "partially_supported" | "unsupported" | "contradicted", '
        '"explanation": "<one or two sentence justification>"}\n\n'
        "- supported: the answer's claims align with the excerpts.\n"
        "- partially_supported: some claims align, others aren't covered by the excerpts.\n"
        "- unsupported: the excerpts don't cover the claims (not necessarily wrong, just unverifiable here).\n"
        "- contradicted: the excerpts directly conflict with the answer."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "unsupported")
        explanation = parsed.get("explanation", "")
    except Exception as e:
        verdict = "check_failed"
        explanation = f"Validation could not be completed: {e}"

    citations = [
        {"source": doc.metadata.get("source", "unknown"), "excerpt": doc.page_content[:300]}
        for doc in docs
    ]

    return {"verdict": verdict, "explanation": explanation, "citations": citations}


# -------------------------------
# Session State Initialization
# -------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "valid_username" not in st.session_state:
    st.session_state["valid_username"] = None
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = "You are an experienced social worker assistant."
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "gpt-4o-mini"
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 1.0
#if "vectorstore" not in st.session_state:
#    st.session_state["vectorstore"] = None
#if "resource_documents" not in st.session_state:
#    st.session_state["resource_documents"] = []
#if "loaded_urls" not in st.session_state:
#    st.session_state["loaded_urls"] = []

# -------------------------------
# Reload Chroma index at startup
# -------------------------------
if "vectorstore" not in st.session_state:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    try:
        # Reload existing Chroma index if available
        st.session_state["vectorstore"] = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        st.session_state["resource_documents"] = []  # optional: track docs separately
        st.success("✅ Reloaded Chroma index from disk.")
    except Exception:
        st.session_state["vectorstore"] = None
        st.session_state["resource_documents"] = []
        st.info("ℹ️ No existing Chroma index found, starting fresh.")


# -------------------------------
# Prompts
# -------------------------------
system_prompt_no_doc = (
    "You are a social worker assistant based in Singapore. "
    "Your role is to provide clear, practical, and supportive answers about Autism Spectrum Disorder (ASD) "
    "and related services, supports, and resources in Singapore.\n\n"

    "Guidelines:\n"
    "- Use the combined context from all loaded Singapore websites and uploaded documents as your primary source.\n"
    "- Synthesize information across sources; do not rely on a single page or file when multiple sources are available.\n"
    "- If the context does not support a detail, respond with practical, general information relevant to Singapore.\n"
    "- When appropriate, include links to the resources mentioned.\n"
    "- If location or navigation information is needed, reference OneMap where relevant.\n"
    "- Do not invent facts, infer unsupported details, or speculate beyond the provided context.\n"
    "- Keep the tone clear, supportive, and professional.\n\n"
)

# -------------------------------
# Chatbot Page
# -------------------------------
def chatbot_page():
    st.title("Chatbot for Autism related queries")

    # 🚫 Block access if not logged in
    if not st.session_state.get("authenticated", False):
        st.warning("You must log in first on the Login page.")
        st.stop()

    st.write(f"Welcome, {st.session_state['valid_username']}!")

    # Crawl trusted sources once
    trusted_vectorstore = build_trusted_vectorstore()


    # --- Sidebar controls ---
    with st.sidebar:
        # --- Mode & Persistence ---
        st.header("📌 Mode")
        persist_mode = st.checkbox("Enable persistence across sessions", value=False)

        if st.session_state.get("vectorstore") is not None:
            if persist_mode:
                st.success("✅ Persistent RAG Mode — saved across sessions")
            else:
                st.info("🔵 Session-only RAG Mode — resets on restart")
        else:
            if persist_mode:
                st.warning("ℹ️ General Mode — persistence enabled, but no documents loaded yet")
            else:
                st.warning("ℹ️ General Mode — session-only, no documents loaded yet")

        # --- Fact-check Sources ---
        with st.expander("✅ Fact-check Sources", expanded=False):
            if trusted_vectorstore is not None:
                st.success("Trusted sources are loaded.")
            else:
                st.error("Trusted sources unavailable.")
            for trusted_url in TRUSTED_SOURCE_URLS:
                st.caption(f"🔗 {trusted_url}")

        # --- Document Input ---
        st.header("📄 Add Resources")
        with st.expander("🌐 Load URLs", expanded=False):
            document_urls = st.text_area("Enter URLs (comma-separated):", value="", key="document_urls")
            # --- Input: allow comma-separated URLs ---
            if document_urls.strip():
                urls = [u.strip() for u in document_urls.split(",") if u.strip()]
                invalid_urls = [u for u in urls if not is_valid_http_url(u)]
                if invalid_urls:
                    st.warning(f"Invalid URLs detected: {', '.join(invalid_urls)}")
                else:
                    st.success(f"Loaded {len(urls)} URL(s).")
                    if st.session_state.get("loaded_urls") != urls:
                        with st.spinner("Crawling internal pages..."):
                            url_chunks = build_url_chunks(urls)
                        if not url_chunks:
                            st.session_state.loaded_urls = []
                            st.session_state.vectorstore = None
                            st.warning("No readable text was found in the URL crawl.")
                        else:
                            st.session_state.loaded_urls = urls
                            extend_resource_store(url_chunks, persist_mode=persist_mode)
                            st.success(f"Ready! Indexed {len(st.session_state.resource_documents)} combined resource chunks.")
            else:
                st.info("Input one or more URLs to enable document Q&A.")

        
        with st.expander("📑 Upload PDF", expanded=False):
            uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
            # --- Input: PDF upload ---
            if uploaded_file is not None:
                chunk_size = st.number_input("Chunk size", min_value=100, value=800, step=50)
                chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=100, step=10)
                chunks = load_and_split(uploaded_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    st.session_state.vectorstore = None
                    st.warning("No readable text could be extracted from that PDF.")
                else:
                    extend_resource_store(chunks, persist_mode=persist_mode)
                    st.success(f"Ready! Indexed {len(st.session_state.resource_documents)} combined resource chunks.")
            else:
                st.info("Upload a PDF to enable document Q&A.")

        # --- Settings ---
        st.header("⚙️ Settings")
        with st.expander("Model & Prompt", expanded=False):
            st.session_state["system_prompt"] = st.text_area("System Prompt", value=st.session_state["system_prompt"])
            st.session_state["selected_model"] = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
            st.session_state["temperature"] = st.slider("Temperature", 0.0, 2.0, st.session_state["temperature"], 0.1)
            k_value = st.slider("Chunks to retrieve (k)", 1, 10, 4)

        # --- Conversation Tools ---
        st.header("💬 Conversation")
        with st.expander("Tools", expanded=False):
            conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state["messages"])
            st.caption(f"Conversation characters: {len(conversation_text)}")
            st.download_button("⬇️ Download Chat", data=conversation_text + "\n", file_name="conversation.txt", mime="text/plain")
            if st.button("🧹 Clear Conversation"):
                st.session_state["messages"] = []


    # --- Display chat history ---
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat input ---
    prompt = st.chat_input("Ask something...")
    if prompt:
        # Save user message first
        st.session_state["messages"].append({"role": "user", "content": prompt})

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        # Build system message BEFORE first response
        if st.session_state.get("vectorstore") is not None:
            context, retrieved_docs = retrieve_context(st.session_state["vectorstore"], prompt, k=k_value)
            system_message = build_rag_system_prompt(context)
        else:
            system_message = system_prompt_no_doc

        # Always build api_messages safely
        api_messages = [{"role": "system", "content": system_message}] + st.session_state.get("messages", [])

        response_chunks = []
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_placeholder = st.empty()
                    for chunk in client.chat.completions.create(
                        model=st.session_state["selected_model"],
                        messages=api_messages,
                        stream=True,
                        temperature=st.session_state["temperature"],
                    ):
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            response_chunks.append(delta)
                            response_placeholder.markdown("".join(response_chunks))
            full_reply = "".join(response_chunks)
        except Exception as e:
            st.error("Sorry, I couldn’t generate a response right now.")
            full_reply = f"Error: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": full_reply})

        # Display retrieved sources
        if st.session_state.get("vectorstore") is not None:
            with st.expander("🔍 View Sources from user provided information"):
                #for doc in retrieved_docs:
                #    st.write(doc.page_content)
                #    st.divider()
                for doc in retrieved_docs:
                    # Show the source URL if available
                    source_url = doc.metadata.get("source")
                    if source_url:
                        st.markdown(f"[View Source]({source_url})")

                    # Show the chunk text
                    st.write(doc.page_content)

                    st.divider()

        # Tone/safety checks
        keyword_checks = check_tone_and_safety(full_reply)

        # Fact-check against trusted sources
        with st.spinner("Fact-checking against trusted sources..."):
            source_check = validate_against_trusted_sources(
                client=client,
                model=st.session_state["selected_model"],
                question=prompt,
                answer=full_reply,
                vectorstore=trusted_vectorstore,
                k=k_value,
            )

        if source_check["verdict"] == "contradicted":
            st.error(f"⚠️ This answer appears to contradict trusted sources. {source_check['explanation']}")
        elif keyword_checks["safety"] == "fail":
            st.error("⚠️ This answer was flagged for unsafe language.")

        with st.expander("🔍 Validation Results"):
            st.write("**Source fact-check** (Enabling Guide SG, NIMH, Autism Association Singapore)")
            st.json(source_check)
            st.write("**Tone / safety checks**")
            st.json(keyword_checks)


# -------------------------------
# Run the chatbot page
# -------------------------------
chatbot_page()