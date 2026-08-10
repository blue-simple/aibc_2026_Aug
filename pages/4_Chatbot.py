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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pypdf import PdfReader
from helper_functions.utility import check_password

# -------------------------------
# Utility Functions
# -------------------------------
def is_valid_http_url(candidate_url):
    """Check if a string is a valid HTTP/HTTPS URL."""
    parsed = urlparse(candidate_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def crawl_internal_pages(start_urls, max_pages):
    """Crawl internal links from multiple websites and return {"url", "text"} page records."""
    if isinstance(start_urls, str):
        start_urls = [start_urls]

    all_page_texts = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for start_url in start_urls:
        if not is_valid_http_url(start_url):
            continue

        target_domain = urlparse(start_url).netloc
        queue = deque([start_url])
        visited = set()
        page_texts = []

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
                page_texts.append({"url": current_url, "text": page_text})

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

        all_page_texts.extend(page_texts)

    return all_page_texts


@st.cache_resource
def build_vectorstore(_chunks):
    """Create a per-session in-memory vector store from document chunks."""
    if not _chunks:
        return None
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    return InMemoryVectorStore.from_documents(documents=_chunks, embedding=embeddings)


# Reliable reference sites used to fact-check the assistant's own answers.
TRUSTED_SOURCES = [
    "https://www.enablingguide.sg/",
    "https://www.nimh.nih.gov/",
    "https://www.autism.org.sg/",
]


@st.cache_resource(show_spinner="Indexing trusted reference sources (first run only)...")
def build_trusted_vectorstore():
    """Crawl the trusted reference sites once per app process and index them for fact-checking."""
    chunks = build_url_chunks(TRUSTED_SOURCES, max_pages=15)
    if not chunks:
        return None
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    return InMemoryVectorStore.from_documents(documents=chunks, embedding=embeddings)


def extend_resource_store(chunks):
    """Append newly loaded resource chunks to the session memory and rebuild the vector store."""
    if not chunks:
        return
    current_docs = st.session_state.get("resource_documents", [])
    current_docs.extend(chunks)
    st.session_state["resource_documents"] = current_docs
    st.session_state["vectorstore"] = build_vectorstore(current_docs)


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
#    """Build a system prompt that instructs the model to answer from the provided URL(s) or PDF(s) first and if that fails then "
#    "synthesizes a generic answer from Singapore context."""
#    return (
#        "You are a social worker assistant based in Singapore. "
#        "Your role is to provide clear, practical, and supportive answers about Autism Spectrum Disorder (ASD) and related resources. "
#
#        "Guidelines: "
#        "- Ground every response in the combined context from all loaded Singapore websites and uploaded documents. "
#        "- Ground every response in the combined context from all loaded Singapore websites and uploaded documents. "
#        "- Synthesize information across sources rather than focusing on a single page or file. "
#        "- If a detail is not supported by the indexed context, keep answers practical, general, and relevant to the resource landscape of Singapore. "
#        "- When appropriate, include links to the resources mentioned. "
#        "- If maps or location references are needed, reference OneMap. "
#        "- Avoid inventing facts or speculating beyond the provided context.\n\n"
#        f"Context:\n{context}"
#    )

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = [page["text"] for page in scraped_pages]
    metadatas = [{"source": page["url"]} for page in scraped_pages]
    return splitter.create_documents(texts, metadatas=metadatas)

# 2nd agent to validate against the 1st agent
def validate_agent_output(output: str, context: str = "") -> dict:
    """
    Validate the output of another agent.
    Returns a dictionary with validation results.
    """
    results = {
        "accuracy": "pass",
        "format": "pass",
        "tone": "pass",
        "safety": "pass",
        "issues": []
    }

    # Accuracy check: must reference context if provided
    if context and "Context:" not in output:
        results["accuracy"] = "fail"
        results["issues"].append("Missing context reference.")

    # Format check: must contain bullet points
    if "-" not in output and "•" not in output:
        results["format"] = "fail"
        results["issues"].append("No bullet points found.")

    # Tone check: must be professional
    if any(word in output.lower() for word in ["lol", "haha", "dumb"]):
        results["tone"] = "fail"
        results["issues"].append("Unprofessional tone detected.")

    # Safety check: avoid harmful language
    if any(word in output.lower() for word in ["kill", "hate"]):
        results["safety"] = "fail"
        results["issues"].append("Unsafe language detected.")

    return results


def build_fact_check_prompt(reply, trusted_context):
    """Build a prompt asking a grounding-check model to compare a reply against trusted excerpts."""
    return (
        "You are a fact-checking assistant reviewing an AI-generated answer about autism spectrum "
        "disorder (ASD) support in Singapore. Compare the answer's factual claims against the excerpts "
        "below, drawn from trusted reference sources (Enabling Guide SG, NIMH, Autism Association Singapore).\n\n"
        "Respond ONLY with a JSON object with these keys:\n"
        '  "verdict": one of "supported", "partially_supported", "unsupported", "contradicted"\n'
        '  "explanation": 1-2 sentences justifying the verdict\n'
        '  "unsupported_claims": a list of specific claims from the answer not backed by the excerpts '
        "(empty list if none)\n\n"
        f"Trusted source excerpts:\n{trusted_context}\n\n"
        f"AI-generated answer to check:\n{reply}"
    )


def validate_against_trusted_sources(client, model, reply, trusted_vectorstore, k=4):
    """Check an assistant reply for grounding against the trusted reference vector store."""
    if trusted_vectorstore is None:
        return {
            "verdict": "unavailable",
            "explanation": "Trusted reference index is not available.",
            "unsupported_claims": [],
            "sources": [],
        }

    trusted_context, docs = retrieve_context(trusted_vectorstore, reply, k=k)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_fact_check_prompt(reply, trusted_context)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(completion.choices[0].message.content)
    except Exception as e:
        result = {
            "verdict": "error",
            "explanation": f"Trusted-source check failed: {e}",
            "unsupported_claims": [],
        }

    result["sources"] = sorted({
        doc.metadata.get("source") for doc in docs if doc.metadata.get("source")
    })
    return result


def render_validation_badge(validation):
    """Render a compact verdict badge plus an expander with full validation detail."""
    trust = validation.get("trusted_source_check", {}) if validation else {}
    quality = validation.get("quality_checks", {}) if validation else {}

    badge_labels = {
        "supported": "✅ Verified against trusted sources",
        "partially_supported": "⚠️ Partially supported by trusted sources",
        "unsupported": "❓ Could not verify against trusted sources",
        "contradicted": "🚫 May conflict with trusted sources",
        "unavailable": "ℹ️ Trusted-source check unavailable",
        "error": "ℹ️ Trusted-source check unavailable",
    }
    st.caption(badge_labels.get(trust.get("verdict"), badge_labels["unavailable"]))

    quality_issues = quality.get("issues", [])
    if quality_issues:
        st.caption("⚠️ " + "; ".join(quality_issues))

    with st.expander("🔍 Validation details"):
        if trust.get("explanation"):
            st.write(trust["explanation"])
        if trust.get("unsupported_claims"):
            st.write("**Unsupported claims:**")
            for claim in trust["unsupported_claims"]:
                st.write(f"- {claim}")
        if trust.get("sources"):
            st.write("**Matched trusted sources:**")
            for source in trust["sources"]:
                st.write(f"- {source}")
        st.json(validation)


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
# Prompts
# -------------------------------
system_prompt_no_doc = (
    "You are a helpful Singapore-focused social worker assistant. "
    "Answer in a practical, generic way and avoid claiming source-specific details unless the user has loaded Singapore resource content. "
    "If resource context is missing, keep the response general and clear."
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

    # --- Sidebar controls ---
    with st.sidebar:
        # --- Input: allow comma-separated URLs --- <start>
        document_urls = st.text_area(
            "Enter one or more URLs (comma-separated):",
            value="",
            key="document_urls"
        )

        if document_urls.strip():
            # Split into list and clean whitespace
            urls = [u.strip() for u in document_urls.split(",") if u.strip()]

            # Validate each URL
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
                        st.warning(
                            "No readable text was found in the URL crawl. "
                            "Please try different sites or upload a PDF instead."
                        )
                    else:
                        st.session_state.loaded_urls = urls
                        extend_resource_store(url_chunks)
                        st.success(
                            f"Ready! Indexed {len(st.session_state.resource_documents)} combined resource chunks."
                        )
        else:
            st.info("Input one or more URLs to enable document Q&A.")
        # --- Input: allow comma-separated URLs --- <end>

        # --- Input: PDF upload --- <start>
        uploaded_file = st.file_uploader(
            "📄 Upload a Document",
            type=["pdf"],
        )

        if uploaded_file is not None:
            chunk_size = st.number_input("Chunk size", min_value=100, value=800, step=50)
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=100, step=10)
            chunks = load_and_split(uploaded_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            if not chunks:
                st.session_state.vectorstore = None
                st.warning(
                    "No readable text could be extracted from that PDF. "
                    "Please upload a different PDF or try a URL source instead."
                )
            else:
                extend_resource_store(chunks)
                st.success(
                    f"Ready! Indexed {len(st.session_state.resource_documents)} combined resource chunks."
                )
        else:
            st.info("Upload a PDF to enable document Q&A.")
        # --- Input: PDF upload --- <end>

        # Settings
        st.header("⚙️ Settings")
        st.session_state["system_prompt"] = st.text_area(
            "System Prompt", value=st.session_state["system_prompt"]
        )
        st.session_state["selected_model"] = st.selectbox(
            "Model", ["gpt-4o-mini", "gpt-4o"], index=0
        )
        st.session_state["temperature"] = st.slider(
            "Temperature", 0.0, 2.0, st.session_state["temperature"], 0.1
        )
        k_value = st.slider("Chunks to retrieve (k)", 1, 10, 4)

        # Conversation tools
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in st.session_state["messages"]
        )
        st.caption(f"Conversation characters: {len(conversation_text)}")
        st.download_button(
            "Download Chat",
            data=conversation_text + "\n",
            file_name="conversation.txt",
            mime="text/plain",
        )
        if st.button("Clear Conversation"):
            st.session_state["messages"] = []

    # --- Display chat history ---
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("validation"):
                render_validation_badge(message["validation"])

    # --- Chat input ---
    prompt = st.chat_input("Ask something...")
    if prompt:
        # save user message first
        st.session_state["messages"].append({"role": "user", "content": prompt})

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        # Build system message (no vectorstore logic here)
        system_message = system_prompt_no_doc

        api_messages = [
            {"role": "system", "content": system_message},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.get("messages", [])],
        ]

        response_chunks = []
        combined_validation = None
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

                with st.spinner("Checking against trusted sources..."):
                    quality_check = validate_agent_output(
                        full_reply, context=context if 'context' in locals() else ""
                    )
                    trusted_vectorstore = build_trusted_vectorstore()
                    trust_check = validate_against_trusted_sources(
                        client, st.session_state["selected_model"], full_reply, trusted_vectorstore, k=k_value
                    )

                combined_validation = {"quality_checks": quality_check, "trusted_source_check": trust_check}
                render_validation_badge(combined_validation)
        except Exception as e:
            st.error("Sorry, I couldn’t generate a response right now.")
            full_reply = f"Error: {e}"

        st.session_state["messages"].append(
            {"role": "assistant", "content": full_reply, "validation": combined_validation}
        )


# -------------------------------
# Run the chatbot page
# -------------------------------
chatbot_page()