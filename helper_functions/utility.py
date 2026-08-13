# helper_functions.py
import os
import re
import json
import tempfile
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import streamlit as st

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

    all_page_records = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for start_url in start_urls:
        if not is_valid_http_url(start_url):
            continue

        target_domain = urlparse(start_url).netloc
        queue = deque([start_url])
        visited = set()
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

        all_page_records.extend(page_records)

    return all_page_records


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


def load_trusted_sources(file_path="data/trusted_sources.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls
    except FileNotFoundError:
        return []

TRUSTED_SOURCE_URLS = load_trusted_sources()

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

    output_words = set(re.findall(r"\b\w+\b", output.lower()))

    #if any(word in output.lower() for word in ["lol", "haha", "dumb"]):
    if output_words & {"lol", "haha", "dumb"}:
        results["tone"] = "fail"
        results["issues"].append("Unprofessional tone detected.")

    #if any(word in output.lower() for word in ["kill", "hate"]):
    if output_words & {"kill", "hate"}:
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
