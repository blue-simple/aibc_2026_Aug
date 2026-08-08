import tempfile
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pypdf import PdfReader

# check password
def check_password(password):
    # Example: simple password check
    if password != "Secret123":
        return False
    if len(password) < 9:
        return False
    if not any(char.isdigit() for char in password):
        return False
    return True

def is_valid_http_url(candidate_url):
    """Return True when the input is an HTTP or HTTPS URL."""
    parsed = urlparse(candidate_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# define multiple pages to crawl
def crawl_internal_pages(start_urls, max_pages=10):
    """
    Crawl internal links from multiple websites and return plain-text page content.
    start_urls: list of URLs to start crawling from
    max_pages: maximum pages to crawl per domain
    """
    if isinstance(start_urls, str):
        start_urls = [start_urls]  # allow single URL input too

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
                page_texts.append(page_text)

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
    return InMemoryVectorStore.from_documents(
        documents=_chunks,
        embedding=embeddings,
    )


def extend_resource_store(chunks):
    """Append newly loaded resource chunks to the session memory and rebuild the vector store."""
    if not chunks:
        return

    current_docs = st.session_state.get("resource_documents", [])
    current_docs.extend(chunks)
    st.session_state.resource_documents = current_docs
    st.session_state.vectorstore = build_vectorstore(current_docs)


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


def build_vector_store(uploaded_file):
    """Create a per-session in-memory vector store from an uploaded PDF."""
    documents = load_and_split(uploaded_file)
    if not documents:
        return None

    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    return InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )


def retrieve_context(vectorstore, query, k=4):
    """Retrieve relevant document chunks from a Chroma vector store."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    joined_context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return joined_context, docs


def build_rag_system_prompt(context):
#   """Build a system prompt that instructs the model to answer from the provided context only."""
    """Build a system prompt that instructs the model to answer from the provided context first and if that fails then "
    "synthesizes a generic answer across the combined Singapore resource context."""
    return (
#        "You are a document-grounded assistant. Answer ONLY from the provided context. "
#        "Never use outside knowledge or make up missing facts. "
#        "If the information is not in the context, say: "
#        "I couldn't find that information in the loaded URL or uploaded PDF.\n\n"
        "You are a Singapore-resource grounded social worker assistant. "
        "Answer by synthesizing the combined context from all loaded Singapore websites and uploaded documents. "
        "Provide a generic, helpful response based on the overall resource set rather than only the current page or file. "
        "If a detail is not clearly supported by the indexed context, say so briefly and avoid inventing facts. "
        "Keep the answer practical, general, and grounded in the available Singapore resource context and provide the links to the resources. "
        "If maps are needed, take reference from onemap.\n\n"
        f"Context:\n{context}"
    )


#def build_url_chunks(start_url, chunk_size=800, chunk_overlap=100):
#    """Crawl a website and split the gathered text into retrievable chunks."""
#    scraped_pages = crawl_internal_pages(start_url)
#    #urls = [
#    #"https://www.enablingguide.sg/disability-info/autism",
#    #"https://www.autism.org.sg/",
#    #"https://www.asdcollaborative.sg/",
#    #"https://familiesforlife.sg/pages/FFLPArticle/Young-Children-Supporting-Resources-Special-Children"
#    #]
#    #scraped_pages = crawl_internal_pages(urls, max_pages=30)
#    if not scraped_pages:
#        return []#
#
#    splitter = RecursiveCharacterTextSplitter(
#        chunk_size=chunk_size,
#        chunk_overlap=chunk_overlap,
#    )
#    return splitter.create_documents(scraped_pages)

def build_url_chunks(start_urls, chunk_size=800, chunk_overlap=100, max_pages=30):
    """
    Crawl one or more websites and split the gathered text into retrievable chunks.
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
    return splitter.create_documents(scraped_pages)

system_prompt_no_doc = (
    "You are a helpful Singapore-focused social worker assistant. "
    "Answer in a practical, generic way and avoid claiming source-specific details unless the user has loaded Singapore resource content. "
    "If resource context is missing, keep the response general and clear."
)


def chatbot_page():
    st.title("Chatbot for Autism related queries")

    # ✅ Initialize session state keys safely
    if "messages" not in st.session_state:
        st.session_state.messages = []   # start with an empty list

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are an experienced social worker assistant."

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o-mini"

    if "temperature" not in st.session_state:
        st.session_state.temperature = 1.0

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "resource_documents" not in st.session_state:
        st.session_state.resource_documents = []

    if "loaded_url" not in st.session_state:
        st.session_state.loaded_url = ""

    if "loaded_urls" not in st.session_state:
        st.session_state.loaded_urls = []

# --- Sidebar Controls --- <start>
    with st.sidebar:
        # Sidebar input: allow comma-separated URLs, to contextual from a few websites
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

        # Sidebar input: allow file upload
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

        # Sidebar input: System setting
        st.header("⚙️ Settings")
        system_prompt_value = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            key="system_prompt_widget",
        )
        st.session_state.system_prompt = system_prompt_value

        st.session_state.selected_model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o"],
            index=["gpt-4o-mini", "gpt-4o"].index(st.session_state.selected_model),
        )

        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
        )

        k_value = st.slider("Chunks to retrieve (k)", 1, 10, 4)

        conversation_text = "\n".join(
            f"{message['role']}: {message['content']}" for message in st.session_state.messages
        )
        st.caption(f"Conversation characters: {len(conversation_text)}")

        st.download_button(
            label="Download Chat",
            data=conversation_text + "\n",
            file_name="conversation.txt",
            mime="text/plain",
        )

        if st.button("Clear Conversation"):
            st.session_state.messages = []
# --- Sidebar Controls --- <end>

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input ---
    prompt = st.chat_input("Ask something...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        if st.session_state.get("vectorstore") is not None:
            retrieved_context, retrieved_docs = retrieve_context(
                st.session_state.vectorstore,
                prompt,
                k=k_value,
            )
            system_message = build_rag_system_prompt(retrieved_context)
        else:
            retrieved_context = ""
            retrieved_docs = []
            system_message = system_prompt_no_doc#
    #        full_reply = "Please load a URL or upload a PDF first so I can answer only from that source."
    #        st.session_state.messages.append({"role": "assistant", "content": full_reply})
    #        with st.chat_message("assistant"):
    #            st.markdown(full_reply)
    #        st.stop()
    #        with st.chat_message("assistant"):

        api_messages = [
            {"role": "system", "content": system_message},
            *st.session_state.messages,
        ]

        response_chunks = []

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_placeholder = st.empty()
                    for chunk in client.chat.completions.create(
                        model=st.session_state.selected_model,
                        messages=api_messages,
                        stream=True,
                        temperature=st.session_state.temperature,
                    ):
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            response_chunks.append(delta)
                            response_placeholder.markdown("".join(response_chunks))

            full_reply = "".join(response_chunks)
        except Exception as e:
            print(f"OpenAI API error: {e}")
            st.error("Sorry, I couldn’t generate a response right now.")
            full_reply = "Sorry, I couldn’t generate a response right now."

        st.session_state.messages.append({"role": "assistant", "content": full_reply})

        if st.session_state.get("vectorstore") is not None:
            with st.chat_message("assistant"):
                with st.expander("🔍 View Sources"):
                    for doc in retrieved_docs:
                        st.write(doc.page_content)
                        st.divider()

#run chatbot page
chatbot_page()
