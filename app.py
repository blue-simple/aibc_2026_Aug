import tempfile
from pathlib import Path

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pypdf import PdfReader


@st.cache_resource
def build_vectorstore(_chunks):
    """Create an in-memory Chroma vector store from document chunks."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"],
    )
    return Chroma.from_documents(documents=_chunks, embedding=embeddings)


def load_and_split(uploaded_file, chunk_size=800, chunk_overlap=100):
    """Load a PDF from an uploaded file and split it into text chunks."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        reader = PdfReader(temp_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.create_documents([text])
    finally:
        Path(temp_path).unlink(missing_ok=True)


def build_vector_store(uploaded_file, persist_directory="./chroma_db"):
    """Create a Chroma vector store from an uploaded PDF."""
    documents = load_and_split(uploaded_file)
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def retrieve_context(vectorstore, query, k=4):
    """Retrieve relevant document chunks from a Chroma vector store."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    joined_context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return joined_context, docs


def build_rag_system_prompt(context):
    """Build a system prompt that instructs the model to answer from the provided context only."""
    return (
        "You are a helpful assistant. Answer ONLY from the provided context. "
        "If the information is not in the context, say: "
        "I couldn't find that information in the uploaded document.\n\n"
        f"Context:\n{context}"
    )


system_prompt_no_doc = "You are a helpful assistant."


st.title("Streamlit Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant."

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o-mini"

if "temperature" not in st.session_state:
    st.session_state.temperature = 1.0

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:
    uploaded_file = st.file_uploader(
        "📄 Upload a Document",
        type=["pdf"],
    )

    if uploaded_file is not None:
        chunk_size = st.number_input("Chunk size", min_value=100, value=800, step=50)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=100, step=10)
        chunks = load_and_split(uploaded_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.session_state.vectorstore = build_vectorstore(chunks)
        st.success(f"Ready! Indexed {len(chunks)} chunks.")
    else:
        st.info("Upload a PDF to enable document Q&A.")

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
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
        system_message = system_prompt_no_doc

    api_messages = [
        {"role": "system", "content": system_message},
        *st.session_state.messages,
    ]

    response_chunks = []

    try:
        def stream_response():
            for chunk in client.chat.completions.create(
                model=st.session_state.selected_model,
                messages=api_messages,
                stream=True,
                temperature=st.session_state.temperature,
            ):
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    response_chunks.append(delta)
                    yield delta

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.write_stream(stream_response())

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

    st.rerun()
