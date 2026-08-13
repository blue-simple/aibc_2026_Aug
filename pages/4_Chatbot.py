import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Import helper functions
from helper_functions.utility import (
    is_valid_http_url,
    build_url_chunks,
    extend_resource_store,
    load_and_split,
    retrieve_context,
    build_rag_system_prompt,
    build_trusted_vectorstore,
    check_tone_and_safety,
    validate_against_trusted_sources,
    load_trusted_sources,
)

TRUSTED_SOURCE_URLS = load_trusted_sources()

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
                st.info("Input one or more URLs to enable Q&A.")

        
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

        # --- Display retrieved sources ---
        if st.session_state.get("vectorstore") is not None:
            # Count how many sources were retrieved
            source_count = len(retrieved_docs) if retrieved_docs else 0
            with st.expander(f"🔍 View Sources from user provided information ({source_count})"):
                if retrieved_docs:
                    for doc in retrieved_docs:
                        # Show the source URL inline at the top with a preview snippet
                        source_url = doc.metadata.get("source")
                        if source_url:
                            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                            st.markdown(f"**Source:** [{source_url}]({source_url})")
                            st.caption(preview)

                        # Full chunk text below
                        st.write(doc.page_content)
                        st.divider()
                else:
                    st.info("No relevant sources were retrieved for this query.")


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

            verdict_labels = {
                "supported": "✅ Supported by trusted sources",
                "partially_supported": "🟡 Partially supported by trusted sources",
                "unsupported": "⚪ Not covered by trusted sources",
                "contradicted": "🔴 Contradicted by trusted sources",
                "no_relevant_source": "⚪ No relevant trusted source found",
                "sources_unavailable": "⚪ Trusted sources unavailable",
                "check_failed": "⚠️ Fact-check could not be completed",
            }
            st.markdown(verdict_labels.get(source_check["verdict"], source_check["verdict"]))
            if source_check.get("explanation"):
                st.caption(source_check["explanation"])
            cited_sources = sorted({c["source"] for c in source_check.get("citations", [])})
            if cited_sources:
                st.caption("Sources checked: " + ", ".join(cited_sources))

            st.write("**Tone / safety checks**")
            if keyword_checks["issues"]:
                for issue in keyword_checks["issues"]:
                    st.caption(f"⚠️ {issue}")
            else:
                st.caption("✅ No tone or safety issues detected.")


# -------------------------------
# Run the chatbot page
# -------------------------------
chatbot_page()