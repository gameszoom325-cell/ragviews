"""Streamlit Web User Interface for RAG Eval Search Engine.

Supports interactive document upload (.txt, .pdf) with in-memory FAISS indexing
and seamless fallback to the pre-built default knowledge base.
"""

from __future__ import annotations

import io
import pypdf
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from config import get_config
from ingest import create_in_memory_vectorstore
from rag import get_vectorstore, query

# Configure page layout and metadata
st.set_page_config(
    page_title="RAG Eval Search Engine",
    page_icon="🔍",
    layout="wide",
)


def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from uploaded .txt or .pdf file."""
    filename = uploaded_file.name.lower()
    content = uploaded_file.read()

    if filename.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages_text: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text.strip())
        return "\n\n".join(pages_text)
    else:
        # Default text decoding
        return content.decode("utf-8", errors="replace")


# Layout: Header
st.title("🔍 RAG Eval Search Engine")
st.caption(
    "Production-grade grounded Question-Answering engine powered by LangChain, OpenAI, and FAISS."
)

# Sidebar: Document Upload & Knowledge Base Management
with st.sidebar:
    st.header("📁 Knowledge Base")
    st.markdown("Upload custom `.txt` or `.pdf` files, or use the default knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload Documents:",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Upload one or more text or PDF documents to query against.",
    )

    # Compute upload signature to detect changes
    current_signature = tuple((f.name, f.size) for f in uploaded_files) if uploaded_files else None

    if uploaded_files:
        # Ingest user documents into in-memory FAISS index if signature changed
        if st.session_state.get("uploaded_signature") != current_signature:
            with st.spinner("Processing uploaded files and building in-memory vector index..."):
                docs: List[Document] = []
                for f in uploaded_files:
                    f.seek(0)
                    extracted_text = extract_text_from_file(f)
                    if extracted_text.strip():
                        docs.append(
                            Document(page_content=extracted_text, metadata={"source": f.name})
                        )

                if docs:
                    cfg = get_config()
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=cfg.chunk_size,
                        chunk_overlap=cfg.chunk_overlap,
                    )
                    chunks = splitter.split_documents(docs)
                    vectorstore = create_in_memory_vectorstore(chunks)

                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["uploaded_signature"] = current_signature
                    st.session_state["is_custom"] = True
                    st.session_state["corpus_desc"] = (
                        f"Custom corpus ({len(uploaded_files)} file(s), {len(chunks)} chunks in-memory)"
                    )
                    st.success(f"Indexed {len(chunks)} chunks in memory!")
                else:
                    st.error("No readable text found in uploaded files.")
    else:
        # Fall back to existing pre-built FAISS index from ./faiss_index
        if "vectorstore" not in st.session_state or st.session_state.get("is_custom"):
            try:
                st.session_state["vectorstore"] = get_vectorstore()
                st.session_state["uploaded_signature"] = None
                st.session_state["is_custom"] = False
                st.session_state["corpus_desc"] = "Default Knowledge Base (pre-built ./faiss_index)"
            except Exception as exc:
                st.error(f"Failed to load default vector index: {exc}")

    st.markdown("---")
    st.markdown(f"**Active Index:**\n`{st.session_state.get('corpus_desc', 'Not Initialized')}`")

    if st.button("🔄 Reset to Default Knowledge Base"):
        st.session_state["vectorstore"] = get_vectorstore()
        st.session_state["uploaded_signature"] = None
        st.session_state["is_custom"] = False
        st.session_state["corpus_desc"] = "Default Knowledge Base (pre-built ./faiss_index)"
        st.rerun()

# Main Panel: Chat / Search Interface
col_main, col_info = st.columns([3, 1])

with col_main:
    # Only show question input if vectorstore exists in session_state
    if "vectorstore" in st.session_state and st.session_state["vectorstore"] is not None:
        user_question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the core architectural stages of RAG?",
            key="user_query_input",
        )

        submit_clicked = st.button("Submit", type="primary")

        if submit_clicked:
            if user_question and user_question.strip():
                with st.spinner("Retrieving knowledge chunks and generating grounded response..."):
                    result = query(
                        user_question.strip(),
                        vectorstore=st.session_state["vectorstore"],
                    )

                st.subheader("Answer")
                st.write(result.get("answer", "No answer generated."))

                with st.expander("View Sources"):
                    contexts = result.get("contexts", [])
                    if contexts:
                        for idx, chunk in enumerate(contexts, start=1):
                            st.markdown(f"**Source Chunk {idx}:**")
                            st.info(chunk)
                    else:
                        st.write("No source contexts available.")
            else:
                st.warning("Please enter a question before submitting.")
    else:
        st.info("Please upload documents or initialize the knowledge base to begin querying.")

with col_info:
    st.subheader("ℹ️ System Info")
    st.markdown("""
        - **Embedding Model**: `text-embedding-3-small`
        - **Language Model**: `gpt-4o-mini`
        - **Retrieval Strategy**: Top-4 Ranked FAISS
        - **Storage**: In-Memory (`st.session_state`)
        - **Grounding**: Strict Negative Fallback
        """)
