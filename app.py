"""Streamlit Web User Interface for RAG Eval Search Engine."""

from __future__ import annotations

import streamlit as st

from rag import query

# Configure page layout and metadata
st.set_page_config(
    page_title="RAG Eval Search Engine",
    page_icon="🔍",
    layout="centered",
)

st.title("RAG Eval Search Engine")
st.write("Ask questions grounded strictly in the verified knowledge base.")

user_question = st.text_input(
    "Ask a question:",
    placeholder="e.g., What is Retrieval-Augmented Generation?",
    key="user_query_input",
)

if st.button("Submit", type="primary"):
    if user_question and user_question.strip():
        with st.spinner("Retrieving knowledge chunks and generating grounded response..."):
            result = query(user_question.strip())

        st.subheader("Answer")
        st.write(result.get("answer", "No answer generated."))

        with st.expander("View Sources"):
            contexts = result.get("contexts", [])
            if contexts:
                for idx, chunk in enumerate(contexts, start=1):
                    st.markdown(f"**Source {idx}:**")
                    st.info(chunk)
            else:
                st.write("No source contexts available.")
    else:
        st.warning("Please enter a question to search.")
