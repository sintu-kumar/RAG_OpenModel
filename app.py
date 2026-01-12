#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:12:33 2026

@author: sintu
"""

import streamlit as st
import os
import time
# No asyncio needed for local Hugging Face
from rag_engine import LocalRAGPipeline

st.set_page_config(page_title="Gemma Local Doc Chat", layout="wide")

# --- CACHING IS CRITICAL FOR LOCAL MODELS ---
@st.cache_resource(show_spinner="Loading Gemma-3-4b model locally...")
def load_rag_engine():
    """
    Load the Gemma-3-4b model only once.
    """
    return LocalRAGPipeline()

def main():
    st.title("🦙 Chat with Docs (Local Gemma 3 Model)")
    st.markdown("Running **Gemma-3-4b-it** locally on your machine.")

    # Load Model (Cached)
    try:
        rag = load_rag_engine()
        st.success("System Ready: Model loaded into memory.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        st.info("Using local CPU/GPU resources.")
        
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        process_btn = st.button("Process Documents")

    # --- Session State ---
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # --- Processing ---
    if process_btn and uploaded_files:
        with st.spinner("Reading PDFs and creating local embeddings..."):
            try:
                # We use the cached rag instance's methods
                st.session_state.vector_store = rag.process_pdfs(uploaded_files)
                st.success(f"Processed {len(uploaded_files)} file(s)!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # --- Q&A ---
    if st.session_state.vector_store:
        st.divider()
        question = st.text_input("Ask a question about your documents:")
        
        if st.button("Ask"):
            if not question:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Gemma is thinking... (this depends on your CPU speed)"):
                    try:
                        result = rag.get_answer(st.session_state.vector_store, question)
                        
                        st.markdown("### 🤖 Gemma Answer")
                        # Basic cleanup to remove the prompt from the answer if the model repeats it
                        clean_answer = result["answer"]
                        if "<start_of_turn>model" in clean_answer:
                             clean_answer = clean_answer.split("<start_of_turn>model")[-1]
                        
                        st.write(clean_answer)
                        
                        st.markdown("### 📚 Context Used")
                        for i, doc in enumerate(result["source_documents"]):
                            with st.expander(f"Source Snippet {i+1}"):
                                st.write(doc.page_content)
                                
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()