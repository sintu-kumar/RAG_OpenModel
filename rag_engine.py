#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:55:21 2026

@author: sintu
"""

import os
import re
import tempfile
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Configuration ---
# We use the 1B small model. It fits in CPU RAM easily.
MODEL_ID = "google/gemma-3-4b-it" 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def clean_text(text: str) -> str:
    """Cleans raw PDF text."""
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class LocalRAGPipeline:
    def __init__(self):
        """
        Initializes the local ML models. 
        Note: This is heavy. We will cache this in the UI.
        """
        print("Loading Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        print(f"Loading LLM ({MODEL_ID})...")
        # Check for GPU, fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Inference Device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32, # float32 for CPU stability
            device_map=device,
            trust_remote_code=True
        )

        # Create a text-generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,  # Limit output length
            temperature=0.2,     # Low temp for factual RAG
            repetition_penalty=1.1,
            do_sample=True,
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

    def process_pdfs(self, uploaded_files) -> FAISS:
        documents = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                for doc in docs:
                    doc.page_content = clean_text(doc.page_content)
                documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            raise ValueError("No text extracted.")

        # Create Vector Store (Locally)
        # Since this is local, we don't need rate limits or batching!
        print("Generating embeddings (this may take a moment on CPU)...")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        return vector_store

    def get_answer(self, vector_store: FAISS, question: str) -> dict:
        # Retrieve Context
        docs = vector_store.similarity_search(question, k=3)
        context_text = "\n\n".join([d.page_content for d in docs])

        # Prompt Engineering for Gemma
        # Gemma uses <start_of_turn> tokens usually, but LangChain handles standard prompts well.
        # We'll use a standard instruction format.
        prompt_template = f"""<start_of_turn>user
            You are a helpful QnA assistant. Answer the question based ONLY on the context below.
            
            Context:
            {context_text}
            
            Question: 
            {question}
            
            Instructions:
                    1. If the answer is not in the context, say "I cannot find the answer in the provided documents."
                    2. Do not hallucinate.
                    3. Cite the page number if available. <end_of_turn>
            <start_of_turn>model
            """
        
        # Direct generation
        response = self.llm.invoke(prompt_template)
        
        # Cleanup: Sometimes models repeat the prompt. We return the raw generation.
        # The pipeline usually returns the full text, so we might need to strip the prompt 
        # depending on the pipeline version, but let's return raw for now.
        
        return {
            "answer": response,
            "source_documents": docs
        }