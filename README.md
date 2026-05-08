# ChronoRAG-Fa ⏳

> **A Time-Aware Retrieval-Augmented Generation (RAG) System for Persian News**

Status: **Work in Progress (Development Phase)**

## 🎯 Overview

Standard RAG systems rely heavily on semantic similarity, often failing to comprehend the temporal constraints of a user's query. This leads to "temporal hallucination" where Large Language Models return factually correct but temporally inaccurate responses. 

**ChronoRAG-Fa** aims to solve this limitation for the Persian language. This project is dedicated to building a decoupled, clean-architecture RAG system that integrates semantic-temporal embeddings and Small Language Models (SLMs) to accurately answer time-sensitive queries based on Persian news feeds.

## 🚀 Project Objectives

* **Mitigate Temporal Hallucination:** Design a system that explicitly understands and respects time constraints (e.g., "After 2022", "Before the elections") rather than treating dates as mere keywords.
* **Time-Aware Representation Learning:** Transition from basic metadata filtering to training a custom temporal encoder using contrastive learning. The model will be fine-tuned to distance temporally incorrect facts from semantically similar ones.
* **Clean & Decoupled Architecture:** Maintain strict separation of concerns between data ingestion, vector retrieval, and text generation to manage technical debt and ensure system scalability.
* **Resource-Efficient Generation:** Utilize Small Language Models (SLMs) tailored for Persian to maintain a minimal resource footprint while delivering high-quality, precise answers.

## 🏗️ Planned Architecture

The system is designed with a modular approach, divided into distinct decoupled components:

### 1. Data Ingestion & Preprocessing
* **Source:** Automated extraction from reliable Persian news RSS feeds.
* **Pipeline:** HTML/Ad stripping, Persian text normalization (spacing, character unification), and strict ISO 8601 timestamp extraction.
* **Storage:** Storing documents and metadata efficiently for vectorization.

### 2. Time-Aware Retriever & Vector DB
* **Database:** Utilizing **ChromaDB** for fast, developer-friendly vector storage and retrieval.
* **Embedding Model:** Initial phase using pre-trained models (e.g., Cohere/Jina), transitioning to a custom fine-tuned semantic-temporal encoder.
* **Retrieval Logic:** Combining semantic cosine similarity with a dynamic time-filtering scoring function.

### 3. Generator (SLM Integration)
* **Rewrite & Reason:** Structuring prompts to rewrite implicit temporal queries into explicit timestamps before generation.
* **Generation:** Feeding the retrieved context to a Persian-optimized SLM to generate concise, accurate answers without hallucinating out-of-context dates.

## 🗺️ Roadmap

- [ ] **Phase 1:** Setup Data Ingestion Pipeline & Text Normalization.
- [ ] **Phase 2:** Initialize ChromaDB & Implement Baseline Semantic Retriever (Metadata Filtering).
- [ ] **Phase 3:** Construct Contrastive Dataset (Positive/Negative temporal pairs).
- [ ] **Phase 4:** Fine-tune the Time-Aware Embedding Model.
- [ ] **Phase 5:** Integrate SLM & Design the Rewrite/Reason Prompting Pipeline.
- [ ] **Phase 6:** System Evaluation ($Precision@k$ for retrieval, hallucination metrics for generation).
