# Multi-Hop Question Generation using Retrieval-Augmented Generation (RAG)

## Overview

This project focuses on developing a Retrieval-Augmented Generation (RAG) system to automatically generate multi-hop open-ended questions and answer keys from OpenStax textbooks.

The system retrieves relevant text chunks from different chapters and uses a Large Language Model (LLM) to generate analytical questions that require connecting information from multiple sources.

## Research Objectives

- Implement cross-chapter retrieval to obtain relevant information from different chapters.
- Generate multi-hop open-ended questions and answer keys using RAG.
- Compare question generation using Non-Chain-of-Thought (Non-CoT) and Chain-of-Thought (CoT) prompting.
- Evaluate the quality of generated questions based on defined evaluation criteria.

## Technologies

- Python
- XML and lxml
- OpenAI Embeddings (`text-embedding-3-small`)
- ChromaDB
- Google Gemini API
- Retrieval-Augmented Generation (RAG)

## Project Structure

```text
multihop-rag-openstax/
├── data/
│   └── psychology/
├── src/
│   ├── embedding/
│   ├── generation/
│   ├── preprocessing/
│   ├── retrieval/
│   └── validation/
├── .gitignore
├── check_pilot_pair.py
├── download_psychology.py
├── inspect_partner_candidates.py
└── README.md
```

## Methodology

The project consists of the following main stages:

1. **Data Preprocessing**  
   Extracting and processing textbook content from OpenStax XML files into text chunks.

2. **Embedding and Storage**  
   Converting text chunks into vector embeddings and storing them in ChromaDB.

3. **Cross-Chapter Retrieval**  
   Retrieving relevant text chunks from different chapters based on semantic similarity and chapter distance.

4. **Question Generation**  
   Generating multi-hop questions and answer keys using Gemini with Non-CoT and CoT prompting strategies.

5. **Evaluation**  
   Evaluating retrieval results and the quality of generated questions.

## Project Status

This project is currently under development as part of an undergraduate thesis. The implementation and evaluation are being carried out progressively.
