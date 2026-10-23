"""Berlin Bureaucracy RAG.

A retrieval-augmented question-answering system for international students
in Berlin, grounded in German statutes and official Berlin administrative
sources.

Modules:
    rag         Corpus loading, BM25, embeddings, retrieval, and answering.
    chunking    Source-aware chunking for statutes and Berlin pages.
    prompts     Prompt templates for query expansion and answer generation.
    citations   Extraction and labelling of cited sources.
    evaluation  Evaluation harness for the question set.
    api         FastAPI service and web interface.
"""