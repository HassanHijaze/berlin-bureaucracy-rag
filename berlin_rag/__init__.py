"""Berlin Bureaucracy RAG.

A retrieval-augmented question-answering system for international students
in Berlin, grounded in German statutes and official Berlin administrative
sources.

Modules:
    config      Settings, paths, and tunable constants.
    corpus      Loading documents and source-aware chunking.
    retrieval   BM25, embeddings and vector cache, query expansion, hybrid search.
    generation  The answer prompt, refusal detection, and AnswerGenerator.
    openai_client  The shared OpenAI client.
    cli         Command-line entry point (ask, evaluate).
    citations   Extraction and labelling of cited sources.
    evaluation  Evaluation harness for the question set.
    api         FastAPI app (main), endpoints (routes), models (schemas), daily limit.
"""
