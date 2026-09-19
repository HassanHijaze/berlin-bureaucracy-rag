# Berlin Bureaucracy RAG

A Retrieval-Augmented Generation (RAG) system for answering questions about
German registration, residence law, student employment rights, and health
insurance for international students in Berlin.

The system retrieves information from German statutes and official Berlin
administrative sources, combines semantic search with BM25 retrieval, and
generates source-grounded answers with citations.

If the available sources do not support an answer, the system is instructed to
decline rather than provide unsupported information.

> **Disclaimer:** This project is for informational and evaluation purposes only
> and does not constitute legal advice.

---

## Live Demo

Try the deployed application directly:

**[Open Berlin Bureaucracy RAG](https://berlin-bureaucracy-rag-1.onrender.com)**

![Web interface](docs/interface.png)

---

## Evaluation

The system was manually evaluated on **100 realistic student questions** against
the underlying source material.

| Result | Count |
|---|---:|
| Correct | 76 |
| Wrong | 2 |
| Partial | 1 |
| Stale source | 1 |
| Declined | 20 |

The stale-source case is reported separately from model errors. The generated
answer accurately reflected the retrieved document, but the document itself
contained an outdated annual value. This is therefore treated as a corpus
freshness issue rather than a retrieval or generation failure.

The declined questions are largely intentional. Topics such as tax
identification numbers, bank accounts, the Rundfunkbeitrag, and pension
contributions are deliberately outside the corpus.

Including unsupported questions makes refusal behavior measurable: the system
should decline when the required information is missing rather than generate an
unsupported answer.

Correctness is not scored automatically. Every generated answer was manually
checked against the source text it cited.

---

## Development & Lessons Learned

The project was developed iteratively, with retrieval and generation failures
used to guide each improvement.

- **From semantic search to hybrid retrieval:** semantic similarity alone did not
  consistently capture exact German legal terminology, so BM25 was added and
  combined with semantic retrieval.

- **German legal query expansion:** English and everyday-language questions often
  differed from the terminology used in statutes and administrative documents.
  An LLM-based German query rewrite was therefore added to improve lexical
  retrieval.

- **Source-aware chunking:** generic text splitting was replaced with different
  strategies for statutory provisions and Berlin administrative pages to preserve
  meaningful legal and procedural context.

- **Stricter grounding and refusal:** testing showed that retrieving a relevant
  source does not guarantee a correct answer. The generation prompt was refined
  to check legal categories, preserve distinctions such as `must` and `may`, and
  refuse when the required rule is not supported by the retrieved sources.

- **Failure analysis shaped the evaluation:** errors were separated into retrieval
  failures, generation errors, unsupported questions, and stale-source problems.
  This made it possible to distinguish model failures from limitations of the
  underlying corpus.

The main lesson was that improving a RAG system requires more than increasing
retrieval accuracy: source quality, chunking, legal scope, refusal behaviour, and
generation constraints all affect the reliability of the final answer.


## Corpus

The corpus contains German legislation and official administrative information
relevant to international students in Berlin.

| Folder | Content |
|---|---|
| `bmg_sections/` | Bundesmeldegesetz, 13 sections in German and English |
| `aufenthg_sections/` | Aufenthaltsgesetz, 6 sections in German and English |
| `sgb_sections/` | SGB V, 7 sections covering student health insurance |
| `aufenthv_sections/` | Aufenthaltsverordnung, 6 sections covering visa procedures and fees |
| `vab_sections/` | Berlin LEA administrative guidance |
| `*.md` | Berlin Service Portal, LEA, health authority, and insurer sources |

The corpus currently contains **68 documents** and approximately **614 chunks**
of up to roughly 750 characters.

It focuses on four areas:

- address registration
- student residence permits
- employment rights during study
- student health insurance

The corpus intentionally prioritizes legal obligations and authoritative
administrative information over broader practical guidance that may become
outdated quickly.

---

## Architecture


The system uses a hybrid retrieval pipeline that combines semantic similarity
with BM25 keyword search. This helps capture both the meaning of natural-language
questions and the exact German legal terminology used in the source documents.

Retrieved results are then passed to GPT-5.6 for source-grounded generation,
with explicit instructions to cite the evidence used and refuse unsupported
questions.

```text
User Question
     │
     ├── Semantic embedding
     │      text-embedding-3-small
     │
     │      → semantic similarity score
     │
     └── German legal query expansion
            │
            ├── original question
            └── rewritten German legal keywords

            → BM25 score
              60% original query
              40% rewritten query

                     │
                     ▼
              Score normalization
                     │
                     ▼
              Weighted fusion
             70% semantic
             30% BM25
                     │
                     ▼
              Top 12 chunks
          max. 2 chunks per source
                     │
                     ▼
                Generation
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Grounded answer        Refusal when
   with citations         unsupported