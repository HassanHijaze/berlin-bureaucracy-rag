# Berlin Bureaucracy RAG

A retrieval-augmented question-answering system for international students in Berlin.
It answers questions on registration, residence permits, working rights, and health
insurance from German statutes and official Berlin sources, cites its sources, and
declines when they do not support an answer.

**[Live demo](https://berlin-bureaucracy-rag-1.onrender.com)** · Not legal advice.

![Web interface](docs/interface.png)

## Results

Evaluated on 100 realistic student questions, each answer checked manually against its cited source.

| Correct | Correctly declined | Wrong | Partial | Stale source |
|:---:|:---:|:---:|:---:|:---:|
| 76 | 20 | 2 | 1 | 1 |

Out-of-scope questions are included on purpose to measure refusal behaviour.
With 100 questions, the margin of error is roughly ±9 points.

## How it works

```text
question → semantic search + BM25 (with German legal query rewrite)
         → weighted fusion (70 / 30) → top 12 chunks
         → GPT-5.6 generation with citations, or refusal
```

## Limitations

- The system covers a limited corpus and does not attempt to answer every administrative question relevant to living in Germany. Topics outside it are intentionally refused.
- Source information may become outdated, so annual amounts and thresholds require regular corpus updates.
- Answers can differ between English and German phrasings of the same question.

## Quick start

```bash
git clone https://github.com/HassanHijaze/berlin-bureaucracy-rag.git
cd berlin-bureaucracy-rag
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Create a file named `API.env` in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

Run the web app at http://127.0.0.1:8000:

```bash
uvicorn berlin_rag.api:app --port 8000
```

Or use the command line:

```bash
python -m berlin_rag.rag ask "How long do I have to register my address?"
python -m berlin_rag.rag          # run the evaluation set
```

With Docker:

```bash
docker build -t berlin-bureaucracy-rag .
docker run --rm --env-file API.env -p 8000:8000 berlin-bureaucracy-rag
```

## Testing

```bash
pytest            # offline tests, no API key needed
pytest -m live    # tests that call the OpenAI API
```

## Tech stack

Python · FastAPI · OpenAI · BM25 · pytest · Docker · Render

## License

MIT. The documents in `data/corpus/` are reproduced from public German government,
Berlin state, and insurer sources for evaluation purposes and remain the property
of their publishers.