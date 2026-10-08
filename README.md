# AI Study Assistant — RAG-Based Generative AI Workflow Orchestrator

A personal chatbot that answers questions about AI/ML research, grounded in a library of landmark papers
(Transformer, BERT, RAG, Self-RAG, LoRA, GPT-3, Llama 2, ReAct, Whisper, YOLO and more).
Every answer comes with its sources, and the system checks whether the answer is actually supported by them.

**Stack:** Python · LangGraph · Pinecone · FastAPI · Docker · AWS (deployment guide)
**Cost:** $0. Groq free tier (LLM), Pinecone Starter (vector DB), local embeddings (fastembed).

## How it works
```
Documents (PDF/DOCX/MD/TXT)
   -> load -> chunk (800 chars, 120 overlap) -> embed (BAAI/bge-small-en-v1.5, local) -> upsert (Pinecone, deterministic IDs)

Question -> LangGraph workflow:
   route -> retrieve -> grade_documents -> generate -> check_grounding -> answer + sources + trace
              ^              |
              +-- rewrite_query (up to 2 retries when nothing relevant is found)
   (greetings / questions about the assistant go route -> direct_answer, no documents used)
```

- **Agentic routing:** an LLM router decides whether a question needs retrieval.
- **Relevance grading:** retrieved chunks are filtered by an LLM grader; if nothing is relevant the query is rewritten and retried.
- **Grounding check:** the final answer is verified against the retrieved context and flagged if unsupported.
- **Resumable ingestion:** batched, low-memory, skips files already indexed (manifest), safe to re-run.

| Part | File |
|---|---|
| Config | `app/config.py` |
| Pinecone + embeddings | `app/vectorstore.py` |
| Ingestion / chunking | `app/ingestion.py`, `scripts/ingest.py` |
| LangGraph workflow | `app/graph.py` |
| REST API + chat page | `app/main.py`, `app/static/chat.html` |
| Evaluation | `eval/evaluate.py`, `eval/dataset.json` |
| Deployment | `Dockerfile`, `docker-compose.yml`, `deploy/aws.md` |

## Quick start
You need two free API keys: [Groq](https://console.groq.com) and [Pinecone](https://www.pinecone.io). Python 3.12 recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env  -> then add your keys
mkdir -p data/docs                 # put your PDFs here (not included in this repo)
python scripts/ingest.py           # index the documents
uvicorn app.main:app --reload      # open http://localhost:8000
```

`LLM_MODEL` must be a model your Groq key can access. List yours with:
`curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`.

## API
```bash
curl -F "file=@paper.pdf" localhost:8000/ingest
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "How does Self-RAG differ from standard RAG?"}'
```
Response: `answer`, `grounded`, `sources[]`, and a `trace` of the graph steps taken.

## Evaluation
`eval/dataset.json` holds 20 questions with reference answers and expected source papers.
`python eval/evaluate.py` writes `eval/report.json` with:

- **Hit@k / MRR:** did retrieval surface the right paper, and how high
- **Answer accuracy:** LLM-as-judge vs the reference answer
- **Groundedness:** share of answers supported by the retrieved context

| Metric | Result |
|---|---|
| Hit@5 | _fill in_ |
| MRR | _fill in_ |
| Answer accuracy | _fill in_ |
| Groundedness | _fill in_ |

Knowledge base used for these numbers: 15 AI papers (about 2,400 chunks). Tune `CHUNK_SIZE`, `CHUNK_OVERLAP` and `TOP_K` and compare reports.

## Docker
```bash
docker compose up --build
```

## AWS
`deploy/aws.md` describes deployment with ECR, Secrets Manager and ECS Fargate behind an ALB.
It is a guide, not a live deployment (AWS is not free beyond trial credits).

## Tests
```bash
pytest -q
```

## Limitations
- No conversation memory yet: each question is answered independently.
- Answers are only as good as the indexed papers; scanned PDFs without text are skipped.
- Free-tier rate limits apply to Groq and Pinecone.