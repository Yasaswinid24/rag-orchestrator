# AI Study Assistant — RAG-Based Generative AI Workflow Orchestrator

A personal chatbot that answers questions about AI/ML research, grounded in a library of landmark papers
(Transformer, BERT, RAG, Self-RAG, LoRA, GPT-3, Llama 2, ReAct, Whisper, YOLO and more).
Every answer comes with its sources, and the system checks whether the answer is actually supported by them.

**Stack:** Python · LangGraph · Pinecone · FastAPI · Docker · AWS (deployment guide)
**Cost:** $0. Groq free tier (LLM), Pinecone Starter (vector DB), local embeddings (fastembed).

## How it works

The system has two parts: an **ingestion pipeline** that prepares the documents once, and a **query workflow** that runs for every question.

### 1. Ingestion: from PDF to searchable vectors
```
PDF / DOCX / MD / TXT  ->  load  ->  chunk  ->  embed  ->  upsert into Pinecone
```

| Step | What happens | Why |
|---|---|---|
| **Load** | PyPDF, Docx2txt or a text loader reads each file. The file name becomes the `source`. | Every answer can name the paper it came from. |
| **Chunk** | A recursive splitter cuts the text into pieces of 800 characters with 120 overlap, preferring paragraph, then line, then sentence boundaries. | Small pieces give precise matches. Overlap stops an idea from being cut in half. |
| **Embed** | A local model (`BAAI/bge-small-en-v1.5`, via fastembed) turns each chunk into a 384-number vector. | Similar meanings get similar vectors, so search works on meaning, not keywords. Runs on the CPU, free, no API key. |
| **Store** | Vectors and metadata (source, chunk index, text) go into a Pinecone serverless index with cosine similarity. | Fast nearest-neighbour search. |

Engineering details that matter in practice:
- **Deterministic IDs:** each chunk ID is a SHA-1 hash of source, position and text. Re-ingesting a file overwrites it instead of creating duplicates.
- **Batching and resume:** chunks are embedded and uploaded in batches of 64 with 2 CPU threads, and a manifest file records which files are done. A crash, or a laptop that runs out of memory, loses at most one file, and re-running skips finished files.
- **Failure handling:** a bad or scanned PDF is reported and skipped, and the rest of the batch continues.

### 2. Query workflow: what happens to every question
The workflow is a LangGraph state graph. Each box is a step, and the arrows are decisions.

```
question + chat history
        |
  [contextualize]  rewrite follow-ups into a standalone question
        |
     [route] ----- greeting / small talk -----> [direct_answer] ---> reply (no documents used)
        |
   needs documents
        |
   [retrieve]  top 5 chunks from Pinecone
        |
 [grade_documents]  keep only chunks that help answer the question
        |
        |-- none relevant, retries left --> [rewrite_query] --> back to [retrieve] (max 2 times)
        |
   [generate]  answer using ONLY the numbered chunks, cite [1], [2]
        |
 [check_grounding]  is every claim supported by those chunks?
        |
 answer + sources + grounded flag + trace of the steps taken
```

| Step | What it does | Why it exists |
|---|---|---|
| **contextualize** | With chat history, an LLM rewrites "What about its limits?" into "What are the limitations of LoRA?". Without history it does nothing. | Vector search cannot resolve "it", so follow-ups would retrieve the wrong text. This is the conversation memory. |
| **route** | An LLM decides: greeting or question about the assistant (answer directly) or anything else (retrieve). | Small talk does not need a database search. General-knowledge questions still go to retrieval, so answers come from the documents. |
| **retrieve** | Embeds the question and fetches the 5 most similar chunks. | Finds the evidence. |
| **grade_documents** | One LLM call returns the numbers of the chunks that are really relevant. | Vector search always returns something, even for an unrelated question. Grading removes the noise before the answer is written. |
| **rewrite_query** | If no chunk was relevant, the question is reformulated and retrieval is tried again, up to 2 times. | Recovers from poorly worded questions. |
| **generate** | The LLM answers from the numbered chunks only and cites them. If there is not enough information it says so. | Reduces hallucination and makes answers checkable. |
| **check_grounding** | A second LLM call checks whether every claim in the answer is supported by the chunks. The chat page shows "grounded" or "not verified". | Flags answers that go beyond the evidence. |

**Reliability:** structured LLM outputs (route, grade, grounding) use tool calling with a JSON-schema fallback and a safe default, so one malformed model reply does not crash a request.

**Transparency:** every response includes a `trace` of the steps taken, and the chat page shows it under "workflow steps". Example for the follow-up "What problem does it solve?":
```
contextualize -> 'What problem does LoRA solve?'
route=retrieve
retrieve(k=5, q='What problem does LoRA solve?')
grade_documents(kept=3)
generate
check_grounding=True
```

### 3. Design decisions
| Decision | Reason |
|---|---|
| Local embeddings instead of an embedding API | Free, no rate limits, no data leaves the machine. |
| Pinecone serverless | Managed vector database on a free tier. |
| LangGraph instead of a single prompt | Explicit steps and loops (grade, rewrite, retry) that can be tested and traced. |
| Grade before generating | Fewer irrelevant chunks means better answers and fewer tokens. |
| Stateless server with history sent by the client | Simple to deploy and scale. Long-term per-user memory would need a database (see Limitations). |
| Evaluation on retrieval and answers separately | Shows whether a problem comes from search or from generation. |

### 4. Where the code lives
| Part | File |
|---|---|
| Config | `app/config.py` |
| LLM + robust structured output | `app/llm.py` |
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

Follow-up questions: pass earlier turns in the optional `history` field and the assistant resolves references like "it":
```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d '{
  "question": "What problem does it solve?",
  "history": [
    {"role": "user", "content": "What is LoRA?"},
    {"role": "assistant", "content": "LoRA freezes the pretrained weights and trains small low-rank matrices."}
  ]}'
```
The `trace` then starts with `contextualize -> 'What problem does LoRA solve?'`. The built-in chat page does this automatically.

## Evaluation
`eval/dataset.json` holds 20 questions with reference answers and expected source papers.
`python eval/evaluate.py` writes `eval/report.json` with:

- **Hit@k / MRR:** did retrieval surface the right paper, and how high
- **Answer accuracy:** LLM-as-judge vs the reference answer
- **Groundedness:** share of answers supported by the retrieved context

| Metric | Result |
|---|---|
| Hit@5 | 1.00 |
| MRR | 0.95 |
| Answer accuracy | 0.95 |
| Groundedness | 0.85 |

Measured on 20 questions over a knowledge base of 15 AI papers plus 4 unrelated policy documents (19 PDFs, 3,686 chunks). Accuracy is judged by the same LLM family that generates the answers, and the set is small, so treat the numbers as indicative. Tune `CHUNK_SIZE`, `CHUNK_OVERLAP` and `TOP_K` and compare reports.

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
- Memory is short-term only (kept in the browser, lost on refresh). Long-term per-user memory would need a database such as Postgres with pgvector.
- Answers are only as good as the indexed papers; scanned PDFs without text are skipped.
- Free-tier rate limits apply to Groq and Pinecone.