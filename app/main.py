import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.ingestion import LOADERS, ingest_file

state: dict = {}
CHAT_PAGE = Path(__file__).parent / "static" / "chat.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.graph import build_graph

    state["graph"] = build_graph()
    yield
    state.clear()


app = FastAPI(title="RAG Generative AI Workflow Orchestrator", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str | None
    chunk_index: int | None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[Source]
    trace: list[str]


@app.get("/", include_in_schema=False)
def chat_page():
    return FileResponse(CHAT_PAGE, media_type="text/html; charset=utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in LOADERS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {sorted(LOADERS)}")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / Path(file.filename).name   # keep original name as 'source'
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        n = ingest_file(dest)
    return {"file": file.filename, "chunks_indexed": n}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question must not be empty")
    result = state["graph"].invoke({"question": req.question, "trace": []})
    return QueryResponse(
        answer=result["generation"],
        grounded=result.get("grounded", False),
        sources=[
            Source(source=d.metadata.get("source"), chunk_index=d.metadata.get("chunk_index"),
                   snippet=d.page_content[:200])
            for d in result.get("documents", [])
        ],
        trace=result.get("trace", []),
    )
