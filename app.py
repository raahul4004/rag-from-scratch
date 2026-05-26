"""Compact FastAPI UI for the nutrition RAG pipeline.

Run:  python app.py        (uvicorn on http://127.0.0.1:7860)
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from rag.config import Config
from rag.prompts import SAMPLE_QUERIES

CFG = Config()

_state: dict = {
    "embeddings": None,
    "pages_and_chunks": None,
    "embedding_model": None,
    "tokenizer": None,
    "llm_model": None,
}

app = FastAPI(title="Nutrition RAG")


def _ensure_index() -> None:
    if _state["embeddings"] is not None:
        return
    if not CFG.embeddings_csv.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Index missing at {CFG.embeddings_csv}. "
                "Run `python scripts/build_index.py` first."
            ),
        )
    try:
        from rag.embeddings import load_chunks_with_embeddings, load_embedding_model
    except ImportError as e:
        raise HTTPException(503, f"ML deps missing: {e}. Run `pip install -e .`") from e

    pages_and_chunks, embeddings = load_chunks_with_embeddings(
        CFG.embeddings_csv, device=CFG.device
    )
    _state["pages_and_chunks"] = pages_and_chunks
    _state["embeddings"] = embeddings
    _state["embedding_model"] = load_embedding_model(
        CFG.embedding_model_name, device=CFG.device
    )


def _ensure_llm() -> None:
    if _state["llm_model"] is not None:
        return
    from rag.llm import load_llm

    tokenizer, llm_model = load_llm(
        CFG.llm_model_id, device=CFG.device, use_quantization=CFG.use_quantization
    )
    _state["tokenizer"] = tokenizer
    _state["llm_model"] = llm_model


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    temperature: float = 0.7
    max_new_tokens: int = 256


@app.get("/api/status")
def status() -> dict:
    return {
        "device": CFG.device,
        "embedding_model": CFG.embedding_model_name,
        "llm_model": CFG.llm_model_id,
        "index_ready": CFG.embeddings_csv.exists(),
        "index_path": str(CFG.embeddings_csv),
        "samples": SAMPLE_QUERIES,
    }


@app.post("/api/search")
def api_search(req: SearchRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    _ensure_index()
    from rag.retrieval import retrieve_relevant_resources

    scores, indices = retrieve_relevant_resources(
        query=req.query,
        embeddings=_state["embeddings"],
        model=_state["embedding_model"],
        n_resources_to_return=req.top_k,
        print_time=False,
    )
    results = []
    for score, idx in zip(scores, indices):
        chunk = _state["pages_and_chunks"][int(idx)]
        results.append(
            {
                "score": float(score),
                "page": int(chunk["page_number"]),
                "text": chunk["sentence_chunk"],
            }
        )
    return {"query": req.query, "results": results}


@app.post("/api/ask")
def api_ask(req: AskRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    _ensure_index()
    try:
        _ensure_llm()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"LLM unavailable: {e}") from e

    from rag.llm import ask as ask_fn

    answer, context_items = ask_fn(
        query=req.query,
        embeddings=_state["embeddings"],
        pages_and_chunks=_state["pages_and_chunks"],
        embedding_model=_state["embedding_model"],
        tokenizer=_state["tokenizer"],
        llm_model=_state["llm_model"],
        n_resources_to_return=req.top_k,
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
        return_context=True,
    )
    context = [
        {
            "score": float(item.get("score", 0)),
            "page": int(item["page_number"]),
            "text": item["sentence_chunk"],
        }
        for item in context_items
    ]
    return {"query": req.query, "answer": answer, "context": context}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")


if __name__ == "__main__":
    main()
