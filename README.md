# RAG from Scratch — Nutrition Textbook

A local Retrieval-Augmented Generation pipeline over a 1200-page human
nutrition textbook. Built without LangChain, LlamaIndex, or a vector
database — just sentence-transformers, PyMuPDF, spaCy, and a local LLM.

Comes with a small FastAPI + plain-HTML UI for interactive search and
question answering.

## What's inside

| Stage | Tool | What happens |
|---|---|---|
| Ingest | `PyMuPDF (fitz)` | Download the textbook PDF, extract text per page |
| Sentencize | `spaCy` | Split each page into sentences |
| Chunk | custom | Group every 10 sentences into a chunk; drop chunks < 30 tokens |
| Embed | `sentence-transformers` (`all-mpnet-base-v2`) | 768-dim embeddings, batched on GPU/MPS/CPU |
| Index | flat tensor on disk | CSV with chunks + embedding vectors (~22 MB) |
| Retrieve | `torch.topk` over `util.dot_score` | Top-k chunks per query, milliseconds |
| Generate | `transformers` (`unsloth/Phi-3-mini-4k-instruct`) | System + user chat template, sampled decoding |

No external services. Everything runs on your machine.

## Project layout

```
.
├── app.py                  FastAPI server + /api/{status,search,ask}
├── static/index.html       Single-page UI (no framework)
├── src/rag/
│   ├── config.py           Paths, model names, device autodetect
│   ├── ingest.py           PDF download + text extraction
│   ├── chunking.py         spaCy sentencizer + chunking + filtering
│   ├── embeddings.py       Encode / save / load embeddings
│   ├── retrieval.py        Dot-product top-k similarity search
│   ├── prompts.py          System + user message templates, sample queries
│   └── llm.py              Phi-3 loader, chat-template prompt, ask()
├── scripts/
│   ├── build_index.py      One-shot: PDF → embeddings.csv
│   ├── search.py           Retrieval-only CLI
│   └── ask.py              Full RAG QA CLI
├── pyproject.toml
└── rag.ipynb               Original walkthrough notebook (preserved)
```

## Quickstart

```bash
git clone https://github.com/raahul4004/rag-from-scratch.git
cd rag-from-scratch

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Recommended: set a HuggingFace token to avoid rate-limited model downloads.
# Get one at https://huggingface.co/settings/tokens (read-only is fine).
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 1. Build the index (downloads PDF + embedding model on first run; ~2 min on Apple Silicon)
python scripts/build_index.py

# 2a. Use the CLI
python scripts/search.py "foods high in fiber"
python scripts/ask.py "What are Vitamins?"

# 2b. Or launch the web UI
python app.py     # http://127.0.0.1:7860
```

> **First Ask call downloads ~7 GB of Phi-3 weights.** This is one-time and
> cached at `~/.cache/huggingface/hub/`. Without `HF_TOKEN` the unauthenticated
> rate limit will throttle/stall this download.

## Web UI

Two tabs:

- **Search** — embed a query, return the top-k most relevant textbook chunks
  with score and page number. Fast, no LLM, runs in milliseconds.
- **Ask** — full RAG: retrieve top-k context, build a chat-template prompt
  (system + user messages), generate an answer with the local LLM.

The UI lazy-loads heavy models, so the server boots instantly. Errors
(missing index, LLM unavailable) surface inline as a red status line.

## Performance

Measured on Apple Silicon (M-series, MPS, fp16):

| Step | Time |
|---|---|
| `/api/search` (cold) | ~3 s for first call (loads embedding model) |
| `/api/search` (warm) | ~30 ms |
| `/api/ask` model load (one-time) | ~7 s |
| `/api/ask` short answer (~50 tokens) | ~5–8 s |
| `/api/ask` typical answer (~150 tokens) | ~15–20 s |
| `/api/ask` full `max_new_tokens=256` | ~25–30 s |
| `/api/ask` `max_new_tokens=512` | ~50–60 s |

LLM throughput is roughly **10 tokens/sec on MPS in fp16**. Drop
`max_new_tokens` in the UI for snappier responses.

## Device support

`src/rag/config.py` auto-detects:

| Device | Embeddings | LLM | Notes |
|---|---|---|---|
| CUDA | fast | fast (4-bit via bitsandbytes if installed) | Best path; install `pip install -e .[cuda]` |
| MPS (Apple Silicon) | fast | usable in fp16 | No `bitsandbytes` / `flash-attn`; uses `sdpa` attention |
| CPU | OK | very slow | Fine for embeddings; LLM gen is minutes per answer |

> **MPS note:** Phi-3 in fp32 will OOM-crash on most Macs (needs ~15 GB working
> memory). `src/rag/llm.py` loads it in fp16 on MPS to stay under the limit.

## Configuration

All knobs live in `src/rag/config.py`:

```python
embedding_model_name = "all-mpnet-base-v2"
llm_model_id         = "unsloth/Phi-3-mini-4k-instruct"
num_sentence_chunk_size = 10     # sentences per chunk
min_token_length        = 30     # filter tiny chunks
n_resources_to_return   = 5      # top-k
```

To use a different textbook, change `pdf_url` and re-run `scripts/build_index.py`.

## API endpoints

The FastAPI server exposes:

```
GET  /                  HTML UI
GET  /api/status        device, model names, index_ready, sample queries
POST /api/search        { query, top_k }                       → ranked chunks
POST /api/ask           { query, top_k, temperature, max_new_tokens } → answer + context
```

OpenAPI docs auto-generated at `/docs`.

## How the pipeline works

```
       ┌──────────────┐
PDF ─► │ open_and_    │ ─► pages_and_texts (1208 pages)
       │   read_pdf   │
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ spaCy        │ ─► page-level sentence lists
       │ sentencizer  │
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ chunk +      │ ─► 1680 chunks (after min-token filter)
       │ filter       │
       └──────────────┘
              │
              ▼
       ┌──────────────┐
       │ all-mpnet-   │ ─► (1680, 768) embedding tensor → CSV
       │ base-v2      │
       └──────────────┘

Query time:
  query → embed (1, 768) → dot-product vs (1680, 768) → topk → context
  context + chat-template prompt → Phi-3 → answer
```

## Things that aren't here (on purpose)

- No vector DB. The corpus is small enough that a single dense tensor + `torch.topk` is faster than spinning up FAISS or Qdrant.
- No LangChain / LlamaIndex wrappers. The pipeline is small enough to read top-to-bottom in `src/rag/`.
- No reranker / query rewriting / hybrid BM25. The point is to see what raw dense retrieval + a small LLM can do.

## Differences from the original notebook

The original `rag.ipynb` is preserved as the walkthrough. The package
fixes a few notebook bugs along the way:

- `page_token)count` typo → `page_token_count`
- Hardcoded `device="cuda"` → autodetect cuda/mps/cpu
- Hardcoded fp32 on non-CUDA → fp16 on MPS (avoids OOM crashes)
- Undefined `use_quantization_config` → gated on CUDA presence
- `prompt_formatter` returned one value; `ask()` unpacked two — now consistent
- Regex `[A-z]` (matches `[\]^_\``) → `[A-Z]`
- Raw-text few-shot prompt with literal `<extract relevant passages>` placeholder
  → proper `tokenizer.apply_chat_template` with system + user messages
- Fragile decoded-string slicing → token-index slicing for clean answer extraction
- Quantization (`bitsandbytes`) and `flash-attn` made optional, not hard deps

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- Textbook: *Human Nutrition* (Open Educational Resources, U. of Hawaiʻi)
- Embeddings: [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- LLM: [`unsloth/Phi-3-mini-4k-instruct`](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct)
- Inspired by Daniel Bourke's "RAG from scratch" walkthrough.
