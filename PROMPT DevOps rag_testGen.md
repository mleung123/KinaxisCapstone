# RAG TestGen — System Context Prompt for LLM Collaborators

## What This Program Does

`_rag_testGen` is a Python pipeline that generates multiple-choice exam items grounded in domain
source material using a retrieval-augmented generation (RAG) architecture with a separate
critic/reviewer model. It ingests domain documents (PDF, PPTX, DOCX, TXT), extracts structured
knowledge chunks via a context model, embeds them into Postgres+pgvector, then uses a generator
LLM + reviewer LLM to produce structured MCQ items with quality scores. The system is designed
to answer two research questions: does a dual-model agentic RAG architecture produce measurably
higher-quality items than a single-model no-RAG baseline, and does agentic critique reduce human
review burden?

---

## Project Structure

```
_rag_testGen/
├── cli.py                  # Entrypoint; dispatches ingest / generate / baseline / pipeline subcommands
├── config.py               # Loads env vars into ResolvedConfig dataclass
├── pipeline.py             # Orchestrates ingest + generate phases; human-in-the-loop checkpoints; writes XLSX
├── ingest.py               # Loads docs, context model extracts knowledge chunks, embeds, upserts to pgvector
├── loaders.py              # Document loaders (PDF, PPTX, DOCX, TXT) + preprocess_text(); text extraction only
├── llm_client.py           # Unified LLM provider abstraction (lmstudio/openai/anthropic/gemini); text + vision
├── embed_lmstudio.py       # LM Studio /v1/embeddings client (embeddings remain local)
├── db_pgvector.py          # All Postgres/pgvector DB access (schema, upsert, search, snapshots)
├── text_utils.py           # Generator output cleaning, schema validation, reviewer hygiene
├── runner.py               # Cross-platform launcher: config persistence, run mode, log capture, run-again loop
├── __init__.py             # Package marker
├── __main__.py             # Allows `python -m _rag_testGen`
├── _prompts/
│   ├── generator_system.txt
│   ├── generator_user.txt      # Contains {{CONTEXT}} placeholder
│   ├── reviewer_system.txt
│   ├── reviewer_user.txt       # Contains {{GEN_ITEM}} and {{CONTEXT}} placeholders
│   ├── context_system.txt      # Knowledge extraction instructions for context model
│   └── context_user.txt        # Contains {{DOCUMENT}} placeholder (text mode only)
├── runs/
│   └── logs_RUNID/
│       ├── console_pipeline.txt / console_generate.txt / console_baseline.txt
│       ├── run_info.txt
│       ├── run_RUNID.xlsx
│       ├── run_manifest_RUNID.json
│       ├── lmstudio.log
│       ├── llm_http.jsonl
│       └── docker_CONTAINERNAME.log
└── config.env              # Auto-written plain KEY=VALUE (no shell syntax)

_run_testGen.bat            # Windows thin shim — calls runner.py
_run_testGen.sh             # Mac/Linux thin shim — calls runner.py (chmod +x required)
```

---

## Launcher

Both `_run_testGen.bat` (Windows) and `_run_testGen.sh` (Mac/Linux) are thin shims that
simply call `python runner.py` or `python3 runner.py`. All orchestration logic lives in
`runner.py`.

### `runner.py`

- Loads and saves `config.env`
- Prompts user: update settings? (Y/N), shows current values
- Prompts user: run mode F / I / G / B
  - F = Full pipeline (ingest + generate, RAG mode)
  - I = Ingest only (extract knowledge chunks, write XLSX)
  - G = Generate only (use existing DB chunks, RAG mode)
  - B = Baseline (no-RAG, load docs directly)
- Generates RUN_ID (UTC timestamp)
- Creates `runs/logs_RUNID/` directory
- Sets environment variables for subprocess
- Calls `cli.py` via subprocess
- When checkpoints are disabled: tees stdout to console + log file
- When checkpoints are enabled: direct subprocess (stdin passes through for interactive prompts)
- Captures docker logs and LM Studio logs into run folder after completion
- Writes `run_info.txt` with config + timing
- Asks "Run again?" loop

---

## Pipeline Flow

```
_run_testGen.bat / _run_testGen.sh
  └── runner.py
        └── cli.py  (pipeline / generate / ingest / baseline subcommand)
              ├── ingest phase:
              │     loaders.py → load_document()         [text extraction, all types]
              │     ingest.py → extract_knowledge_chunks()
              │       ├── PDFs: llm_client.py → call_llm_vision()
              │       │     lmstudio/openai/gemini: render pages → image_url blocks (batched)
              │       │     anthropic: raw PDF → document block (one call, native)
              │       └── PPTX/DOCX/TXT: llm_client.py → call_llm() [text mode]
              │     embed_lmstudio.py → embed_texts()
              │     db_pgvector.py → upsert_chunks()
              │     [CHECKPOINT 1] human reviews knowledge chunks (optional)
              │
              └── generate phase (per item):
                    db_pgvector.py → get_random_chunks() [seed]
                    embed_lmstudio.py → embed_texts() [seed embedding]
                    db_pgvector.py → similarity_search() [top-k retrieval]
                    llm_client.py → call_llm() [generator]
                    [CHECKPOINT 2] human reviews generated items (optional)
                    llm_client.py → call_llm() [reviewer]
                    [CHECKPOINT 3] human reviews flagged items (optional)
                    pipeline.py → write_run_xlsx()
```

---

## Key Modules

### `loaders.py`
- `load_document(path)` — dispatches by extension: `.pdf` → PyMuPDF + pypdf fallback,
  `.pptx` → python-pptx (text + speaker notes), `.docx` → python-docx, `.txt/.md` → plain read
- Returns `LoadedDoc(path, sha256, text, page_count)` — text extraction only, no chunking
- `preprocess_text(text, source_ext)` — normalizes whitespace, removes page number artifacts,
  de-hyphenates PDF line wraps, collapses repeated header/footer lines (freq ≥ 4),
  drops standalone integers (slide numbers), drops Wingdings 'z' artifacts
- `_is_spoken_math_transcript()` — detects and rejects audio transcript files with spoken math notation
- Note: PDF text extraction is used for identity/metadata only; knowledge extraction uses vision

### `ingest.py`
- `extract_knowledge_chunks(doc, cfg, ...)` — routes to vision or text based on file extension
  and LLM_PROVIDER; returns list of knowledge chunk strings
- `_extract_vision_lmstudio()` — renders PDF pages to base64 PNG, sends in batches of
  `vision_pages_per_batch` (default 4) to LM Studio vision endpoint
- `_extract_vision_anthropic()` — sends raw PDF bytes as native document block (one API call)
- `_split_knowledge_output()` — splits context model output (blank-line separated paragraphs)
  into individual chunks, merges short fragments, hard-splits overlong chunks
- `ingest_domain(cfg, prompts_dir)` — full ingest loop with batch embedding and upsert

### `llm_client.py`
- `call_llm(lm_url, model, system_prompt, user_prompt, ...)` — text-only call, all providers
- `call_llm_vision(lm_url, model, system_prompt, user_prompt, image_b64_list, pdf_path, ...)` —
  vision call; provider determines format:
  - lmstudio/openai/gemini: `image_url` blocks with base64 PNG data
  - anthropic: `document` block with raw PDF bytes (native PDF support, no rendering)
- `render_pdf_pages_b64(pdf_path, dpi)` — renders PDF pages to base64 PNG list (MuPDF warnings suppressed)
- `pdf_to_b64(pdf_path)` — raw PDF bytes as base64 (for Anthropic native PDF)
- All calls log to `runs/logs_RUNID/llm_http.jsonl` (provider, model, mode, elapsed_ms, token counts)
- `LLM_API_KEY` read from env only, never logged or written to config

### `embed_lmstudio.py`
- `embed_texts(cfg, texts)` — calls LM Studio `/v1/embeddings`
- Embeddings remain local regardless of LLM_PROVIDER setting

### `pipeline.py`
- Three human-in-the-loop checkpoints (all optional, controlled by config flags):
  - `_checkpoint_chunks()` — after ingest, review/edit/skip knowledge chunks before embedding
  - `_checkpoint_items()` — after generation, review/edit/skip items before reviewer
  - `_checkpoint_review()` — after review, override/correct flagged items
- `generate_from_db(cfg)` — RAG mode: retrieves top-k chunks, generates + reviews MCQ items
- `generate_baseline(cfg)` — baseline mode: loads docs directly, generates without pgvector
- `run_pipeline(cfg)` — orchestrates ingest + generate
- `write_run_xlsx()` — writes multi-sheet XLSX:
  Run Metadata, DB Snapshot, Chunk Preview, Items, Reviewer Decisions, Traceability, Quality Metrics
- Note: old context-rewrite-at-query-time agent removed; context model now runs at ingest time

### `config.py`
- `load_config_from_env()` — reads all settings from environment variables
- `ResolvedConfig` — frozen dataclass; `.with_overrides()` for CLI arg application
- Supports backward-compatible alias `SME_MODEL` → `GENERATOR_MODEL`
- New fields: `llm_provider`, `context_model`, `baseline_mode`, `checkpoint_chunks`,
  `checkpoint_items`, `checkpoint_review`

### `db_pgvector.py`
- Schema: `rag_chunks` (doc_path, doc_sha256, chunk_index, chunk_text, embedding vector(N), meta jsonb)
- Schema: `rag_meta` (k, v — stores embed_model, embedding_dim, source_root, context_model)
- `similarity_search()` — cosine distance via pgvector `<->` operator
- `get_db_snapshot_summary()` / `get_db_snapshot_per_doc()` — for XLSX DB Snapshot sheet

---

## Configuration (`config.env`)

Plain `KEY=VALUE` file, no shell syntax. Written by `runner.py`. `LLM_API_KEY` is never
written here — entered at runtime via masked prompt, stored only in process environment.

```
RAG_ROOT=C:\Users\kadek\Documents\GitHub\KinaxisCapstone\_rag_testGen
DOMAIN_DIR=C:\Users\kadek\Documents\GitHub\KinaxisCapstone\example1
N_ITEMS=1
DB_DSN=postgresql://postgres:postgres@localhost:5435/kinaxis_ragtestdb
DOCKER_CONTAINER=pgvector17
LM_URL=http://localhost:1234
LLM_PROVIDER=lmstudio
EMBED_MODEL=text-embedding-nomic-embed-text-v1.5@q8_0
CONTEXT_MODEL=qwen/qwen2.5-vl-7b-instruct
GENERATOR_MODEL=qwen2.5-7b-instruct-uncensored
REVIEW_MODEL=qwen2.5-7b-instruct-uncensored
LMSTUDIO_LOGPATH=C:\Users\kadek\AppData\Roaming\LM Studio\logs\main.log
CHECKPOINT_CHUNKS=true
CHECKPOINT_ITEMS=true
CHECKPOINT_REVIEW=true
VISION_TIMEOUT_SECONDS=600
RENDER_DPI=96
```

Mac equivalent paths use `/Users/username/...` and `~/Library/Logs/LM Studio/main.log`.
The DB requires manual creation:
`docker exec -it pgvector17 psql -U postgres -c "CREATE DATABASE kinaxis_ragtestdb;"`

---

## Hardware Constraint — PDF Vision Ingest

The operator machine has a GTX 1650 (4GB VRAM). The `qwen2.5-vl-7b` vision model requires
~8GB VRAM and runs on CPU on this hardware. PDF page image encoding takes ~2 minutes per
4-page batch on CPU, making local vision ingest of large PDF corpora impractical (~12 hours
for 230 pages).

**Current recommended approach:**
- PDFs: route to Anthropic API (`LLM_PROVIDER=anthropic`) — native PDF support, no rendering,
  fast and accurate math extraction, pennies per document
- PPTX/DOCX/TXT: run locally via LM Studio text model (fast, no vision needed)
- A `PDF_PROVIDER` override config key to split providers by file type is a planned improvement

---

## Current Known State

- `runner.py` implemented and working; BAT/SH are thin shims
- `llm_client.py` replaces `lmstudio_client.py` — unified provider abstraction with vision support
- Context model runs at ingest time (not query time); old context-rewrite-at-query-time agent removed
- PDF ingest uses vision (rendered pages for lmstudio, native PDF for anthropic)
- PPTX/DOCX/TXT ingest uses text extraction via context model
- Three human-in-the-loop checkpoints implemented (chunks, items, review)
- Baseline (no-RAG) mode implemented as `generate_baseline()` and `baseline` CLI subcommand
- `chunking.py` is vestigial — character-based chunking replaced by context model output splitting
- Vision page batching implemented (`vision_pages_per_batch=4`) to respect LM Studio context limits
- LM Studio context window must be set to 32768+ for vision model to handle 4-page batches
- MuPDF structural warnings suppressed via `fitz.TOOLS.mupdf_display_errors(False)`
- XLSX output unchanged: Run Metadata, DB Snapshot, Chunk Preview, Items, Reviewer Decisions,
  Traceability, Quality Metrics sheets
- No full successful pipeline run completed yet (all attempts hit vision performance wall)
- Generator/reviewer output quality unvalidated end-to-end

---

## Still To Do — Priority Order

### HIGH PRIORITY: End-to-End Validation

Complete a full pipeline run with:
- PDFs routed to Anthropic API
- PPTX via local text model
- N_ITEMS=5, checkpoint all three stages
- Review XLSX output for chunk quality, item quality, reviewer scores

### HIGH PRIORITY: Competency Tagging

Add `competency` / `learning_objective` field to generator output schema and XLSX Items sheet.
Currently not in generator prompt or output.

### MEDIUM PRIORITY: Research Condition Comparison

Implement multiple condition runs:
- RAG + critic (current)
- RAG + no critic
- No-RAG + critic (baseline mode exists)
- No-RAG + no critic
Store condition label in XLSX metadata for side-by-side analysis.

---

## Research Design Context

**RQ1:** Does a dual-model agentic RAG architecture produce higher-quality MCQ items than
single-model generation?
- IV: generation approach (RAG+critic vs no-RAG, RAG+no-critic, no-RAG+no-critic)
- DVs: source alignment, distractor quality, stem clarity, difficulty calibration scores

**RQ2:** Can agentic critique meaningfully reduce human review burden?
- Measured by: proportion of items requiring substantive human revision after critic pass

**Quality dimensions:**
- Source alignment (1-5): grounded in retrieved chunks?
- Distractor quality (1-5): plausible but unambiguously wrong?
- Stem clarity (1-5): unambiguous, grammatically correct, no clues?
- Difficulty calibration (bool): stated difficulty matches cognitive demand?

---

## Constraints and Conventions

- Minimal diffs; full-file replacements preferred when changes are extensive
- Do NOT auto-load LM Studio models from Python
- Prompts do NOT control model selection — model is set in config
- All DB access via `psycopg.connect(dsn)`, no ORM
- Python files use dataclasses, psycopg3, openpyxl, requests — no heavy frameworks
- No f-strings or type annotations with `|` syntax — must support Python 3.9 for compatibility
- `LLM_API_KEY` must never appear in logs, XLSX output, or `config.env`
- When advising on code changes, always provide either:
  (a) full function replacement from the `def` line to the final return/end of function, or
  (b) full file replacement if changes span multiple functions or are extensive
  Do NOT provide surgical line-level snippets or partial function bodies
- When advising on prompt changes, provide full replacement text for the `.txt` file

---

## Session Workflow

**WAIT** — do not proceed with analysis until the user confirms all file uploads are complete.
Acknowledge receipt and state you are ready and waiting.

When the user confirms uploads are complete:
1. Read and acknowledge uploaded files
2. Identify any regressions, bugs, or inconsistencies vs known state above
3. Wait for user to state session objective before proposing changes
4. Propose changes incrementally, confirm before implementing
5. After each run, review uploaded logs and XLSX to verify changes had intended effect
