Capstone Agentic RAG Prototype (_rag_testGen)

Overview
This folder is a local-first prototype for a staged RAG pipeline:
1) Domain ingestion: documents loaded, knowledge extracted by a context model, embedded into
   PostgreSQL + pgvector.
2) Retrieval + human-in-the-loop quality checkpoints (chunks, items, reviewer flags).
3) Test generation, agentic review, and statistical analysis across conditions.

The BAT and SH launcher files are intentionally minimal — they only call runner.py.
All real behavior lives in Python modules.

--------------------------------------------------
Folder layout
--------------------------------------------------
_rag_testGen\
  _prompts\
    context_system.txt
    context_user.txt
    generator_system.txt
    generator_user.txt
    reviewer_system.txt
    reviewer_user.txt
  runs\
    logs_RUNID\
      console_pipeline.txt
      run_info.txt
      run_RUNID.xlsx
      run_manifest_RUNID.json
      llm_http.jsonl
      lmstudio.log
      docker_CONTAINERNAME.log
  cli.py
  config.py
  config.env              (auto-created on first run)
  db_pgvector.py
  embed_lmstudio.py
  ingest.py
  llm_client.py
  loaders.py
  pipeline.py
  runner.py
  text_utils.py
  __init__.py
  __main__.py

_run_testGen.bat          (Windows launcher — sits beside _rag_testGen\)
_run_testGen.sh           (Mac/Linux launcher — chmod +x required)

--------------------------------------------------
Prerequisites
--------------------------------------------------

1) Python 3.10+

2) LM Studio
   - Running locally with OpenAI-compatible server enabled
   - Required endpoints: /v1/embeddings, /v1/chat/completions
   - For vision (PDF ingest): context window must be set to 32768+ for vision model
   - Models needed:
     - Embedding model (e.g. nomic-embed-text)
     - Context model (e.g. qwen2.5-vl-7b-instruct for vision, or any instruct model for text)
     - Generator model
     - Reviewer model (can be same as generator)

3) Docker Desktop (Linux containers mode)

4) API key (recommended for PDF ingest on low-VRAM hardware)
   - Set via runtime prompt in runner.py — never saved to config.env
   - Required if LLM_PROVIDER=anthropic or PDF_PROVIDER=anthropic

--------------------------------------------------
Hardware note
--------------------------------------------------

PDF vision ingest requires a vision-capable LLM. On hardware with less than
8GB VRAM, local vision models run on CPU and are extremely slow (~2 min per
4-page batch). For PDF-heavy corpora, routing PDF ingest to the API is
strongly recommended. PPTX/DOCX/TXT ingestion uses text extraction and runs fast
on any hardware via local LM Studio.

--------------------------------------------------
Docker-based PostgreSQL + pgvector
--------------------------------------------------

Run PostgreSQL 17 with pgvector preinstalled:

  docker run --name pgvector17 ^
    --restart unless-stopped ^
    -e POSTGRES_PASSWORD=postgres ^
    -p 5435:5432 ^
    -d pgvector/pgvector:pg17

Verify:
  docker ps --filter "name=pgvector17"

Initialize database:
  docker exec -it pgvector17 psql -U postgres -c "CREATE DATABASE kinaxis_ragtestdb;"

Enable pgvector extension:
  psql "host=localhost port=5435 dbname=kinaxis_ragtestdb user=postgres password=postgres" ^
       -c "CREATE EXTENSION IF NOT EXISTS vector;"

--------------------------------------------------
Python dependencies
--------------------------------------------------

  pip install requests psycopg pgvector python-docx pypdf python-pptx pymupdf openpyxl

--------------------------------------------------
Configuration (config.env)
--------------------------------------------------

Auto-created on first run. Key settings:

  RAG_ROOT          Path to _rag_testGen\ folder
  DOMAIN_DIR        Path to folder containing source documents
  N_ITEMS           Number of MCQ items to generate per run
  DB_DSN            PostgreSQL connection string
  DOCKER_CONTAINER  Docker container name for log capture
  LM_URL            LM Studio server URL (default http://localhost:1234)
  LLM_PROVIDER      lmstudio / openai / anthropic / gemini (default lmstudio)
  EMBED_MODEL       Embedding model name (always local via LM Studio)
  CONTEXT_MODEL     Knowledge extraction model (vision-capable for PDFs)
  GENERATOR_MODEL   MCQ generation model
  REVIEW_MODEL      MCQ review model (can match GENERATOR_MODEL)
  LMSTUDIO_LOGPATH  Path to LM Studio main.log for log capture
  CHECKPOINT_CHUNKS Pause after ingest to review knowledge chunks (true/false)
  CHECKPOINT_ITEMS  Pause after generation to review items (true/false)
  CHECKPOINT_REVIEW Pause after review to correct flagged items (true/false)
  VISION_TIMEOUT_SECONDS  Request timeout for vision calls (default 600)
  RENDER_DPI        DPI for PDF page rendering (default 96)

LLM_API_KEY is never written to config.env. It is entered via masked prompt
at runtime and stored only in the process environment for that session.

--------------------------------------------------
Running the project
--------------------------------------------------

Windows:
  Double-click _run_testGen.bat
  OR: python _rag_testGen\runner.py

Mac/Linux:
  ./_run_testGen.sh
  OR: python3 _rag_testGen/runner.py

First run:
  When prompted "Update settings?" enter Y and set all paths and model names.

Run modes:
  F = Full pipeline  (ingest domain folder + generate items, RAG mode)
  I = Ingest only    (extract knowledge chunks, embed, write XLSX — no generation)
  G = Generate only  (use existing DB chunks, generate + review items)
  B = Baseline       (no-RAG: load docs directly, generate without pgvector)

Human-in-the-loop checkpoints:
  After ingest:      review extracted knowledge chunks, edit or skip before embedding
  After generation:  review generated items, edit or skip before reviewer
  After review:      see flagged items, override or correct before final output

Checkpoints can be disabled per-stage in config.env (set to false).
When checkpoints are enabled, the terminal must remain interactive (do not pipe to file).

--------------------------------------------------
Output (runs\logs_RUNID\)
--------------------------------------------------

  run_RUNID.xlsx          Multi-sheet workbook:
                            Run Metadata, DB Snapshot, Chunk Preview,
                            Items, Reviewer Decisions, Traceability, Quality Metrics
  run_manifest_RUNID.json Run config snapshot
  console_pipeline.txt    Captured stdout (when checkpoints disabled)
  run_info.txt            Config + timing
  llm_http.jsonl          Per-request LLM log (provider, model, elapsed_ms, token counts)
  lmstudio.log            LM Studio log tail
  docker_CONTAINERNAME.log Docker container log tail

--------------------------------------------------
Corpus Material Source Citations
--------------------------------------------------

Massachusetts Institute of Technology. (2009). Logistics and supply chain management
(ESD.273J) [Lecture notes]. MIT OpenCourseWare.
https://ocw.mit.edu/courses/esd-273j-logistics-and-supply-chain-management-fall-2009/

Adjemian, M. K., Wilson, W. W., Bullock, D. W., & Lakkakula, P. (2021).
Recent surges in ocean transportation rates and their effects on selected agricultural
markets [Presentation slides]. University of Georgia; North Dakota State University.

Beer Game Supply Chain Simulation. (n.d.).
The beer game: Supply chain dynamics and the bullwhip effect [PowerPoint slides].

Overcoming the barriers to supply chain integration. (n.d.).
Chapter 16: Building integrated supply chains [PowerPoint slides].

Supply chain management. (n.d.).
Introduction to supply chain management and the bullwhip effect [PowerPoint slides].

Simchi-Levi, D. (2021). Supply chain modernization and digitization [Video interview].
Algo Podcast. https://www.youtube.com/watch?v=Bo6J4gkoBXM

Supply Chain Management and the Bullwhip Effect. (n.d.).
Operations management overview [Video transcript].
https://www.youtube.com/watch?v=jM0k3em1G_A
