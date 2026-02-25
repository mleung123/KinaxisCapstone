Capstone Agentic RAG prototype (_rag_testGen)

Overview
This folder is a local-first prototype for a staged RAG pipeline:
1) Domain ingestion (document loading, chunking, embeddings) into PostgreSQL + pgvector.
2) Retrieval + human-in-the-loop quality checkpoints.
3) Test generation, agentic review, and non-agentic statistical analysis.

The BAT file is intentionally minimal. It only:
- Loads persisted configuration
- Optionally prompts for updates
- Launches Python

All real behavior lives in Python modules.

--------------------------------------------------
Folder layout (expected)
--------------------------------------------------
_rag_testGen\
  _prompts\
  runs\
  run_capstone.bat
  cli.py
  ingest.py
  loaders.py
  chunking.py
  db_pgvector.py
  embed_lmstudio.py
  config.bat        (auto-created on first run)

--------------------------------------------------
Prerequisites
--------------------------------------------------

1) Python
- Python 3.10+ recommended

2) LM Studio
- LM Studio running locally
- OpenAI-compatible server enabled
Required endpoints:
- /embeddings
- /chat/completions

3) Docker Desktop

4) PostgreSQL 17 recommended


--------------------------------------------------
Docker-based PostgreSQL + pgvector
--------------------------------------------------

Due to ABI and compiler incompatibilities when building PostgreSQL extensions
against native Windows headers, the recommended setup for this project is to
run PostgreSQL + pgvector inside Docker.

This avoids:
- MSVC / UCRT / Datum-size mismatches
- Manual pgvector compilation on Windows
- Header incompatibilities across PostgreSQL builds

--------------------------------------------------
Docker setup
--------------------------------------------------

Prerequisite:
- Docker Desktop (Linux containers mode)

Verify:
  docker info

--------------------------------------------------
Run PostgreSQL 17 with pgvector
--------------------------------------------------

The following command runs PostgreSQL 17 with pgvector preinstalled,
mapped to localhost port 5435 (to avoid conflicts with local clusters):

  docker run --name pgvector17 ^
    --restart unless-stopped ^
    -e POSTGRES_PASSWORD=postgres ^
    -p 5435:5432 ^
    -d pgvector/pgvector:pg17

Verify container:
  docker ps --filter "name=pgvector17"

--------------------------------------------------
Database initialization
--------------------------------------------------

Enable pgvector (idempotent):

  psql "host=localhost port=5435 dbname=postgres user=postgres password=postgres" ^
       -c "CREATE EXTENSION IF NOT EXISTS vector;"

Create project database:

  psql "host=localhost port=5435 dbname=postgres user=postgres password=postgres" ^
       -c "CREATE DATABASE kinaxis_ragTestDB;"
--------------------------------------------------
Python dependencies
--------------------------------------------------

Required:
- requests
- psycopg (recommended) OR psycopg2
- pgvector (Python helper library)
- python-docx
- pypdf

Install:
  pip install requests psycopg pgvector python-docx pypdf python-pptx PyMuPDF

--------------------------------------------------
Running the project
--------------------------------------------------

First run (create config):
- Double-click run_testGen.bat
- When prompted: Update directories/settings? (Y/N): Y

Set:
- DOMAIN_DIR		(e.g. C:\Users\kadek\Documents\GitHub\KinaxisCapstone\example1)
- DB_DSN		(default postgresql://user:pass@host:5435/db)
- MODEL_URL		(default http://localhost:1234)
- EMBED_MODEL		(embedding model name from LM Studio)
- SME_MODEL		(SME model)
- ENEMY_MODEL		(test review)

This writes config.bat for future runs.

Normal run:
- Double-click run_capstone.bat
- Answer: N
- BAT loads config and launches Python

Corpus Material Source Citations

Massachusetts Institute of Technology. (2009). Logistics and supply chain management (ESD.273J) [Lecture notes]. MIT OpenCourseWare.
https://ocw.mit.edu/courses/esd-273j-logistics-and-supply-chain-management-fall-2009/

Adjemian, M. K., Wilson, W. W., Bullock, D. W., & Lakkakula, P. (2021).
Recent surges in ocean transportation rates and their effects on selected agricultural markets
[Presentation slides]. University of Georgia; North Dakota State University.

Beer Game Supply Chain Simulation. (n.d.).
The beer game: Supply chain dynamics and the bullwhip effect [PowerPoint slides].

Overcoming the barriers to supply chain integration. (n.d.).
Chapter 16: Building integrated supply chains [PowerPoint slides].

Supply chain management. (n.d.).
Introduction to supply chain management and the bullwhip effect [PowerPoint slides].

Simchi-Levi, D. (2021).
Supply chain modernization and digitization [Video interview transcript].
Algo Podcast.
https://www.youtube.com/watch?v=Bo6J4gkoBXM

Supply Chain Management and the Bullwhip Effect. (n.d.).
Operations management overview [Video transcript].
https://www.youtube.com/watch?v=jM0k3em1G_A
