@echo off

REM ===== Core paths =====
set "RAG_ROOT=C:\Users\kadek\Documents\GitHub\KinaxisCapstone\_rag_testGen"
set "DOMAIN_DIR=C:\Users\kadek\Documents\GitHub\KinaxisCapstone\example1"

REM ===== Run behavior =====
set "N_ITEMS=5"
set "FORCE_INGEST="

REM ===== Postgres (Docker pgvector) =====
set "DB_DSN=postgresql://postgres:postgres@localhost:5435/kinaxis_ragtestdb"
set "DOCKER_CONTAINER=pgvector17"

REM ===== LM Studio =====
set "LM_URL=http://localhost:1234"

set "EMBED_MODEL=text-embedding-nomic-embed-text-v1.5@q8_0"
set "SME_MODEL=Qwen2.5-7B-Instruct-Q5_K_M"
set "REVIEW_MODEL=Qwen2.5-7B-Instruct-Q5_K_M"

REM LM Studio server log
set "LMSTUDIO_LOG_PATH=C:\Users\kadek\AppData\Roaming\LM Studio\logs\main.log"
