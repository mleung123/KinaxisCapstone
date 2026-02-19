@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ===== Program root =====
set "RAG_ROOT=C:\Users\kadek\Documents\GitHub\KinaxisCapstone\_rag_testGen"
cd /d "%RAG_ROOT%"

REM ===== Persisted config =====
set "CFG_FILE=%RAG_ROOT%\config.bat"
if exist "%CFG_FILE%" (
  call "%CFG_FILE%"
)

REM ===== Defaults if config.bat is missing fields =====
if "%DOMAIN_DIR%"=="" set "DOMAIN_DIR=%RAG_ROOT%"
if "%N_ITEMS%"=="" set "N_ITEMS=5"
if "%DB_DSN%"=="" set "DB_DSN=postgresql://postgres:postgres@localhost:5435/kinaxis_ragtestdb"
if "%LM_URL%"=="" set "LM_URL=http://localhost:1234"

if "%EMBED_MODEL%"=="" set "EMBED_MODEL=text-embedding-nomic-embed-text-v1.5@q8_0"
if "%SME_MODEL%"=="" set "SME_MODEL=Qwen2.5-7B-Instruct-Q5_K_M"
if "%REVIEW_MODEL%"=="" set "REVIEW_MODEL=%SME_MODEL%"

if "%DOCKER_CONTAINER%"=="" set "DOCKER_CONTAINER=pgvector17"
if "%LMSTUDIO_LOG_PATH%"=="" set "LMSTUDIO_LOG_PATH="

REM ===== Ask whether to update settings =====
set "UPDATE_CHOICE="
set /p UPDATE_CHOICE=Update directories/settings? (Y/N): 

if /i "%UPDATE_CHOICE%"=="Y" goto CONFIGURE
if /i "%UPDATE_CHOICE%"=="YES" goto CONFIGURE
goto RUN

:CONFIGURE
echo.
echo Enter new values, or press Enter to keep defaults.
echo.

set /p RAG_ROOT=Program folder (contains py files) (default %RAG_ROOT%):
if "%RAG_ROOT%"=="" set "RAG_ROOT=%cd%"
cd /d "%RAG_ROOT%"

set /p DOMAIN_DIR=Domain folder (subject matter) (default %DOMAIN_DIR%):
if "%DOMAIN_DIR%"=="" set "DOMAIN_DIR=%DOMAIN_DIR%"

set /p N_ITEMS=How many items to generate? (default %N_ITEMS%):
if "%N_ITEMS%"=="" set "N_ITEMS=5"

set /p DB_DSN=Postgres DSN (default %DB_DSN%):
if "%DB_DSN%"=="" set "DB_DSN=%DB_DSN%"

set /p LM_URL=LM Studio base URL (default %LM_URL%):
if "%LM_URL%"=="" set "LM_URL=%LM_URL%"

set /p EMBED_MODEL=Embedding model (default %EMBED_MODEL%):
if "%EMBED_MODEL%"=="" set "EMBED_MODEL=%EMBED_MODEL%"

set /p SME_MODEL=SME chat model (default %SME_MODEL%):
if "%SME_MODEL%"=="" set "SME_MODEL=%SME_MODEL%"

set /p REVIEW_MODEL=Review chat model (default %REVIEW_MODEL%):
if "%REVIEW_MODEL%"=="" set "REVIEW_MODEL=%REVIEW_MODEL%"

set /p DOCKER_CONTAINER=Docker container name for pgvector logs (default %DOCKER_CONTAINER%):
if "%DOCKER_CONTAINER%"=="" set "DOCKER_CONTAINER=%DOCKER_CONTAINER%"

set /p LMSTUDIO_LOG_PATH=LM Studio log file path to copy (optional; blank to skip):
if "%LMSTUDIO_LOG_PATH%"=="" set "LMSTUDIO_LOG_PATH="

REM ===== Write config (overwrite) =====
(
  echo @echo off
  echo REM Auto-generated. Edit if needed.
  echo.
  echo REM ===== Core paths =====
  echo set "RAG_ROOT=%RAG_ROOT%"
  echo set "DOMAIN_DIR=%DOMAIN_DIR%"
  echo.
  echo REM ===== Run behavior =====
  echo set "N_ITEMS=%N_ITEMS%"
  echo set "FORCE_INGEST=%FORCE_INGEST%"
  echo.
  echo REM ===== Postgres (Docker pgvector) =====
  echo set "DB_DSN=%DB_DSN%"
  echo set "DOCKER_CONTAINER=%DOCKER_CONTAINER%"
  echo.
  echo REM ===== LM Studio =====
  echo set "LM_URL=%LM_URL%"
  echo set "EMBED_MODEL=%EMBED_MODEL%"
  echo set "SME_MODEL=%SME_MODEL%"
  echo set "REVIEW_MODEL=%REVIEW_MODEL%"
  echo set "LMSTUDIO_LOG_PATH=%LMSTUDIO_LOG_PATH%"
) > "%CFG_FILE%"

:RUN
REM ===== Basic validation =====
if not exist "%RAG_ROOT%\cli.py" (
  echo Missing Python entrypoint:
  echo %RAG_ROOT%\cli.py
  pause
  exit /b 1
)

if not exist "%DOMAIN_DIR%" (
  echo Domain folder not found:
  echo %DOMAIN_DIR%
  pause
  exit /b 1
)

REM ===== Generate a stable RUN_ID (UTC) =====
set "RUN_ID="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmssZ')"` ) do set "RUN_ID=%%I"

REM ===== Log folder =====
set "LOG_DIR=%RAG_ROOT%\runs\logs_%RUN_ID%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM ===== Export for Python (IMPORTANT) =====
set "RUN_ID=%RUN_ID%"
set "LOG_DIR=%LOG_DIR%"

REM ===== Run info =====
echo RUN_ID=%RUN_ID% > "%LOG_DIR%\run_info.txt"
echo RAG_ROOT=%RAG_ROOT% >> "%LOG_DIR%\run_info.txt"
echo DOMAIN_DIR=%DOMAIN_DIR% >> "%LOG_DIR%\run_info.txt"
echo DB_DSN=%DB_DSN% >> "%LOG_DIR%\run_info.txt"
echo LM_URL=%LM_URL% >> "%LOG_DIR%\run_info.txt"
echo EMBED_MODEL=%EMBED_MODEL% >> "%LOG_DIR%\run_info.txt"
echo SME_MODEL=%SME_MODEL% >> "%LOG_DIR%\run_info.txt"
echo REVIEW_MODEL=%REVIEW_MODEL% >> "%LOG_DIR%\run_info.txt"
echo DOCKER_CONTAINER=%DOCKER_CONTAINER% >> "%LOG_DIR%\run_info.txt"
echo LMSTUDIO_LOG_PATH=%LMSTUDIO_LOG_PATH% >> "%LOG_DIR%\run_info.txt"

set "RUN_PIPE="
set /p RUN_PIPE=Attempt domain parsing and ingestion now (Y/N)? N to skip and generate questions from existing: 

if /i "%RUN_PIPE%"=="YES" set "RUN_PIPE=Y"
if /i "%RUN_PIPE%"=="Y" (
  python "%RAG_ROOT%\cli.py" pipeline 1> "%LOG_DIR%\console_pipeline.txt" 2>&1
) else (
  python "%RAG_ROOT%\cli.py" generate 1> "%LOG_DIR%\console_generate.txt" 2>&1
)

REM ===== Snapshot docker logs =====
docker logs --tail 5000 "%DOCKER_CONTAINER%" 1> "%LOG_DIR%\docker_%DOCKER_CONTAINER%.log" 2>&1

REM ===== Snapshot LM Studio logs (tail) =====
powershell -NoProfile -Command "Get-Content -Path '%LMSTUDIO_LOG_PATH%' -Tail 5000" 1> "%LOG_DIR%\lmstudio.log" 2>&1

echo.
echo Logs written to:
echo %LOG_DIR%
echo.

pause
exit /b 0
