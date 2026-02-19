@echo off
setlocal EnableExtensions DisableDelayedExpansion

REM ===== Program root =====
set "RAG_ROOT=C:\Users\kadek\Documents\GitHub\KinaxisCapstone\_rag_testGen"
cd /d "%RAG_ROOT%"

REM ===== Load persisted configuration (if present) =====
set "CFG_FILE=%RAG_ROOT%\config.bat"
if exist "%CFG_FILE%" call "%CFG_FILE%"

REM ===== Defaults (only if still empty) =====
if "%DOMAIN_DIR%"=="" set "DOMAIN_DIR=%RAG_ROOT%"
if "%N_ITEMS%"=="" set "N_ITEMS=5"
if "%DB_DSN%"=="" set "DB_DSN=postgresql://postgres:postgres@localhost:5435/kinaxis_ragtestdb"
if "%LM_URL%"=="" set "LM_URL=http://localhost:1234"
if "%EMBED_MODEL%"=="" set "EMBED_MODEL=text-embedding-nomic-embed-text-v1.5@q8_0"
if "%SME_MODEL%"=="" set "SME_MODEL="
if "%REVIEW_MODEL%"=="" set "REVIEW_MODEL=%SME_MODEL%"
if "%DOCKER_CONTAINER%"=="" set "DOCKER_CONTAINER=pgvector17"
if "%LMSTUDIO_LOG_PATH%"=="" set "LMSTUDIO_LOG_PATH="

echo.
set "UPDATE_CHOICE="
set /p UPDATE_CHOICE=Update directories/settings? (Y/N): 
if /i "%UPDATE_CHOICE%"=="Y" goto CONFIGURE
if /i "%UPDATE_CHOICE%"=="YES" goto CONFIGURE
goto RUN

:CONFIGURE
echo.
echo Enter a new value, or press Enter to keep the current value shown in parentheses.
echo.

REM Capture current values as defaults (so Enter really keeps them)
set "DEF_RAG_ROOT=%RAG_ROOT%"
set "DEF_DOMAIN_DIR=%DOMAIN_DIR%"
set "DEF_N_ITEMS=%N_ITEMS%"
set "DEF_DB_DSN=%DB_DSN%"
set "DEF_DOCKER_CONTAINER=%DOCKER_CONTAINER%"
set "DEF_LM_URL=%LM_URL%"
set "DEF_EMBED_MODEL=%EMBED_MODEL%"
set "DEF_SME_MODEL=%SME_MODEL%"
set "DEF_REVIEW_MODEL=%REVIEW_MODEL%"
set "DEF_LMSTUDIO_LOG_PATH=%LMSTUDIO_LOG_PATH%"

set "TMP="
set /p TMP=Program folder (contains py files) (%DEF_RAG_ROOT%): 
if not "%TMP%"=="" set "RAG_ROOT=%TMP%"
if "%TMP%"=="" set "RAG_ROOT=%DEF_RAG_ROOT%"
cd /d "%RAG_ROOT%"

set "TMP="
set /p TMP=Domain folder (subject matter) (%DEF_DOMAIN_DIR%): 
if not "%TMP%"=="" set "DOMAIN_DIR=%TMP%"
if "%TMP%"=="" set "DOMAIN_DIR=%DEF_DOMAIN_DIR%"

set "TMP="
set /p TMP=How many items to generate? (%DEF_N_ITEMS%): 
if not "%TMP%"=="" set "N_ITEMS=%TMP%"
if "%TMP%"=="" set "N_ITEMS=%DEF_N_ITEMS%"

REM DB_DSN can contain !, so keep delayed expansion disabled (it is).
set "TMP="
set /p TMP=Postgres DSN (%DEF_DB_DSN%): 
if not "%TMP%"=="" set "DB_DSN=%TMP%"
if "%TMP%"=="" set "DB_DSN=%DEF_DB_DSN%"

set "TMP="
set /p TMP=Docker container name for pgvector logs (%DEF_DOCKER_CONTAINER%): 
if not "%TMP%"=="" set "DOCKER_CONTAINER=%TMP%"
if "%TMP%"=="" set "DOCKER_CONTAINER=%DEF_DOCKER_CONTAINER%"

set "TMP="
set /p TMP=LM Studio URL (%DEF_LM_URL%): 
if not "%TMP%"=="" set "LM_URL=%TMP%"
if "%TMP%"=="" set "LM_URL=%DEF_LM_URL%"

set "TMP="
set /p TMP=Embedding model (%DEF_EMBED_MODEL%): 
if not "%TMP%"=="" set "EMBED_MODEL=%TMP%"
if "%TMP%"=="" set "EMBED_MODEL=%DEF_EMBED_MODEL%"

set "TMP="
set /p TMP=SME model (%DEF_SME_MODEL%): 
if not "%TMP%"=="" set "SME_MODEL=%TMP%"
if "%TMP%"=="" set "SME_MODEL=%DEF_SME_MODEL%"

set "TMP="
set /p TMP=Review model (%DEF_REVIEW_MODEL%): 
if not "%TMP%"=="" set "REVIEW_MODEL=%TMP%"
if "%TMP%"=="" set "REVIEW_MODEL=%DEF_REVIEW_MODEL%"

set "TMP="
set /p TMP=LM Studio log file path (blank to skip snapshot) (%DEF_LMSTUDIO_LOG_PATH%): 
if not "%TMP%"=="" set "LMSTUDIO_LOG_PATH=%TMP%"
if "%TMP%"=="" set "LMSTUDIO_LOG_PATH=%DEF_LMSTUDIO_LOG_PATH%"

REM Persist exactly what will be used for this run
(
  echo @echo off
  echo REM Auto-generated. Edit if needed.
  echo.
  echo set "RAG_ROOT=%RAG_ROOT%"
  echo set "DOMAIN_DIR=%DOMAIN_DIR%"
  echo set "N_ITEMS=%N_ITEMS%"
  echo set "DB_DSN=%DB_DSN%"
  echo set "DOCKER_CONTAINER=%DOCKER_CONTAINER%"
  echo set "LM_URL=%LM_URL%"
  echo set "EMBED_MODEL=%EMBED_MODEL%"
  echo set "SME_MODEL=%SME_MODEL%"
  echo set "REVIEW_MODEL=%REVIEW_MODEL%"
  echo set "LMSTUDIO_LOG_PATH=%LMSTUDIO_LOG_PATH%"
) > "%CFG_FILE%"

call "%CFG_FILE%"

goto RUN

:RUN
REM Basic validation
if not exist "%RAG_ROOT%\cli.py" (
  echo Missing Python entrypoint: %RAG_ROOT%\cli.py
  pause
  exit /b 1
)
if not exist "%DOMAIN_DIR%" (
  echo Domain folder not found: %DOMAIN_DIR%
  pause
  exit /b 1
)

REM RUN_ID
set "RUN_ID="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmssZ')"` ) do set "RUN_ID=%%I"

set "LOG_DIR=%RAG_ROOT%\runs\logs_%RUN_ID%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "RUN_ID=%RUN_ID%"
set "LOG_DIR=%LOG_DIR%"

echo RUN_ID=%RUN_ID% > "%LOG_DIR%\run_info.txt"
echo RAG_ROOT=%RAG_ROOT% >> "%LOG_DIR%\run_info.txt"
echo DOMAIN_DIR=%DOMAIN_DIR% >> "%LOG_DIR%\run_info.txt"
echo N_ITEMS=%N_ITEMS% >> "%LOG_DIR%\run_info.txt"
echo DB_DSN=%DB_DSN% >> "%LOG_DIR%\run_info.txt"
echo LM_URL=%LM_URL% >> "%LOG_DIR%\run_info.txt"
echo EMBED_MODEL=%EMBED_MODEL% >> "%LOG_DIR%\run_info.txt"
echo SME_MODEL=%SME_MODEL% >> "%LOG_DIR%\run_info.txt"
echo REVIEW_MODEL=%REVIEW_MODEL% >> "%LOG_DIR%\run_info.txt"
echo set "LMSTUDIO_LOG_PATH=%LMSTUDIO_LOG_PATH%"

set "RUN_PIPE="
set /p RUN_PIPE=Attempt domain parsing and ingestion now (Y/N)? N to skip and generate from existing: 

if /i "%RUN_PIPE%"=="YES" set "RUN_PIPE=Y"
if /i "%RUN_PIPE%"=="Y" (
  python "%RAG_ROOT%\cli.py" pipeline 1> "%LOG_DIR%\console_pipeline.txt" 2>&1
) else (
  python "%RAG_ROOT%\cli.py" generate 1> "%LOG_DIR%\console_generate.txt" 2>&1
)

docker logs --tail 5000 "%DOCKER_CONTAINER%" 1> "%LOG_DIR%\docker_%DOCKER_CONTAINER%.log" 2>&1

if not "%LMSTUDIO_LOG_PATH%"=="" (
  powershell -NoProfile -Command "if (Test-Path -LiteralPath '%LMSTUDIO_LOG_PATH%') { Get-Content -LiteralPath '%LMSTUDIO_LOG_PATH%' -Tail 5000 }" 1> "%LOG_DIR%\lmstudio.log" 2>&1
)

echo.
echo Logs written to: %LOG_DIR%
echo.
pause
exit /b 0
