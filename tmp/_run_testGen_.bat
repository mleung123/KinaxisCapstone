@echo off
setlocal EnableExtensions DisableDelayedExpansion

REM Program root
set "RAG_ROOT=C:\Users\kadek\Documents\GitHub\KinaxisCapstone\_rag_testGen"
cd /d "%RAG_ROOT%"

REM Load persisted configuration (if present)
set "CFG_FILE=%RAG_ROOT%\config.bat"
if exist "%CFG_FILE%" call "%CFG_FILE%"

REM Defaults (only if still empty)
if "%DOMAIN_DIR%"=="" set "DOMAIN_DIR=%RAG_ROOT%"
if "%N_ITEMS%"=="" set "N_ITEMS=5"
if "%DB_DSN%"=="" set "DB_DSN=postgresql://postgres:postgres@localhost:5435/kinaxis_ragtestdb"
if "%LM_URL%"=="" set "LM_URL=http://localhost:1234"
if "%EMBED_MODEL%"=="" set "EMBED_MODEL=text-embedding-nomic-embed-text-v1.5@q8_0"
if "%SME_MODEL%"=="" set "SME_MODEL="
if "%REVIEW_MODEL%"=="" set "REVIEW_MODEL=%SME_MODEL%"
if "%DOCKER_CONTAINER%"=="" set "DOCKER_CONTAINER=pgvector17"
if "%LMSTUDIO_LOGPATH%"=="" set "LMSTUDIO_LOGPATH="

:RUN_AGAIN

echo.
set "UPDATE_CHOICE="
set /p UPDATE_CHOICE=Update directories/settings? (Y/N): 
if /i "%UPDATE_CHOICE%"=="Y" goto CONFIGURE
if /i "%UPDATE_CHOICE%"=="YES" goto CONFIGURE
goto RUN

:CONFIGURE
echo.
echo Enter a new value, or press Enter to keep the current value shown in parentheses.

set "DEF_RAG_ROOT=%RAG_ROOT%"
set "DEF_DOMAIN_DIR=%DOMAIN_DIR%"
set "DEF_N_ITEMS=%N_ITEMS%"
set "DEF_DB_DSN=%DB_DSN%"
set "DEF_DOCKER_CONTAINER=%DOCKER_CONTAINER%"
set "DEF_LM_URL=%LM_URL%"
set "DEF_EMBED_MODEL=%EMBED_MODEL%"
set "DEF_SME_MODEL=%SME_MODEL%"
set "DEF_REVIEW_MODEL=%REVIEW_MODEL%"
set "DEF_LMSTUDIO_LOGPATH=%LMSTUDIO_LOGPATH%"

echo.
set "TMP="
echo Update program folder (contains py files)?
echo Current stored value: %DEF_RAG_ROOT%
set /p TMP=New value (or enter to keep): 
if not "%TMP%"=="" set "RAG_ROOT=%TMP%"
if "%TMP%"=="" set "RAG_ROOT=%DEF_RAG_ROOT%"
cd /d "%RAG_ROOT%"

echo.
set "TMP="
echo Update domain folder (subject matter)?
echo Current stored value: %DEF_DOMAIN_DIR%
set /p TMP=New value (or enter to keep): 
if not "%TMP%"=="" set "DOMAIN_DIR=%TMP%"
if "%TMP%"=="" set "DOMAIN_DIR=%DEF_DOMAIN_DIR%"

echo.
set "TMP="
echo How many items to generate?
echo Current stored value: %DEF_N_ITEMS%
set /p TMP=New value (or enter to keep): 
if not "%TMP%"=="" set "N_ITEMS=%TMP%"
if "%TMP%"=="" set "N_ITEMS=%DEF_N_ITEMS%"

echo.
set "TMP="
echo Update Postgres DSN?
echo Current stored value: %DEF_DB_DSN%
set /p TMP=New value (or enter to keep): 
if not "%TMP%"=="" set "DB_DSN=%TMP%"
if "%TMP%"=="" set "DB_DSN=%DEF_DB_DSN%"

echo.
set "TMP="
echo Update Docker container name for pgvector logs?
echo Current stored value: %DEF_DOCKER_CONTAINER%
set /p TMP=New value (or enter to keep): 
if not "%TMP%"=="" set "DOCKER_CONTAINER=%TMP%"
if "%TMP%"=="" set "DOCKER_CONTAINER=%DEF_DOCKER_CONTAINER%"

echo.
set "TMP="
echo Update LM Studio URL?
echo Current stored value: %DEF_LM_URL%
set /p TMP=New value (or enter to keep): 
if not "%TMP%"=="" set "LM_URL=%TMP%"
if "%TMP%"=="" set "LM_URL=%DEF_LM_URL%"

echo.
set "TMP="
echo Update LM Studio log path?
echo Current stored value: %DEF_LMSTUDIO_LOGPATH%
set /p TMP=New value (or enter to keep): 
if not "%TMP%"=="" set "LMSTUDIO_LOGPATH=%TMP%"
if "%TMP%"=="" set "LMSTUDIO_LOGPATH=%DEF_LMSTUDIO_LOGPATH%"

echo.
set "TMP="
echo Update embedding model?
echo Current stored value: %DEF_EMBED_MODEL%
set /p TMP=New value (or enter to keep): 
if not "%TMP%"=="" set "EMBED_MODEL=%TMP%"
if "%TMP%"=="" set "EMBED_MODEL=%DEF_EMBED_MODEL%"

echo.
set "TMP="
echo Update SME model?
echo Current stored value: %DEF_SME_MODEL%
set /p TMP=New value (or enter to keep): 
if not "%TMP%"=="" set "SME_MODEL=%TMP%"
if "%TMP%"=="" set "SME_MODEL=%DEF_SME_MODEL%"

echo.
set "TMP="
echo Update review model?
echo Current stored value: %DEF_REVIEW_MODEL%
set /p TMP=New value (or enter to keep): 
if not "%TMP%"=="" set "REVIEW_MODEL=%TMP%"
if "%TMP%"=="" set "REVIEW_MODEL=%DEF_REVIEW_MODEL%"

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
  echo set "LMSTUDIO_LOGPATH=%LMSTUDIO_LOGPATH%"
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

echo.
set "RUN_PIPE="
set /p RUN_PIPE=Force re-ingestion of domain files? (Y = pipeline with --force-ingest / N = generate only): 

echo RUN_ID=%RUN_ID% > "%LOG_DIR%\run_info.txt"
echo UPDATE_CHOICE=%UPDATE_CHOICE% >> "%LOG_DIR%\run_info.txt"
echo RUN_PIPE=%RUN_PIPE% >> "%LOG_DIR%\run_info.txt"
echo RAG_ROOT=%RAG_ROOT% >> "%LOG_DIR%\run_info.txt"
echo DOMAIN_DIR=%DOMAIN_DIR% >> "%LOG_DIR%\run_info.txt"
echo N_ITEMS=%N_ITEMS% >> "%LOG_DIR%\run_info.txt"
echo DB_DSN=%DB_DSN% >> "%LOG_DIR%\run_info.txt"
echo LM_URL=%LM_URL% >> "%LOG_DIR%\run_info.txt"
echo EMBED_MODEL=%EMBED_MODEL% >> "%LOG_DIR%\run_info.txt"
echo SME_MODEL=%SME_MODEL% >> "%LOG_DIR%\run_info.txt"
echo REVIEW_MODEL=%REVIEW_MODEL% >> "%LOG_DIR%\run_info.txt"
echo LMSTUDIO_LOGPATH=%LMSTUDIO_LOGPATH% >> "%LOG_DIR%\run_info.txt"

echo.
echo Directories/settings:
echo RUN_ID=%RUN_ID%
echo =====================
echo RAG_ROOT=%RAG_ROOT%
echo DOMAIN_DIR=%DOMAIN_DIR%
echo N_ITEMS=%N_ITEMS%
echo DB_DSN=%DB_DSN%
echo DOCKER_CONTAINER=%DOCKER_CONTAINER%
echo LM_URL=%LM_URL%
echo EMBED_MODEL=%EMBED_MODEL%
echo SME_MODEL=%SME_MODEL%
echo REVIEW_MODEL=%REVIEW_MODEL%
echo LMSTUDIO_LOGPATH=%LMSTUDIO_LOGPATH%
echo RAG_TESTGEN_LOGPATH: %LOG_DIR%

set "RUN_START="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmssZ')"`) do set "RUN_START=%%I"
echo RUN_START=%RUN_START% >> "%LOG_DIR%\run_info.txt"
echo.
echo Started:  %RUN_START%

if /i "%RUN_PIPE%"=="Y" (
  python "%RAG_ROOT%\cli.py" pipeline --force-ingest --clear-first 1> "%LOG_DIR%\console_pipeline.txt"
) else (
  python "%RAG_ROOT%\cli.py" generate 1> "%LOG_DIR%\console_generate.txt"
)

set "RUN_END="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmssZ')"`) do set "RUN_END=%%I"
echo RUN_END=%RUN_END% >> "%LOG_DIR%\run_info.txt"
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "$s=[datetime]::ParseExact('%RUN_START%','yyyyMMdd_HHmmssZ',$null);$e=[datetime]::ParseExact('%RUN_END%','yyyyMMdd_HHmmssZ',$null);$d=$e-$s;'{0}h {1}m {2}s' -f [int]$d.TotalHours,[int]$d.Minutes,$d.Seconds"`) do set "RUN_DURATION=%%D"
echo RUN_DURATION=%RUN_DURATION% >> "%LOG_DIR%\run_info.txt"
echo.
echo Finished: %RUN_END%
echo Duration: %RUN_DURATION%

docker logs --tail 5000 "%DOCKER_CONTAINER%" 1> "%LOG_DIR%\docker_%DOCKER_CONTAINER%.log" 2>&1

if not "%LMSTUDIO_LOGPATH%"=="" (
  powershell -NoProfile -Command "if (Test-Path -LiteralPath '%LMSTUDIO_LOGPATH%') { Get-Content -LiteralPath '%LMSTUDIO_LOGPATH%' -Tail 5000 }" 1> "%LOG_DIR%\lmstudio.log" 2>&1
)

echo.
pause

:ASK_AGAIN
echo.
set "RUN_AGAIN="
set /p RUN_AGAIN=Run again? (Y/N): 

if /i "%RUN_AGAIN%"=="Y" goto RUN_AGAIN
if /i "%RUN_AGAIN%"=="YES" goto RUN_AGAIN

exit /b 0