#!/bin/zsh

set -e

########################################
# Defaults
########################################

RAG_ROOT="$HOME/Documents/Programming/Capstone/KinaxisCapstone/_rag_testGen"
DOMAIN_DIR="$HOME/Documents/Programming/Capstone/KinaxisCapstone/example1"
N_ITEMS="5"
DB_DSN="postgresql://postgres:postgres@localhost:5435/kinaxis_ragtestdb"
LM_URL="http://localhost:1234"
EMBED_MODEL="text-embedding-nomic-embed-text-v1.5@q8_0"
CONTEXT_MODEL="qwen2.5-7b-instruct-uncensored"
GENERATOR_MODEL="qwen2.5-7b-instruct-uncensored"
REVIEW_MODEL="qwen2.5-7b-instruct-uncensored"
DOCKER_CONTAINER="pgvector17"
LMSTUDIO_LOGPATH="$HOME/Library/Logs/LM Studio/main.log"

CFG_FILE="./config.env"

########################################
# Load config if exists
########################################

if [[ -f "$CFG_FILE" ]]; then
  source "$CFG_FILE"
fi

########################################
# Configure
########################################

configure() {
  echo ""
  echo "Press Enter to keep current value."

  read "tmp?Program folder (RAG_ROOT) [$RAG_ROOT]: "
  [[ -n "$tmp" ]] && RAG_ROOT="$tmp"

  read "tmp?Domain folder [$DOMAIN_DIR]: "
  [[ -n "$tmp" ]] && DOMAIN_DIR="$tmp"

  read "tmp?Number of items [$N_ITEMS]: "
  [[ -n "$tmp" ]] && N_ITEMS="$tmp"

  read "tmp?Postgres DSN [$DB_DSN]: "
  [[ -n "$tmp" ]] && DB_DSN="$tmp"

  read "tmp?Docker container name [$DOCKER_CONTAINER]: "
  [[ -n "$tmp" ]] && DOCKER_CONTAINER="$tmp"

  read "tmp?LM Studio URL [$LM_URL]: "
  [[ -n "$tmp" ]] && LM_URL="$tmp"

  read "tmp?LM Studio log path [$LMSTUDIO_LOGPATH]: "
  [[ -n "$tmp" ]] && LMSTUDIO_LOGPATH="$tmp"

  read "tmp?Embedding model [$EMBED_MODEL]: "
  [[ -n "$tmp" ]] && EMBED_MODEL="$tmp"

  read "tmp?Generator model [$GENERATOR_MODEL]: "
  [[ -n "$tmp" ]] && GENERATOR_MODEL="$tmp"

  read "tmp?Context model [$CONTEXT_MODEL]: "
  [[ -n "$tmp" ]] && CONTEXT_MODEL="$tmp"

  read "tmp?Review model [$REVIEW_MODEL]: "
  [[ -n "$tmp" ]] && REVIEW_MODEL="$tmp"

  cat > "$CFG_FILE" <<EOF
RAG_ROOT=$RAG_ROOT
DOMAIN_DIR=$DOMAIN_DIR
N_ITEMS=$N_ITEMS
DB_DSN=$DB_DSN
LM_URL=$LM_URL
EMBED_MODEL=$EMBED_MODEL
CONTEXT_MODEL=$CONTEXT_MODEL
GENERATOR_MODEL=$GENERATOR_MODEL
REVIEW_MODEL=$REVIEW_MODEL
DOCKER_CONTAINER=$DOCKER_CONTAINER
LMSTUDIO_LOGPATH=$LMSTUDIO_LOGPATH
EOF

  echo "Configuration saved."
}

########################################
# Main Run Logic
########################################

run_pipeline() {

  if [[ ! -f "$RAG_ROOT/cli.py" ]]; then
    echo "Missing cli.py in $RAG_ROOT"
    exit 1
  fi

  if [[ ! -d "$DOMAIN_DIR" ]]; then
    echo "Domain folder not found: $DOMAIN_DIR"
    exit 1
  fi

  RUN_ID=$(date -u +"%Y%m%d_%H%M%SZ")
  LOG_DIR="$RAG_ROOT/runs/logs_$RUN_ID"
  mkdir -p "$LOG_DIR"

  echo ""
  echo "F = Full pipeline"
  echo "I = Ingest only"
  echo "G = Generate only"
  read "choice?Choice (F/I/G): "

  echo "RUN_ID=$RUN_ID" > "$LOG_DIR/run_info.txt"
  echo "RAG_ROOT=$RAG_ROOT" >> "$LOG_DIR/run_info.txt"
  echo "DOMAIN_DIR=$DOMAIN_DIR" >> "$LOG_DIR/run_info.txt"

  RUN_START=$(date -u +"%Y%m%d_%H%M%SZ")
  echo "RUN_START=$RUN_START" >> "$LOG_DIR/run_info.txt"
  echo ""
  echo "Started: $RUN_START"

  ########################################
  # ✅ EXPORT ALL VARIABLES FOR PYTHON
  ########################################
  export RAG_ROOT
  export DOMAIN_DIR
  export N_ITEMS
  export DB_DSN
  export LM_URL
  export EMBED_MODEL
  export CONTEXT_MODEL
  export GENERATOR_MODEL
  export REVIEW_MODEL
  export DOCKER_CONTAINER
  export LMSTUDIO_LOGPATH

  if [[ "$choice" == "F" || "$choice" == "f" ]]; then
    python "$RAG_ROOT/cli.py" pipeline --force-ingest --clear-first \
      > "$LOG_DIR/console_pipeline.txt" 2>&1

  elif [[ "$choice" == "I" || "$choice" == "i" ]]; then
    python "$RAG_ROOT/cli.py" pipeline --force-ingest --clear-first --ingest-only \
      > "$LOG_DIR/console_pipeline.txt" 2>&1

  elif [[ "$choice" == "G" || "$choice" == "g" ]]; then
    python "$RAG_ROOT/cli.py" generate \
      > "$LOG_DIR/console_generate.txt" 2>&1

  else
    echo "Invalid choice."
    return
  fi

  RUN_END=$(date -u +"%Y%m%d_%H%M%SZ")
  echo "RUN_END=$RUN_END" >> "$LOG_DIR/run_info.txt"

  echo ""
  echo "Finished: $RUN_END"

  docker logs --tail 5000 "$DOCKER_CONTAINER" \
    > "$LOG_DIR/docker_$DOCKER_CONTAINER.log" 2>&1 || true

  if [[ -f "$LMSTUDIO_LOGPATH" ]]; then
    tail -n 5000 "$LMSTUDIO_LOGPATH" \
      > "$LOG_DIR/lmstudio.log" 2>&1
  fi

  echo ""
  read "again?Run again? (Y/N): "
  if [[ "$again" == "Y" || "$again" == "y" ]]; then
    run_pipeline
  fi
}

########################################
# Entry
########################################

echo ""
read "update?Update settings? (Y/N): "
if [[ "$update" == "Y" || "$update" == "y" ]]; then
  configure
fi

run_pipeline