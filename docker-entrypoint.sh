#!/usr/bin/env sh
set -eu

is_true() {
  case "${1:-}" in
    true|TRUE|1|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

if is_true "${WAIT_FOR_QDRANT:-true}" && [ -n "${QDRANT_URL:-}" ]; then
  uv run python - <<'PY'
import os
import time

from qdrant_client import QdrantClient

url = os.environ["QDRANT_URL"]
api_key = os.environ.get("QDRANT_API_KEY") or None

for attempt in range(1, 31):
    try:
        QdrantClient(url=url, api_key=api_key).get_collections()
        print(f"[docker] Qdrant reachable: {url}")
        break
    except Exception as exc:
        if attempt == 30:
            raise
        print(f"[docker] Waiting for Qdrant ({attempt}/30): {exc}")
        time.sleep(2)
PY
fi

if is_true "${SKIP_DEPLOY_INGEST:-false}"; then
  echo "[docker] Skipping deployment ingestion because SKIP_DEPLOY_INGEST=true"
else
  ingest_args=""
  if is_true "${INGEST_REBUILD_PAGEINDEX:-true}"; then
    ingest_args="$ingest_args --rebuild-pageindex"
  fi
  if is_true "${INGEST_INCLUDE_FAQ:-true}"; then
    ingest_args="$ingest_args --include-faq"
  fi

  echo "[docker] Running deployment ingestion: scripts/deploy_ingest.py$ingest_args"
  # shellcheck disable=SC2086
  uv run python scripts/deploy_ingest.py $ingest_args
fi

exec "$@"
