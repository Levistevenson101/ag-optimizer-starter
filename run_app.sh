#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate

# find first free port starting at 8501
PORT=8501
while nc -z localhost "$PORT" >/dev/null 2>&1; do
  PORT=$((PORT+1))
done

echo "✅ Starting Streamlit on http://localhost:$PORT"

exec python -m streamlit run src/app/app.py --server.headless true --server.port "$PORT"
