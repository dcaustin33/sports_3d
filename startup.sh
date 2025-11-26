echo "Running in web environment..."

if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

python -m venv .venv
source .venv/bin/activate
pip install uv
uv sync