echo "Running in web environment..."

if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

python -m venv .venv
source .venv/bin/activate
pip install uv
uv sync
uv pip install \
  --no-build-isolation \
  --no-deps \
  "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9"
uv pip install "git+https://github.com/microsoft/MoGe.git"