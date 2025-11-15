#!/usr/bin/env bash
set -euo pipefail

REPO_URL=${CYMINIBALL_REPO:-https://github.com/Ludwig-H/cyminiball}
REF=${CYMINIBALL_REF:-main}
WORKDIR=${CYMINIBALL_SRC:-$(pwd)/.deps/cyminiball}

mkdir -p "$(dirname "$WORKDIR")"
if [ ! -d "$WORKDIR/.git" ]; then
  echo "Cloning $REPO_URL into $WORKDIR" >&2
  git clone "$REPO_URL" "$WORKDIR"
else
  echo "Updating existing checkout in $WORKDIR" >&2
  git -C "$WORKDIR" fetch --tags --prune
fi

if git -C "$WORKDIR" rev-parse --verify --quiet "$REF" >/dev/null; then
  git -C "$WORKDIR" checkout "$REF"
else
  git -C "$WORKDIR" fetch "$REPO_URL" "$REF"
  git -C "$WORKDIR" checkout FETCH_HEAD
fi

if git -C "$WORKDIR" symbolic-ref -q HEAD >/dev/null; then
  git -C "$WORKDIR" pull --ff-only || true
fi

python -m pip install --upgrade "pip>=24.0"
python -m pip install --upgrade "setuptools>=68" wheel "Cython>=3.0" "numpy>=1.24"
python -m pip install --no-build-isolation "$WORKDIR"

COMMIT=$(git -C "$WORKDIR" rev-parse HEAD)
echo "Installed cyminiball from $REPO_URL@$COMMIT" >&2
