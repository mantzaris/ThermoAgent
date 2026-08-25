#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT/.venv/bin/python"
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON=python3
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
cd "$ROOT"
"$PYTHON_BIN" -m pytest --import-mode=importlib -q tests/statmech_v15

# Historical manuscript tests intentionally compile in place.  Run every
# pre-V15 suite in a sparse disposable worktree so differing TeX engines can
# never modify frozen PDFs or leave ignored intermediates in the review tree.
HISTORICAL_ROOT="$(mktemp -d /tmp/thermoagent-v15-historical-tests.XXXXXX)"
cleanup_historical_worktree() {
  git -C "$ROOT" worktree remove --force "$HISTORICAL_ROOT" >/dev/null 2>&1 || true
  # If registration itself failed, only remove the still-empty mktemp shell;
  # never recursively delete a partially populated diagnostic worktree.
  rmdir "$HISTORICAL_ROOT" >/dev/null 2>&1 || true
}
trap cleanup_historical_worktree EXIT
git -C "$ROOT" worktree add --quiet --detach --no-checkout "$HISTORICAL_ROOT" HEAD
# Git 2.25 does not create the per-worktree ``info`` directory for a
# ``--no-checkout`` worktree.  Its sparse-checkout command nevertheless writes
# the lock file there.  Create only this disposable metadata directory so the
# isolation works on both the original workstation Git and newer RunPod Git.
HISTORICAL_GIT_DIR="$(git -C "$HISTORICAL_ROOT" rev-parse --git-dir)"
mkdir -p "$HISTORICAL_GIT_DIR/info"
git -C "$HISTORICAL_ROOT" sparse-checkout init --cone
git -C "$HISTORICAL_ROOT" sparse-checkout set \
  configs \
  paper \
  tests \
  thermoagent \
  results/llm_agent_entropy_v10 \
  results/llm_agent_entropy_v11 \
  results/llm_agent_statmech_v12 \
  results/collective_agent_statmech_v13 \
  results/collective_agent_statmech_v14
git -C "$HISTORICAL_ROOT" checkout --quiet HEAD
cd "$HISTORICAL_ROOT"
export PYTHONPATH="$HISTORICAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
for suite in statmech_v10 statmech_v11 statmech_v12 statmech_v13; do
  "$PYTHON_BIN" -m pytest --import-mode=importlib -q "tests/$suite"
done
# This V14-only assertion compares its checkout with the V13 parent and, by
# construction, rejects every later V15 path.  Preserve the historical test
# unchanged; the V15 suite above replaces it with a reconstruction-base
# immutability check that rejects changes to all pre-V15 namespaces.
"$PYTHON_BIN" -m pytest --import-mode=importlib -q tests/statmech_v14 \
  -k 'not test_v1_through_v13_namespaces_are_immutable'

cd "$ROOT"
cleanup_historical_worktree
trap - EXIT
