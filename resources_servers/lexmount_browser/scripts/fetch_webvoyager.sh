#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fetch the official WebVoyager tasks and reproduce the validated
# "webvoyager-clean" 168-task training set used by the 0721 growth-curve run.
#
#   1. Download data/WebVoyager_data.jsonl from the official MIT-licensed repo
#      (MinorJerry/WebVoyager), pinned to a commit, and verify its SHA-256.
#   2. Run scripts/convert_webvoyager.py --clean, which:
#        - drops "Cambridge Dictionary"            (643 -> 600 tasks, SHA-checked)
#        - keeps ArXiv / BBC News / Coursera / GitHub (600 -> 168 tasks, SHA-checked,
#          byte-identical to the validated run's task manifest)
#        - cross-checks the IDs against data/webvoyager_clean_task_ids.txt
#        - converts to this env's rollout-input format with judge:true
#   3. Print row counts + SHA-256 of every artifact.
#
# Outputs (under data/):
#   webvoyager_data_upstream.jsonl   raw upstream download          (643 rows)
#   webvoyager_clean_source.jsonl    cleaned raw task set           (168 rows)
#   webvoyager_clean.jsonl           training input for this env    (168 rows)
#
# Usage:  bash scripts/fetch_webvoyager.sh
# ---------------------------------------------------------------------------
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$ENV_DIR/data"

# Pinned upstream source (MIT license; see the WebVoyager repo).
UPSTREAM_COMMIT="091544539eba485dbd74ef3742011ddeede37336"
UPSTREAM_URL="https://raw.githubusercontent.com/MinorJerry/WebVoyager/${UPSTREAM_COMMIT}/data/WebVoyager_data.jsonl"
UPSTREAM_SHA256="69b19fd86c23f1a500244a3724e039aa7ca6a1223d03e11eb10e308d4f11c488"

UPSTREAM_OUT="$DATA_DIR/webvoyager_data_upstream.jsonl"
CLEAN_SOURCE_OUT="$DATA_DIR/webvoyager_clean_source.jsonl"
CLEAN_OUT="$DATA_DIR/webvoyager_clean.jsonl"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

echo "==> Downloading WebVoyager tasks (pinned to ${UPSTREAM_COMMIT:0:12})"
if [[ -f "$UPSTREAM_OUT" && "$(sha256 "$UPSTREAM_OUT")" == "$UPSTREAM_SHA256" ]]; then
  echo "    cached: $UPSTREAM_OUT"
else
  curl -fsSL --retry 3 -o "$UPSTREAM_OUT" "$UPSTREAM_URL"
fi
actual="$(sha256 "$UPSTREAM_OUT")"
if [[ "$actual" != "$UPSTREAM_SHA256" ]]; then
  echo "ERROR: upstream SHA-256 mismatch: got $actual, expected $UPSTREAM_SHA256" >&2
  exit 1
fi
echo "    sha256 OK: $UPSTREAM_SHA256"

echo "==> Reproducing the validated webvoyager-clean 168-task set"
python3 "$SCRIPT_DIR/convert_webvoyager.py" \
  --clean --judge \
  --source "$UPSTREAM_OUT" \
  --output "$CLEAN_OUT" \
  --clean-source-out "$CLEAN_SOURCE_OUT"

echo "==> Artifact summary"
for f in "$UPSTREAM_OUT" "$CLEAN_SOURCE_OUT" "$CLEAN_OUT"; do
  rows="$(grep -c . "$f")"   # non-empty lines (upstream file has no trailing newline)
  printf '    %-34s rows=%-4s sha256=%s\n' "$(basename "$f")" "$rows" "$(sha256 "$f")"
done
echo "Done. Training input: $CLEAN_OUT (verifier_metadata.judge=true on every row)"
