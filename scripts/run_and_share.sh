#!/usr/bin/env bash
# Run the manga-ai pipeline (or one stage), snapshot a small result set into
# results/, and push it so a reviewer can pull and inspect.
#
# Usage examples:
#   ./scripts/run_and_share.sh                              # full run, defaults below
#   ./scripts/run_and_share.sh --stage translate           # translation only (fast)
#   ./scripts/run_and_share.sh --stage clean --force        # re-clean, ignore cache
#   ./scripts/run_and_share.sh --chapter "data/chapters/Ch.133 - ..." --config hybrid_qwen_gemma_it
#   ./scripts/run_and_share.sh --no-push                    # run + collect, don't push
#
# Run it from Git Bash on Windows, or any Linux/macOS shell.
set -uo pipefail

# ---- defaults (edit these) ------------------------------------------------
CHAPTER="data/chapters/Ch.133 - Suddenly Became A Princess One Day"
CONFIG="hybrid_qwen_nllb"     # bare config name under configs/
# transtest = translator-only smoke test (no OCR needed); others run the pipeline
STAGE="full"                      # full | detect | ocr | translate | mask | clean | transtest
LIMIT=10                          # full run: only first N pages (0 = whole chapter)
SAMPLES=6
CROPS=40
PUSH=1
FORCE=""
PYTHON="${PYTHON:-python}"
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chapter) CHAPTER="$2"; shift 2;;
    --config)  CONFIG="$2";  shift 2;;
    --stage)   STAGE="$2";   shift 2;;
    --samples) SAMPLES="$2"; shift 2;;
    --crops)   CROPS="$2"; shift 2;;
    --limit)   LIMIT="$2"; shift 2;;
    --force)   FORCE="--force"; shift;;
    --no-push) PUSH=0; shift;;
    --python)  PYTHON="$2"; shift 2;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

# cd to repo root (parent of this script's dir)
cd "$(dirname "$0")/.." || exit 1

# Special: translator-only smoke test (no OCR / chapter output needed)
if [[ "$STAGE" == "transtest" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
  echo "== repo:   $(pwd)"; echo "== branch: $BRANCH"
  echo "== stage:  transtest (translator smoke test)"; echo "== config: $CONFIG"
  echo "-- git pull --rebase --autostash origin $BRANCH"
  git pull --rebase --autostash origin "$BRANCH" || echo "!! pull failed — continuing"
  rm -rf results/latest && mkdir -p results/latest
  LOG="$(mktemp)"
  set -o pipefail
  $PYTHON scripts/smoke_translate.py -c "$CONFIG" --out results/latest 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  cp "$LOG" results/latest/run.txt 2>/dev/null || true
  rm -f "$LOG"
  if [[ "$PUSH" -eq 1 ]]; then
    git add -A results/
    if git diff --cached --quiet; then echo "-- nothing to commit"; else
      git commit -m "transtest: $CONFIG @ $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null
      for i in 1 2 3 4; do
        git push origin "HEAD:$BRANCH" && { echo "-- pushed"; break; } \
          || { echo "!! push retry $i"; git pull --rebase --autostash origin "$BRANCH" || true; sleep $((2 ** i)); }
      done
    fi
  fi
  echo "== done (transtest rc=$RC). See results/latest/translation_smoke.txt"
  exit "$RC"
fi

# map stage -> CLI subcommand
case "$STAGE" in
  full)      CMD="process";;
  detect)    CMD="detect";;
  ocr)       CMD="ocr";;
  translate) CMD="translate";;
  mask)      CMD="mask";;
  clean)     CMD="clean";;
  *) echo "Unknown stage: $STAGE (use full|detect|ocr|translate|mask|clean)" >&2; exit 2;;
esac

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
echo "== repo:   $(pwd)"
echo "== branch: $BRANCH"
echo "== stage:  $STAGE  (manga-ai $CMD)"
echo "== config: $CONFIG"
echo "== chapter:$CHAPTER"

# 1) pull latest code first (so you're running the reviewer's changes)
echo "-- git pull --rebase --autostash origin $BRANCH"
git pull --rebase --autostash origin "$BRANCH" || echo "!! pull failed (offline or conflict) — continuing with local code"

# 2) run the pipeline, capturing a log
# --limit applies to every image stage (not translate, which has no pages)
LIMIT_ARG=""
if [[ "$CMD" != "translate" && "${LIMIT:-0}" -gt 0 ]]; then
  LIMIT_ARG="--limit $LIMIT"
fi
LOG="$(mktemp)"
echo "-- running: $PYTHON -m manga_ai $CMD \"$CHAPTER\" -c $CONFIG $FORCE $LIMIT_ARG"
set -o pipefail
$PYTHON -m manga_ai "$CMD" "$CHAPTER" -c "$CONFIG" $FORCE $LIMIT_ARG 2>&1 | tee "$LOG"
RUN_RC=${PIPESTATUS[0]}
echo "-- pipeline exit code: $RUN_RC"

# 3) collect a small shareable snapshot
echo "-- collecting results"
$PYTHON scripts/collect_results.py --chapter "$CHAPTER" --config "$CONFIG" --out results/latest --samples "$SAMPLES" --crops "$CROPS" || {
  echo "!! collect_results.py failed" >&2; }
cp "$LOG" results/latest/run.txt 2>/dev/null || true
rm -f "$LOG"

# 4) commit + push
if [[ "$PUSH" -eq 1 ]]; then
  git add -A results/
  if git diff --cached --quiet; then
    echo "-- nothing new in results/ to commit"
  else
    STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    git commit -m "run results: $STAGE / $CONFIG @ $STAMP" >/dev/null
    echo "-- pushing results to origin/$BRANCH"
    for i in 1 2 3 4; do
      if git push origin "HEAD:$BRANCH"; then
        echo "-- pushed. Reviewer can: git pull && open results/latest/SUMMARY.md"
        break
      fi
      echo "!! push failed (attempt $i). Rebasing on remote and retrying..."
      git pull --rebase --autostash origin "$BRANCH" || true
      sleep $((2 ** i))
    done
  fi
else
  echo "-- --no-push set; results are in results/latest/ (not pushed)"
fi

echo "== done (pipeline rc=$RUN_RC). See results/latest/SUMMARY.md"
exit "$RUN_RC"
