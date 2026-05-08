#!/usr/bin/env bash
set -euo pipefail

BRIEF=data/production/daily_brief.md
TODAY=$(date +%Y-%m-%d)

if [ -f "$BRIEF" ] && grep -q "date: $TODAY" "$BRIEF" 2>/dev/null; then
    echo "=== Daily Brief (cached) ==="
    cat "$BRIEF"
else
    echo "=== Generating Daily Brief ==="
    python production/daily_run.py 2>/dev/null || echo "(daily_run skipped — run manually)"
    [ -f "$BRIEF" ] && cat "$BRIEF" || true
fi

echo ""
echo "=== README ==="
cat README.md

echo ""
echo "Confirm: README.md and daily brief loaded."
