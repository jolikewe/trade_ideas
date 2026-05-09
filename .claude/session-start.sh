#!/usr/bin/env bash
set -euo pipefail

BRIEF=data/production/daily_brief.md
TODAY=$(date +%Y-%m-%d)

# --- CLAUDE.md ---
echo "=== CLAUDE.md ==="
cat CLAUDE.md

# --- README ---
echo ""
echo "=== README ==="
cat README.md

# --- Daily Brief ---
echo ""
if [ -f "$BRIEF" ] && grep -q "date: $TODAY" "$BRIEF" 2>/dev/null; then
    echo "=== Daily Brief (cached) ==="
    cat "$BRIEF"
else
    echo "=== Generating Daily Brief ==="
    python production/daily_run.py 2>/dev/null || echo "(daily_run skipped — run manually)"
    [ -f "$BRIEF" ] && cat "$BRIEF" || true
fi

# --- Session Context ---
echo ""
if [ -f ".session_context.md" ]; then
    echo "=== Session Context ==="
    cat .session_context.md
else
    echo "=== Session Context ==="
    echo "(none — .session_context.md does not exist yet)"
fi

echo ""
echo "Session loaded: CLAUDE.md | README | daily brief | session context"
