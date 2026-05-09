---
name: No unsolicited git commits
description: Never run git add/commit/push unless the user explicitly asks
type: feedback
---

Never run git add, git commit, or git push unless the user explicitly requests it.

**Why:** User was surprised by an unsolicited commit and rejected it.

**How to apply:** Treat all git write operations as requiring explicit instruction every time. "stage and commit this" or "commit these changes" are the triggers. Finishing a task is not.
