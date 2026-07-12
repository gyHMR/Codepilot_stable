---
name: demo-review
version: 1.0.0
description: Use a compact review checklist for a code change.
command: demo-review
allowed_modes:
  - plan
  - execute
required_tools:
  - read_file
---

# Demo Review Checklist

When this skill is applied, answer with three short sections:

1. Goal: restate the user request in one sentence.
2. Risk: list the most likely implementation risk.
3. Verification: name the smallest useful check to run.

Read `references/checklist.md` when a more detailed review checklist is useful.
