---
name: project-restructure
description: Analyzes a project's folder structure and file organization, proposes a cleaner/more scalable layout following stack-appropriate best practices, and reorganizes the project after explicit user confirmation. Use this when asked to clean up, reorganize, audit, or improve a project's folder structure.
license: MIT
---

# Project Restructure

## What I do
- Detect the project's stack and frameworks before suggesting anything
- Audit the current folder/file structure against best practices for that stack
- Find unclear file/folder names, duplicate or near-duplicate files, and structural violations
- Propose a plan — nothing is moved, renamed, or deleted until the user approves it
- After approval, execute the plan with git mv (preserves history) and fix any imports/paths the move breaks
- Report what changed and flag anything that needs manual follow-up

## When to use me
Use this when the user asks to clean up, reorganize, restructure, or audit a project's folder structure, or wants confusing file/folder names improved.

## Step 1 — Detect stack
Check for signals before assuming anything:
- pyproject.toml / equirements.txt / manage.py  Python. Check deps/imports for astapi, lask, django to narrow further.
- package.json  Node. Check dependencies for express, 
ext, eact, etc.
- go.mod  Go.
- Monorepo or mixed signals  ask the user which part of the project to focus on instead of guessing.

State the detected stack before moving on.

## Step 2 — Audit current structure
- Walk the project tree, respecting .gitignore.
- Flag file/folder names that don't communicate intent (utils2.py, 	emp.py, stuff/, inal_v2/).
- Flag duplicate or overlapping logic — compare actual responsibility, not just filename.
- Flag structural deviations from the detected stack's conventions. For FastAPI specifically: routers/endpoints, Pydantic schemas, business logic/services, DB models, and core config should be separated — flag if mixed together in one file or folder.
- Don't invent abstractions the project doesn't need. A 3-endpoint FastAPI app doesn't need a services/ layer just because bigger apps have one — only flag what's actually causing confusion at the project's current size.

## Step 3 — Propose, don't execute
Output the plan as a table:

| Current path | Proposed path | Why | Risk |
|---|---|---|---|

"Risk" = likelihood that imports, Docker volumes, CI configs, or hardcoded paths break.

For duplicates: never propose silent deletion. Identify which copy looks canonical and ask the user to confirm before removing the other.

Stop here and wait for explicit confirmation. Do not move, rename, or delete anything until the user approves the plan.

## Step 4 — Execute (only after confirmation)
- Use git mv instead of mv so history is preserved.
- After each move, grep the codebase for references to the old path/module and update imports.
- Run the test suite (or at minimum an import/syntax check if there are no tests) to confirm nothing broke.
- If something can't be safely auto-fixed (e.g. a path baked into .env or docker-compose.yml), stop and surface it rather than guessing.

## Step 5 — Summary
Report:
1. What was moved, renamed, or removed
2. What still needs manual attention
3. Anything flagged but deliberately left alone, and why
