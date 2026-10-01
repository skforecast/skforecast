---
name: open-pr
description: Prepares and opens a skforecast pull request against the current release branch, after running the tests and checks that CI does not run on release branches.
disable-model-invocation: true
argument-hint: "[short PR title]"
---

# Open a pull request

Title hint from the user: $ARGUMENTS

## 1. Branches

- Release branch: read `__version__` in `skforecast/__init__.py` and derive `X.Y.x` (e.g. `0.26.0` → `0.26.x`). The PR base is that branch, never `main`.
- If the current branch is `main` or the release branch, create a descriptive `feature/<topic>` or `fix/<topic>` branch first. In cloud sessions, keep the branch the session was started on.
- Review `git diff <release-branch>...HEAD --stat` and `git status`; uncommitted changes must be committed or left out on purpose (ask if unclear).

## 2. Checks

CI does not run unit tests on pull requests to release branches (only the AI context check), so the tests run here are the only ones before merge.

1. Run the `verify` skill on the whole branch (scope: `git diff --name-only <release-branch>...HEAD` plus the working tree) and stop on failures.
2. Always run `python tools/ai/generate_ai_context_files.py --check`, even if no AI context source changed (CI runs it on every PR and fails if a generated file is stale).
3. User-facing changes are described in `docs/releases/releases.md`, in the section of the version in development, with the right badge (Feature, Enhancement, API Change, Fix, Docs) and under Added, Changed or Fixed. If missing, draft the entry and show it to the user.
4. No `dev/handoff_*.md` file is part of the diff (delete it first, asking the user).

## 3. Open the PR

- Push the branch (`git push -u origin <branch>`; this asks for confirmation).
- Fill `.github/pull_request_template.md`: a Description of what changes and why (link the issue if any) and the checklist, ticking only what was actually verified.
- Create it with `gh pr create --base <release-branch> --title "<title>" --body-file <file>` (in cloud sessions use the session's PR flow if `gh` is not authenticated).
- Use the GitHub account configured for this repository on the current machine; do not switch accounts.
- The title, body and commits carry only the author's identity: no AI co-author trailer, session link, "Generated with" line or mention of Claude (a hook blocks them).

## 4. Report

Give the PR URL, the checks run with their results, and anything left for the reviewer.
