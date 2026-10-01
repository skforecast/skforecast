---
name: handoff
description: Writes a handoff document in dev/ so the work can continue in another session (cloud to local, local to cloud, another person, or a later session).
disable-model-invocation: true
argument-hint: "<topic slug>"
---

# Handoff

Write `dev/handoff_$ARGUMENTS.md` (ask for a slug if none was given; overwrite it if it already exists for this topic), in the language of the conversation:

1. **Title and state**: date, branch, and whether the changes are committed (and pushed) or only in the working tree.
2. **Goal and decisions**: what we are trying to achieve and why each non-obvious decision was taken, including the options discarded and any measurements that supported them.
3. **Done**: grouped by topic, with the public names and files that changed, and whether each block has tests and a release note in `docs/releases/releases.md`.
4. **Pending**: for the author (review, commit, decisions) and for the next session, in order, each item concrete enough to start without rereading this conversation.
5. **Verification**: the commands from `/verify` that check the current state, and their result at the end of this session (tests run, lint, `--check`).

Keep it factual and short; link to files instead of copying code.

Do not commit it unless asked. In a cloud session it must reach the remote to be read elsewhere: commit it to the session branch together with the rest of the work, and remind the user to delete it before the pull request is merged.
