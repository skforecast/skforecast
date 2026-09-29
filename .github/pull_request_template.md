## Description

<!-- What does this PR change and why? Link the related issue, for example "Closes #123". -->

## Checklist

- [ ] The PR targets the current release branch (for example `0.26.x`), not `main`.
- [ ] I have read and agree to the [Contributor License Agreement](https://github.com/skforecast/skforecast/blob/main/CONTRIBUTOR_LICENSE_AGREEMENT.md).
- [ ] Tests are added or updated, and the tests of the affected modules pass locally.
- [ ] Public functions and classes have NumPy-style docstrings.
- [ ] User-facing changes are described in [`docs/releases/releases.md`](https://github.com/skforecast/skforecast/blob/main/docs/releases/releases.md).
- [ ] If the public API changed, the AI context sources are updated (`tools/ai/llms-base.txt`, `skills/`) and `python tools/ai/generate_ai_context_files.py --check` passes.
- [ ] If the documentation changed, it builds locally with `mkdocs serve`.
