# Tools

Development scripts and notebooks. They are not part of the skforecast package.

| Path | Purpose |
|:-----|:--------|
| [`ai/`](ai/) | Generates the AI context files (`AGENTS.md`, `llms-full.txt`, ...). See its README. |
| [`docs/`](docs/) | Scripts and notebooks used to build and maintain the documentation (see below). |
| `check_case_in_file_names.py` | Detects test and fixture files whose names break the lowercase convention. |
| `check_foundation_models_metadata.py` | Compares the license, weights repository and gating declared for each foundation model with the Hugging Face Hub. Runs weekly in CI. |

## docs/

| Path | Purpose |
|:-----|:--------|
| [`animations_to_video/`](docs/animations_to_video/) | Exports the documentation animations (`docs/animations/`) to MP4 videos. See its README. |
| [`execute_notebooks/`](docs/execute_notebooks/) | Re-executes the documentation notebooks and replaces their widgets with static output. |
| [`figures/`](docs/figures/) | Notebooks that create GIFs and figures used in the documentation. |
| [`home_page/`](docs/home_page/) | Generates the data and social card of the documentation home page. See its README. |
| [`hooks/`](docs/hooks/) | MkDocs hooks, loaded from `mkdocs.yml`. |
| `check_published_links.ipynb` | Crawls the published website and reports broken links. |
| `vendor_katex.py` | Downloads KaTeX into `docs/vendor/katex`, so math renders without a CDN. |
