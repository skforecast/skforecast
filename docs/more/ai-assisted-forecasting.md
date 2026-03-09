# Using AI assistants with skforecast

Skforecast provides machine-readable context files so that AI coding assistants — ChatGPT, Claude, Gemini, GitHub Copilot, Cursor, and others — can generate **accurate, up-to-date code** for time series forecasting.


## Quick start: paste a URL into any LLM

Copy the following URL and paste it into any AI chat:

```
https://skforecast.org/latest/llms-full.txt
```

This single file (~2 500 lines) contains the complete API reference, all forecaster signatures, workflow examples, and best practices. It gives any LLM enough context to help you with skforecast without hallucinating deprecated methods or wrong parameter names.

**Example prompt:**

> Using the context from https://skforecast.org/latest/llms-full.txt, create a ForecasterRecursiveMultiSeries with LightGBM, fit it on a DataFrame with 3 series, and run backtesting with conformal prediction intervals.


## IDE integration (automatic)

If you clone or open the skforecast repository in an AI-enabled IDE, the context is loaded **automatically** — no manual setup required:

| IDE / Tool | File loaded | How |
|------------|-------------|-----|
| **VS Code + GitHub Copilot** | `.github/copilot-instructions.md` | Injected into every Copilot prompt |
| **Claude Code** | `AGENTS.md` | Read automatically at project root |
| **OpenAI Codex / Aider** | `AGENTS.md` | Read automatically at project root |
| **Cursor** | `AGENTS.md` | Read automatically at project root |

These files contain the same core content: project structure, all forecasters, code style, and testing conventions.


## What's included in the context

The AI context covers:

- **All 8 forecaster types** — constructors, `fit()`, `predict()`, `predict_interval()`, `predict_quantiles()`, `predict_dist()`, and their parameter differences.
- **Model selection** — `backtesting_forecaster`, `grid_search_forecaster`, `bayesian_search_forecaster`, `TimeSeriesFold`, `OneStepAheadFold`, and their multi-series variants.
- **Preprocessing** — `RollingFeatures`, `TimeSeriesDifferentiator`, `DateTimeFeatureTransformer`, `QuantileBinner`.
- **Statistical models** — `Arima`, `Sarimax`, `Ets`, `Arar` wrapped by `ForecasterStats`.
- **Deep learning** — `ForecasterRnn` with `create_and_compile_model`, LSTM/GRU architectures.
- **12 specialized workflow skills** — step-by-step guides for common tasks, loaded on-demand by advanced AI agents.


## Workflow skills

Skforecast includes 12 modular **skills** — self-contained guides that AI agents can load on demand when a user asks about a specific topic. Each skill covers a complete workflow with decision trees, code examples, and common pitfalls.

| Skill | What it covers |
|-------|----------------|
| `forecasting-single-series` | End-to-end forecasting with `ForecasterRecursive`: data prep, fit, predict, backtest, intervals |
| `forecasting-multiple-series` | Global model with `ForecasterRecursiveMultiSeries`: encoding, dict input, multi-level predictions |
| `statistical-models` | `ForecasterStats` with `Arima`, `Sarimax`, `Ets`, `Arar`: auto-ARIMA, seasonal config |
| `hyperparameter-optimization` | Grid, random, and Bayesian search with `TimeSeriesFold` and `OneStepAheadFold` |
| `prediction-intervals` | Bootstrapping, conformal prediction, quantile regression, interval calibration |
| `feature-engineering` | `RollingFeatures`, `DateTimeFeatureTransformer`, custom features, exogenous variables |
| `feature-selection` | `RFECV`, `SelectFromModel`: selecting lags, window features, and exog |
| `drift-detection` | `RangeDriftDetector` and `PopulationDriftDetector` for production monitoring |
| `deep-learning-forecasting` | `ForecasterRnn` with Keras: `create_and_compile_model`, LSTM/GRU architectures |
| `choosing-a-forecaster` | Decision guide: "I have X situation → use Y forecaster" |
| `troubleshooting-common-errors` | Frequent mistakes AI assistants make with skforecast and their corrections |
| `complete-api-reference` | Full method signatures and availability matrix for all forecasters |

These skills are bundled into `llms-full.txt`. AI agents that support the [Agent Skills](https://agentskills.io) spec (such as GitHub Copilot in VS Code) can also load them individually from the `skills/` directory.


## Context files overview

| File | Audience | Description |
|------|----------|-------------|
| [`llms-full.txt`](../llms-full.txt) | Any LLM user | Complete context: API + 12 workflow skills |
| [`llms.txt`](../llms.txt) | LLMs with web search | Public index with links to all documentation sections |
| `.github/copilot-instructions.md` | Contributors (VS Code) | Auto-injected into GitHub Copilot |
| `AGENTS.md` | Contributors (Claude Code, Codex, Aider) | Standard agent context file |

The context files are **auto-generated** from a single source of truth (`tools/ai/llms-base.txt`) to ensure they stay in sync with the library. They are regenerated on every release.


## Tips for better results

1. **Always provide the context URL** — Without it, LLMs may hallucinate methods that don't exist or use outdated API names (e.g., `ForecasterAutoreg` instead of `ForecasterRecursive`).
2. **Be specific about your forecaster** — Mention which forecaster you're using. Parameter names and defaults differ across forecasters.
3. **Mention the version** — Say "skforecast 0.21.0" so the LLM doesn't mix advice from older versions.
4. **Validate the output** — AI-generated code is a starting point. Always run backtesting to verify model performance.
