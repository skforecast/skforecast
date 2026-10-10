# Skforecast AI

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![PyPI](https://img.shields.io/pypi/v/skforecast-ai)](https://pypi.org/project/skforecast-ai/)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/skforecast-ai?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/skforecast-ai)
[![License](https://img.shields.io/github/license/skforecast/skforecast-ai)](https://github.com/skforecast/skforecast-ai/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-ai.skforecast.org-f79939?logo=readthedocs)](https://ai.skforecast.org/)
[![GitHub](https://img.shields.io/badge/GitHub-skforecast--ai-181717?logo=github)](https://github.com/skforecast/skforecast-ai)
[![MCP server](https://img.shields.io/badge/MCP-server-blue?logo=modelcontextprotocol&logoColor=white)](https://ai.skforecast.org/stable/user-guides/mcp-server.html)


**[Skforecast AI](https://ai.skforecast.org/)** is an AI-assisted forecasting package from the skforecast team. It combines a deterministic forecasting engine powered by [skforecast](https://skforecast.org/) with an optional LLM reasoning layer.

Provide a time series and the assistant can profile the data, choose a forecasting strategy using established best practices, evaluate its performance, and return both the forecast and the runnable skforecast code that produced it.


## Why Skforecast AI?

- :dart: **Deterministic by design**: The rule-based forecasting engine produces consistent results for the same input.
- :mag: **Inspectable and reproducible**: The generated script is the code that ran, so you can inspect, version, and execute it independently with skforecast.
- :zap: **From data to forecast in one call**: Automates profiling, model and estimator selection, feature engineering, and backtesting.
- :computer: **Python and CLI workflows**: Use the assistant from Python or run the complete pipeline from the terminal.
- :speech_balloon: **Optional LLM reasoning**: Get plain-language explanations and configuration advice while keeping the core forecasting workflow available offline.
- :building_construction: **Built on skforecast**: Supports recursive and direct forecasters, multi-series forecasting, statistical models, and foundation models.


## Installation

Skforecast AI requires Python 3.10 or later.

```bash
pip install skforecast-ai
```

Install the optional LLM reasoning layer with:

```bash
pip install "skforecast-ai[llm]"
```


## Quick Start

```python
from skforecast.datasets import load_demo_dataset
from skforecast_ai import ForecastingAssistant

data = load_demo_dataset(verbose=False)
assistant = ForecastingAssistant()
result = assistant.forecast(data=data, target="y", steps=12)

print(result.predictions)
print(result.metrics)
print(result.code)
```

The result includes the predictions, backtesting metrics, data profile, selected modeling plan, and the standalone skforecast script used to produce the forecast.


## Use it from your coding agent

Since version 0.4.0, Skforecast AI includes an [MCP](https://modelcontextprotocol.io) server that brings skforecast to coding agents such as Claude Code, Claude Desktop or VS Code. Ask in plain language ("Forecast the next 12 months of `data/sales.csv` and tell me how accurate it is") and the agent profiles the file, plans a forecaster, backtests it, compares candidates and forecasts.

=== "Claude Code"

    Install it as a plugin. Run these commands inside Claude Code:

    ```
    /plugin marketplace add skforecast/skforecast-ai
    /plugin install skforecast-ai@skforecast-ai
    ```

=== "Other MCP clients"

    Register this command as an MCP server in your client (requires [uv](https://docs.astral.sh/uv/)):

    ```bash
    uvx skforecast-ai-mcp --allow-dir /absolute/path/to/project
    ```

    The server is listed in the official MCP registry as `io.github.skforecast/skforecast-ai`.

- **The rules decide, the agent explains**: the forecaster, estimator, lags, metric and cross-validation come from deterministic rules, not from the language model. The agent provides the model and explains the results.
- **Every result comes with its code**: each result includes the skforecast script that produced it, which you can run on its own.
- **Your data stays local**: the server only reads CSV files inside the allowed directory and never returns data rows to the agent, only summaries.

The server requires skforecast 0.26.0 or later. See the [MCP server user guide](https://ai.skforecast.org/stable/user-guides/mcp-server.html) for the configuration of each client, the available tools and the security model.


## Learn More

- :books: **[Documentation](https://ai.skforecast.org/)**: Tutorials, user guides, API reference, and release notes.
- :rocket: **[Quick start](https://ai.skforecast.org/stable/quick-start/quick-start.html)**: Create your first AI-assisted forecast.
- :book: **[Introduction to agentic forecasting](https://ai.skforecast.org/stable/user-guides/agentic-forecasting.html)**: Learn how the deterministic engine and optional reasoning layer work together.
- :robot: **[MCP server for coding agents](https://ai.skforecast.org/stable/user-guides/mcp-server.html)**: Use skforecast as tools from Claude Code, VS Code and other MCP clients.
- :octicons-mark-github-16: **[GitHub repository](https://github.com/skforecast/skforecast-ai)**: Browse the source, report issues, and contribute.


## Feedback and Issues

Skforecast AI is developed by the skforecast team. If you encounter a problem or have a suggestion, please [open an issue](https://github.com/skforecast/skforecast-ai/issues).