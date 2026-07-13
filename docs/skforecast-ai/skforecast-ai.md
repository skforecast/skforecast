# Skforecast AI

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
[![PyPI](https://img.shields.io/pypi/v/skforecast-ai)](https://pypi.org/project/skforecast-ai/)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Downloads](https://static.pepy.tech/badge/skforecast-ai)](https://pepy.tech/project/skforecast-ai)
[![License](https://img.shields.io/github/license/skforecast/skforecast-ai)](https://github.com/skforecast/skforecast-ai/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-ai.skforecast.org-f79939?logo=readthedocs)](https://ai.skforecast.org/)
[![GitHub](https://img.shields.io/badge/GitHub-skforecast--ai-181717?logo=github)](https://github.com/skforecast/skforecast-ai)


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


## Learn More

- :books: **[Documentation](https://ai.skforecast.org/)**: Tutorials, user guides, API reference, and release notes.
- :rocket: **[Quick start](https://ai.skforecast.org/stable/quick-start/quick-start.html)**: Create your first AI-assisted forecast.
- :book: **[Introduction to agentic forecasting](https://ai.skforecast.org/stable/user-guides/agentic-forecasting.html)**: Learn how the deterministic engine and optional reasoning layer work together.
- :octicons-mark-github-16: **[GitHub repository](https://github.com/skforecast/skforecast-ai)**: Browse the source, report issues, and contribute.


## Feedback and Issues

Skforecast AI is developed by the skforecast team. If you encounter a problem or have a suggestion, please [open an issue](https://github.com/skforecast/skforecast-ai/issues).