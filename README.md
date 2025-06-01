# AI-Driven Crypto Trading System

## Overview

This project is an AI-driven cryptocurrency trading system designed for research, backtesting, and simulated trading. It features a modular architecture that allows for the integration of various components, including data ingestion, feature engineering, AI-based signal generation, risk management, and strategy backtesting. The system aims to leverage Large Language Models (LLMs) like Llama3 via Ollama for generating trading insights, combined with traditional technical analysis.

The core philosophy is to create a flexible framework where different strategies, risk management rules, and AI models can be tested and evaluated systematically.

## Features Implemented

The system currently includes the following key components:

*   **Data Ingestion:** OHLCV (CSV) & News (Cryptopanic API, mockable).
*   **Feature Engineering:** Standard TAs (`pandas-ta`), placeholders for ICT & News Sentiment.
*   **AI Signal Generation:** Ollama Llama3 integration, basic prompting, mockable, historical signal generation.
*   **Trading Signal Storage:** SQLite DB for all signals.
*   **Strategy Framework:** Base class, Random, SMA Cross, Historical AI strategies.
*   **Backtesting Engine:** Iterative, fees & slippage, basic metrics (P&L, Drawdown, Trades).
*   **Risk Management:** Position sizing (equity/risk based), SL/TP calculation & integration.
*   **Simulated Execution Engine:** Market order simulation, balance tracking.
*   **Orchestration:** `WORKFLOWS.md`, `orchestrator_example.py`.

## Directory Structure

-   `data/`: Raw and processed data (CSVs, SQLite DB).
-   `src/`: All Python source code (modularized by function: `data_ingestion`, `feature_engineering`, etc.).
-   `config/`: `settings.yaml` for configurations.
-   `logs/`: Application log files.
-   `notebooks/`: (Placeholder) For analysis.
-   `tests/`: (Placeholder) For tests.

## Setup Instructions

1.  **Python:** 3.9+ recommended.
2.  **Virtual Environment:** `python -m venv venv && source venv/bin/activate`
3.  **Dependencies:** `pip install -r requirements.txt`
4.  **Ollama (Optional):** Install from [ollama.com](https://ollama.com/), pull model (e.g., `ollama pull llama3`). Configure in `settings.yaml` or use mock mode.
5.  **Cryptopanic API Key (Optional):** Add to `settings.yaml` for live news, else mock is used.

## Running the System

Refer to `WORKFLOWS.md` for detailed operational flows.

*   **Main Workflow (Data, Features, Historical Signals, Backtest):**
    ```bash
    python orchestrator_example.py
    ```
*   **Individual Modules:** Most scripts in `src/` subdirectories can be run directly for specific tasks if needed.

## Configuration

All major configurations (DB paths, API keys, Ollama settings, risk/backtest defaults) are in `config/settings.yaml`.

## Current Status & Limitations

*   ICT features and News Sentiment are placeholders.
*   AI prompting is basic.
*   Sandbox SQLite visibility issue noted during multi-script workflow testing (data written by one script not always seen by next immediately).
*   Uses dummy/mocked data for some components in default tests.
*   Short selling not implemented.

## Next Steps & Future Enhancements (Summary)

-   **Features:** Implement advanced ICT & News Sentiment.
-   **AI:** Enhance prompting, explore fine-tuning & other models.
-   **Backtesting:** Add advanced metrics, plotting, optimization.
-   **Risk Management:** Portfolio-level controls, dynamic adjustments.
-   **Execution:** Limit orders, paper trading, live exchange integration.
-   **Infrastructure:** Consider more robust DB, UI/Dashboard.
-   **Quality:** Improve docstrings & test coverage.

---
This project provides a foundational framework. Contributions are welcome.
