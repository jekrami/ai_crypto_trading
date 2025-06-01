# Project Workflows

This document outlines the main operational workflows for the trading system.

## Workflow A: Initial Data Setup & Full Historical Backtest Preparation

This workflow is typically run once to set up the system with historical data and prepare for backtesting strategies that rely on pre-generated signals (like AI-driven signals).

**Steps:**

1.  **Import Historical Market Data (OHLCV):**
    *   **Script:** `python src/data_ingestion/market_data_importer.py`
    *   **Action:** Populates `ohlcv_{SYMBOL}` tables (e.g., `ohlcv_BTCUSD`, `ohlcv_ETHUSD`) from CSV files found in `data/raw_ohlcv/`.
    *   **Assumes:** Raw CSV files are present in the `data/raw_ohlcv/` directory.

2.  **Import Historical News Data:**
    *   **Script:** `python src/data_ingestion/news_importer.py`
    *   **Action:** Fetches news from Cryptopanic API (or uses mock data if API key is placeholder) and stores it in the `news_articles` table.
    *   **Assumes:** `config/settings.yaml` has Cryptopanic API key or is set to use mock data.

3.  **Generate Features:**
    *   **Script:** `python src/feature_engineering/feature_pipeline.py`
    *   **Action:** Reads data from `ohlcv_{SYMBOL}` tables, calculates technical indicators and other features, and stores them in `features_{SYMBOL}` tables.
    *   **Assumes:** `ohlcv_{SYMBOL}` tables exist and contain data.

4.  **Generate Historical Trading Signals (e.g., AI-based):**
    *   **Script:** `python src/signal_generation/historical_signal_generator.py`
    *   **Action:** Iterates through historical data in `features_{SYMBOL}` tables, generates prompts, calls an AI model (e.g., Ollama, possibly mocked) for trading signals (BUY/SELL/HOLD), and stores these signals in the `trading_signals` table with `signal_type` like 'AI_Llama3_Hist'.
    *   **Assumes:** `features_{SYMBOL}` tables exist. `config/settings.yaml` is configured for Ollama (or mock).

5.  **Run Backtest with Historical Signals:**
    *   **Script:** `python src/strategy_backtesting/main_backtest_runner.py`
    *   **Action:**
        *   Configure `main_backtest_runner.py` to use `HistoricalAiSignalStrategy`.
        *   This strategy loads signals from the `trading_signals` table for a specific symbol and `signal_type_filter`.
        *   The backtester runs through the historical feature data, applying trades based on the pre-generated AI signals.
        *   Performance metrics are calculated and printed.
    *   **Assumes:** `features_{SYMBOL}` and `trading_signals` tables are populated.

## Workflow B: Simulated Live Trading

This workflow simulates live trading on a bar-by-bar basis using the latest available data and a chosen strategy. It uses a simulated exchange to track balances and execute orders.

**Steps:**

1.  **Ensure Data is Up-to-Date:**
    *   This workflow assumes that `ohlcv_{SYMBOL}` and `features_{SYMBOL}` tables are reasonably up-to-date. If not, run relevant parts of Workflow C first.

2.  **Run Live Simulator:**
    *   **Script:** `python src/execution/main_live_simulator.py`
    *   **Action:**
        *   Loads the most recent feature data for a specified symbol (e.g., BTCUSD).
        *   Initializes a `SimulatedExchange` with starting balances (from config).
        *   Iterates through the loaded feature data, bar by bar.
        *   For each bar:
            *   Checks for SL/TP on any open position.
            *   Calls the specified strategy (e.g., `RandomStrategy`, `SmaCrossStrategy`, or potentially a live AI signal generator if integrated) to get a trading decision.
            *   Applies risk management (position sizing, SL/TP calculation for new trades).
            *   Executes market orders on the `SimulatedExchange`.
        *   Prints final balances and trade history from the simulation.
    *   **Assumes:** `features_{SYMBOL}` table exists. `config/settings.yaml` is configured for simulated exchange and risk management.

## Workflow C: Incremental Data Update & Signal Generation (for Live Operation)

This workflow is designed to be run periodically (e.g., via a cron job or scheduler) to keep the system's data fresh and generate new signals for potential live trading.

**Steps:**

1.  **Incrementally Update Market Data (OHLCV):**
    *   **Script:** (Future enhancement) `src/data_ingestion/market_data_updater.py` (or modify `market_data_importer.py` to handle incremental updates).
    *   **Action:** Fetches only the *newest* OHLCV data since the last update from an API (e.g., CCXT for exchanges) and appends it to `ohlcv_{SYMBOL}` tables.
    *   **Currently:** `market_data_importer.py` re-imports entire CSVs. For true incremental, API polling is needed.

2.  **Incrementally Update News Data:**
    *   **Script:** `python src/data_ingestion/news_importer.py` (if run, it fetches recent news).
    *   **Action:** Fetches the latest news articles and adds new ones to `news_articles`.
    *   **Note:** The current `news_importer.py` might re-fetch some existing news if not designed for strict incremental updates based on last fetched ID/timestamp.

3.  **Incrementally Update Features:**
    *   **Script:** (Future enhancement) `src/feature_engineering/feature_pipeline.py` (modified for incremental processing).
    *   **Action:** Calculates features only for the newly added OHLCV data in `ohlcv_{SYMBOL}` and updates/appends to `features_{SYMBOL}` tables.
    *   **Currently:** `feature_pipeline.py` reprocesses all data for a symbol.

4.  **Generate New Trading Signal (for current market state):**
    *   **Script:** `python src/signal_generation/main_signal_generator.py`
    *   **Action:**
        *   Fetches the *latest* set of features for a symbol.
        *   Formats a prompt for the AI model.
        *   Calls Ollama (or other AI model interface) to get a trading signal.
        *   Stores this new signal in the `trading_signals` table with a `signal_type` like 'AI_Llama3_Live'.
    *   **Note:** This signal is for the *next* trading period and could be used by an execution bot.

5.  **(Optional) Monitor and Execute Trades:**
    *   **Component:** (Future) An execution bot.
    *   **Action:** Reads the latest signal from `trading_signals`. If a strong BUY/SELL signal, places an order with a real brokerage/exchange API. Manages open positions.

This workflow structure provides a roadmap for both backtesting historical performance and moving towards simulated/live trading operations.
