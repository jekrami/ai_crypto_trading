import os
import sys
import pandas as pd
from typing import Optional, Dict, List, Any # Added typing imports

# Setup paths
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /app/src
PROJECT_ROOT = os.path.dirname(SRC_DIR) # /app
sys.path.append(SRC_DIR)

from utils.db_manager import DBManager
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

# Import available strategies
from strategy_backtesting.strategy import RandomStrategy, SmaCrossStrategy, HistoricalAiSignalStrategy
from strategy_backtesting.backtester import Backtester

logger = setup_logger(__name__)

def get_db_and_config():
    """Initializes and returns ConfigManager and DBManager."""
    config_file_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found at {config_file_path}.")
        raise FileNotFoundError(f"Configuration file not found at {config_file_path}")

    cm = ConfigManager(config_file_path=config_file_path)

    db_path_from_config = cm.get('database.path', "data/trading_system.db")
    db_full_path = db_path_from_config
    if not os.path.isabs(db_full_path):
        db_full_path = os.path.join(PROJECT_ROOT, db_full_path)

    db_dir = os.path.dirname(db_full_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    db_m = DBManager(db_path=db_full_path)
    return cm, db_m

def fetch_feature_data_for_backtest(db_manager: DBManager, symbol: str) -> Optional[pd.DataFrame]:
    """Fetches all data from features_{symbol} table for backtesting."""
    table_name = f"features_{symbol.upper()}"
    query = f"SELECT * FROM {table_name} ORDER BY timestamp ASC;" # Get all data, oldest first

    logger.info(f"Fetching all feature data for {symbol} from {table_name} for backtesting...")
    try:
        pragma_query = f"PRAGMA table_info({table_name});"
        columns_info = db_manager.fetch_data(pragma_query)
        if not columns_info:
            logger.error(f"Could not fetch column info for table {table_name}.")
            return None
        cols = [info[1] for info in columns_info]

        data = db_manager.fetch_data(query)
        if not data:
            logger.warning(f"No feature data found for {symbol} in table {table_name}.")
            return None

        df = pd.DataFrame(data, columns=cols)

        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') # Assuming Unix timestamps stored as int
                df.set_index('timestamp', inplace=True)
            except Exception as e:
                logger.error(f"Could not process 'timestamp' column for {symbol} features: {e}. Backtesting may fail.")
                return None # Timestamp index is critical for backtester
        else:
            logger.error(f"Timestamp column missing in fetched data for {symbol}. Cannot proceed.")
            return None

        # Convert numeric columns, coercing errors to NaN. Backtester should handle NaNs.
        # This is important because data from SQLite might be TEXT for TIs if they were all NAs.
        numeric_cols_to_try = ['open', 'high', 'low', 'close', 'volume'] + \
                              [col for col in df.columns if 'sma_' in col or 'ema_' in col or 'rsi_' in col or \
                               'macd_' in col or 'bb' in col or 'atr_' in col]
        for col in numeric_cols_to_try:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Successfully fetched and prepared {len(df)} feature rows for {symbol} for backtest.")
        return df
    except Exception as e:
        logger.error(f"Error fetching features for backtesting {symbol}: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    logger.info("Starting Backtest Runner script...")

    try:
        cm, db_m = get_db_and_config()
    except FileNotFoundError:
        logger.critical("Exiting: Config file not found.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Exiting: Error initializing services: {e}", exc_info=True)
        sys.exit(1)

    symbol_to_backtest = "BTCUSD" # Or fetch from config/args

    features_df: Optional[pd.DataFrame] = None
    with db_m: # Ensure DB connection is managed
        # Ensure trading_signals table exists (though not directly used by backtester, good practice)
        db_m.create_trading_signals_table()
        features_df = fetch_feature_data_for_backtest(db_m, symbol_to_backtest)

    if features_df is None or features_df.empty:
        logger.error(f"No feature data available for {symbol_to_backtest}. Cannot run backtest.")
        sys.exit(1)

    if len(features_df) < 5: # Arbitrary small number, SMA might need more
        logger.warning(f"Feature data for {symbol_to_backtest} is very short ({len(features_df)} rows). Backtest results may not be meaningful.")

    # --- Strategy Selection ---
    # strategy_to_run = RandomStrategy(strategy_params={'random_seed': 123})
    # logger.info(f"Using strategy: {strategy_to_run.strategy_name}")

    # For SmaCrossStrategy, ensure the columns exist from feature engineering.
    # The dummy data (3 rows) + feature engineering (SMA 20, 50) will result in all NaN SMAs.
    # SmaCrossStrategy should handle NaNs by HOLDing.
    # To test SMA Cross with dummy data, we'd need longer dummy data or very short SMA windows
    # that were calculated in feature_pipeline.py.
    # Let's assume feature_pipeline.py added 'sma_20' and 'sma_50'.
    # The dummy data is only 3 rows, so SMA20 and SMA50 will be all NaN.
    # SmaCrossStrategy(short_window=20, long_window=50) will produce only HOLDs.
    # For a more active test with SmaCross and dummy data, one would need to:
    # 1. Create longer dummy CSV data (e.g., 50-100 rows).
    # 2. Ensure feature_pipeline calculates SMAs for windows that can actually produce values (e.g., sma_5, sma_10).
    # 3. Then use SmaCrossStrategy(short_window=5, long_window=10).

    # --- CHOOSE STRATEGY TO RUN ---
    # strategy_to_run = RandomStrategy(strategy_params={'random_seed': 42})

    # Option 2: Run with HistoricalAiSignalStrategy
    # Ensure historical_signal_generator.py has been run before this to populate signals.
    logger.info("Attempting to use HistoricalAiSignalStrategy.")
    try:
        # Make sure the DBManager (db_m) is passed, as HistoricalAiSignalStrategy needs it.
        # The db_m is already in context from the 'with db_m:' block earlier if that was used,
        # but HistoricalAiSignalStrategy needs its own explicit db_manager.
        # Let's re-get a db_manager instance or ensure the one from get_db_and_config() is passed.
        # For simplicity here, we assume db_m from get_db_and_config() is available and connected if needed.
        # A better pattern might be to pass db_m explicitly to functions needing it.
        # The HistoricalAiSignalStrategy constructor will use its db_manager instance.
        strategy_to_run = HistoricalAiSignalStrategy(db_manager=db_m, symbol=symbol_to_backtest) # Ensure db_m is active or strategy handles connections
        if strategy_to_run.ai_signals_df.empty:
            logger.warning(f"HistoricalAiSignalStrategy loaded no signals for {symbol_to_backtest}. Backtest might be trivial. Ensure historical_signal_generator.py was run.")
    except Exception as e:
        logger.error(f"Failed to initialize HistoricalAiSignalStrategy: {e}. Defaulting to RandomStrategy.")
        strategy_to_run = RandomStrategy(strategy_params={'random_seed': 42})

    logger.info(f"Selected strategy for backtest: {strategy_to_run.strategy_name}")


    # --- Backtester Configuration ---
    # Get defaults from config or use hardcoded values
    initial_capital = cm.get('backtesting_defaults.initial_capital', 10000.0)
    fee_percent = cm.get('backtesting_defaults.fee_percent', 0.001)
    slippage_percent = cm.get('backtesting_defaults.slippage_percent', 0.0005)

    logger.info(f"Backtester Config: Initial Capital=${initial_capital}, Fee={fee_percent*100}%, Slippage={slippage_percent*100}%")

    backtester = Backtester(
        strategy=strategy_to_run,
        historical_data_df=features_df,
        initial_capital=initial_capital,
        transaction_fee_percent=fee_percent,
        slippage_percent=slippage_percent,
        price_col='close', # Assuming 'close' price for trade execution
        config_manager=cm # Pass ConfigManager instance
    )

    logger.info(f"Running backtest for {symbol_to_backtest} with {strategy_to_run.strategy_name}...")
    portfolio_history, trades_history = backtester.run_backtest()

    if not portfolio_history:
        logger.error("Backtest did not produce any portfolio history. Exiting.")
        sys.exit(1)

    logger.info("Backtest run completed. Calculating performance metrics...")
    metrics = backtester.calculate_performance_metrics()

    print(f"\n--- Backtest Results for {symbol_to_backtest} using {strategy_to_run.strategy_name} ---")
    print("Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")

    if trades_history:
        print("\nTrades History (first 5 trades):")
        trades_df = pd.DataFrame(trades_history)
        print(trades_df.head().to_string())
    else:
        print("\nNo trades were executed during this backtest.")

    if portfolio_history:
        print("\nPortfolio History (last 5 periods):")
        portfolio_df = pd.DataFrame(portfolio_history)
        print(portfolio_df.tail().to_string())

    logger.info("Backtest Runner script finished.")

def main(strategy_choice: str = "AI"): # strategy_choice can be "AI" or "Random" for this example
    """Main entry point for backtest runner, callable from orchestrator."""
    logger.info(f"--- Starting Backtest Runner via main() with strategy_choice: {strategy_choice} ---")

    try:
        cm, db_m = get_db_and_config()
    except FileNotFoundError:
        logger.critical("Exiting: Config file not found.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Exiting: Error initializing services: {e}", exc_info=True)
        sys.exit(1)

    symbol_to_backtest = "BTCUSD"

    features_df: Optional[pd.DataFrame] = None
    with db_m:
        db_m.create_trading_signals_table()
        features_df = fetch_feature_data_for_backtest(db_m, symbol_to_backtest)

    if features_df is None or features_df.empty:
        logger.error(f"No feature data available for {symbol_to_backtest}. Cannot run backtest.")
        sys.exit(1)

    if len(features_df) < 5:
        logger.warning(f"Feature data for {symbol_to_backtest} is very short ({len(features_df)} rows). Backtest results may not be meaningful.")

    if strategy_choice.upper() == "AI":
        logger.info(f"Attempting to use HistoricalAiSignalStrategy for {symbol_to_backtest}.")
        try:
            strategy_to_run = HistoricalAiSignalStrategy(db_manager=db_m, symbol=symbol_to_backtest)
            if strategy_to_run.ai_signals_df.empty:
                logger.warning(f"HistoricalAiSignalStrategy loaded no signals for {symbol_to_backtest}. Backtest might be trivial. Ensure historical_signal_generator.py was run.")
        except Exception as e:
            logger.error(f"Failed to initialize HistoricalAiSignalStrategy: {e}. Defaulting to RandomStrategy.")
            strategy_to_run = RandomStrategy(strategy_params={'random_seed': 42})
    else: # Default to Random or other specified
        logger.info(f"Using RandomStrategy for {symbol_to_backtest}.")
        strategy_to_run = RandomStrategy(strategy_params={'random_seed': 42})

    logger.info(f"Selected strategy for backtest: {strategy_to_run.strategy_name}")

    initial_capital = cm.get('backtesting_defaults.initial_capital', 10000.0)
    fee_percent = cm.get('backtesting_defaults.fee_percent', 0.001)
    slippage_percent = cm.get('backtesting_defaults.slippage_percent', 0.0005)

    logger.info(f"Backtester Config: Initial Capital=${initial_capital}, Fee={fee_percent*100}%, Slippage={slippage_percent*100}%")

    backtester = Backtester(
        strategy=strategy_to_run,
        historical_data_df=features_df,
        initial_capital=initial_capital,
        transaction_fee_percent=fee_percent,
        slippage_percent=slippage_percent,
        price_col='close',
        config_manager=cm
    )

    logger.info(f"Running backtest for {symbol_to_backtest} with {strategy_to_run.strategy_name}...")
    portfolio_history, trades_history = backtester.run_backtest()

    if not portfolio_history:
        logger.error("Backtest did not produce any portfolio history. Exiting.")
        sys.exit(1)

    logger.info("Backtest run completed. Calculating performance metrics...")
    metrics = backtester.calculate_performance_metrics()

    print(f"\n--- Backtest Results for {symbol_to_backtest} using {strategy_to_run.strategy_name} ---")
    print("Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")

    if trades_history:
        print("\nTrades History (first 5 trades):")
        trades_df = pd.DataFrame(trades_history)
        print(trades_df.head().to_string())
    else:
        print("\nNo trades were executed during this backtest.")

    if portfolio_history:
        print("\nPortfolio History (last 5 periods):")
        portfolio_df = pd.DataFrame(portfolio_history)
        print(portfolio_df.tail().to_string())

    logger.info("Backtest Runner main() call complete.")

# The main() function above serves as a good entry point for direct script execution or simple orchestration.
# For more programmatic control as requested by the subtask, here's run_backtest_for_workflow:

def run_backtest_for_workflow(symbol_to_test: str = 'BTCUSD', strategy_key: str = 'AI_HISTORICAL', config_manager_instance: Optional[ConfigManager] = None):
    """
    Encapsulates the setup and execution of a backtest run for a specific symbol and strategy.

    :param symbol_to_test: The symbol to backtest (e.g., 'BTCUSD').
    :param strategy_key: Key to select the strategy ('AI_HISTORICAL', 'RANDOM', etc.).
    :param config_manager_instance: Optional pre-initialized ConfigManager. If None, it will be created.
    """
    logger.info(f"--- Running Backtest via Workflow Function for {symbol_to_test} with Strategy Key: {strategy_key} ---")

    if config_manager_instance:
        cm = config_manager_instance
        # Assume db_manager needs to be created based on this cm, or also passed in if already available
        db_path_from_config = cm.get('database.path', "data/trading_system.db")
        db_full_path = db_path_from_config
        if not os.path.isabs(db_full_path):
            PROJECT_ROOT_FOR_DB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_full_path = os.path.join(PROJECT_ROOT_FOR_DB, db_full_path)
        db_m = DBManager(db_path=db_full_path)
    else:
        try:
            cm, db_m = get_db_and_config() # Uses existing helper to get fresh instances
        except FileNotFoundError:
            logger.critical("Exiting run_backtest_for_workflow: Config file not found.")
            return
        except Exception as e:
            logger.critical(f"Exiting run_backtest_for_workflow: Error initializing services: {e}", exc_info=True)
            return

    features_df: Optional[pd.DataFrame] = None
    with db_m:
        # It's good practice to ensure tables exist, though create_trading_signals_table might be called elsewhere too
        db_m.create_trading_signals_table()
        features_df = fetch_feature_data_for_backtest(db_m, symbol_to_test)

    if features_df is None or features_df.empty:
        logger.error(f"No feature data available for {symbol_to_test}. Cannot run backtest in workflow.")
        return

    if len(features_df) < 5 and strategy_key != 'RANDOM': # Random might not care about data length for signals
        logger.warning(f"Feature data for {symbol_to_test} is very short ({len(features_df)} rows). Backtest results may not be meaningful for complex strategies.")

    strategy_to_run: Optional[BaseStrategy] = None
    if strategy_key.upper() == "AI_HISTORICAL":
        logger.info(f"Attempting to use HistoricalAiSignalStrategy for {symbol_to_test}.")
        try:
            # Pass the db_m that has an active connection context or let strategy create its own.
            # The HistoricalAiSignalStrategy as modified now creates its own fresh DBManager for loading.
            # So, passing db_m here is more for consistency or if it were to use it directly.
            strategy_to_run = HistoricalAiSignalStrategy(db_manager=db_m, symbol=symbol_to_test) # Corrected variable name
            if strategy_to_run.ai_signals_df.empty:
                logger.warning(f"HistoricalAiSignalStrategy loaded no signals for {symbol_to_test}. Backtest will likely be trivial.") # Corrected variable name
        except Exception as e:
            logger.error(f"Failed to initialize HistoricalAiSignalStrategy: {e}. Defaulting to RandomStrategy for safety.")
            strategy_to_run = RandomStrategy(strategy_params={'random_seed': 42})

    elif strategy_key.upper() == "RANDOM":
        logger.info(f"Using RandomStrategy for {symbol_to_backtest}.")
        strategy_to_run = RandomStrategy(strategy_params={'random_seed': 42})

    # Add other strategy key mappings here e.g. elif strategy_key == "SMA_CROSS":
    #    strategy_to_run = SmaCrossStrategy(short_window=20, long_window=50)


    if not strategy_to_run: # If no valid strategy was set
        logger.error(f"No valid strategy selected for key '{strategy_key}'. Defaulting to RandomStrategy.")
        strategy_to_run = RandomStrategy(strategy_params={'random_seed': 42})

    logger.info(f"Selected strategy for backtest: {strategy_to_run.strategy_name}")

    initial_capital = cm.get('backtesting_defaults.initial_capital', 10000.0)
    fee_percent = cm.get('backtesting_defaults.fee_percent', 0.001)
    slippage_percent = cm.get('backtesting_defaults.slippage_percent', 0.0005)

    logger.info(f"Backtester Config: Initial Capital=${initial_capital}, Fee={fee_percent*100}%, Slippage={slippage_percent*100}%")

    backtester = Backtester(
        strategy=strategy_to_run,
        historical_data_df=features_df,
        initial_capital=initial_capital,
        transaction_fee_percent=fee_percent,
        slippage_percent=slippage_percent,
        price_col='close',
        config_manager=cm
    )

    logger.info(f"Running backtest for {symbol_to_test} with {strategy_to_run.strategy_name}...")
    portfolio_history, trades_history = backtester.run_backtest()

    if not portfolio_history:
        logger.error("Backtest did not produce any portfolio history within workflow function.")
        return # Exit if backtest fundamentally failed

    logger.info("Backtest run completed. Calculating performance metrics...")
    metrics = backtester.calculate_performance_metrics()

    print(f"\n--- Workflow Backtest Results for {symbol_to_test} using {strategy_to_run.strategy_name} ---")
    print("Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")

    if trades_history:
        print("\nTrades History (first 5 trades):")
        trades_df = pd.DataFrame(trades_history)
        print(trades_df.head().to_string())
    else:
        print("\nNo trades were executed during this backtest.")

    if portfolio_history:
        print("\nPortfolio History (last 5 periods):")
        portfolio_df = pd.DataFrame(portfolio_history)
        print(portfolio_df.tail().to_string())

    logger.info(f"--- Backtest via Workflow Function for {symbol_to_test} Finished ---")
