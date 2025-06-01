import os
import sys
import pandas as pd
from typing import Dict, Any, Optional, List

# Setup paths
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /app/src
PROJECT_ROOT = os.path.dirname(SRC_DIR) # /app
sys.path.append(SRC_DIR)

from utils.db_manager import DBManager
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

from signal_generation.ollama_client import OllamaClient
from signal_generation.prompt_engineering import format_market_context_prompt, get_trading_signal_prompt
from signal_generation.signal_processor import parse_llama_response

logger = setup_logger(__name__)

def get_services():
    """Initializes and returns common services like ConfigManager and DBManager."""
    config_file_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found at {config_file_path}. Cannot proceed.")
        raise FileNotFoundError(f"Configuration file not found at {config_file_path}")

    cm = ConfigManager(config_file_path=config_file_path)

    db_path_from_config = cm.get('database.path', "data/trading_system.db")
    db_full_path = db_path_from_config
    if not os.path.isabs(db_full_path):
        db_full_path = os.path.join(PROJECT_ROOT, db_full_path)

    # Ensure DB directory exists before DBManager tries to connect
    db_dir = os.path.dirname(db_full_path)
    if not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir)
            logger.info(f"Created database directory: {db_dir}")
        except OSError as e:
            logger.error(f"Failed to create database directory {db_dir}: {e}")
            # Depending on desired behavior, may re-raise or handle

    db_m = DBManager(db_path=db_full_path)
    ollama_c = OllamaClient(config_manager=cm) # Pass CM to OllamaClient

    return cm, db_m, ollama_c

def fetch_features_for_symbol(db_manager: DBManager, symbol: str, num_rows: int = 50) -> Optional[pd.DataFrame]:
    """Fetches the latest N rows from the features_{symbol} table."""
    table_name = f"features_{symbol.upper()}"
    # Query for latest N rows, assuming timestamp is sortable (unix epoch or datetime string)
    # If timestamp is INTEGER PRIMARY KEY from unix epoch, this is fine.
    query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT {num_rows};"

    logger.info(f"Fetching latest {num_rows} feature rows for {symbol} from {table_name}...")
    try:
        # DBManager.fetch_data returns list of tuples. Need column names for DataFrame.
        # A robust way is to get column names from PRAGMA table_info(table_name)
        # For now, assume we can get them or they are somewhat fixed after features table creation.

        # Quick way to get column names if DBManager can provide them or if we query pragma
        pragma_query = f"PRAGMA table_info({table_name});"
        columns_info = db_manager.fetch_data(pragma_query)
        if not columns_info:
            logger.error(f"Could not fetch column info for table {table_name}.")
            return None
        cols = [info[1] for info in columns_info] # Column name is the second item in tuple from PRAGMA

        data = db_manager.fetch_data(query)
        if not data:
            logger.warning(f"No feature data found for {symbol} in table {table_name}.")
            return None

        df = pd.DataFrame(data, columns=cols)

        # Data is ORDER BY timestamp DESC, so reverse it to have most recent last for context
        df = df.iloc[::-1].reset_index(drop=True)

        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') # Assuming Unix timestamps
                df.set_index('timestamp', inplace=True)
            except Exception as e:
                logger.warning(f"Could not process 'timestamp' column for {symbol} features: {e}. Proceeding without DatetimeIndex if possible.")

        logger.info(f"Successfully fetched {len(df)} feature rows for {symbol}.")
        return df
    except Exception as e:
        logger.error(f"Error fetching features for {symbol}: {e}", exc_info=True)
        return None

def generate_signal_for_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Generates a trading signal for a given symbol and stores it in the database.
    Returns the parsed signal dictionary if successful, otherwise None.
    """
    logger.info(f"Starting signal generation for symbol: {symbol}")

    try:
        cm, db_m, ollama_c = get_services()
    except FileNotFoundError:
        logger.error("Cannot generate signal due to missing configuration file.")
        return None
    except Exception as e:
        logger.error(f"Error initializing services: {e}", exc_info=True)
        return None

    latest_feature_timestamp: Optional[int] = None
    features_df: Optional[pd.DataFrame] = None

    with db_m: # Manages DB connection for all DB ops in this function scope
        features_df = fetch_features_for_symbol(db_m, symbol, num_rows=10)

    if features_df is None or features_df.empty:
        logger.warning(f"Cannot generate signal for {symbol}: No feature data available.")
        return None

    # Get the timestamp of the most recent feature data point used for the signal
    if isinstance(features_df.index, pd.DatetimeIndex) and not features_df.index.empty:
        latest_feature_timestamp = int(features_df.index[-1].timestamp())
    else:
        logger.warning(f"Could not determine latest feature timestamp for {symbol}. Signal will not be accurately timestamped for relation.")
        # Using current time as a fallback for the main 'timestamp' field of the signal
        latest_feature_timestamp = int(pd.Timestamp.now(tz='UTC').timestamp())


    mock_news = [
        "Market sentiment appears cautiously optimistic following recent inflation data.",
        f"Specific news for {symbol}: A new partnership announced which may impact price positively.",
        "Global economic outlook remains uncertain, advise caution."
    ]
    logger.info(f"Using mock news for {symbol} context.")

    market_context_str = format_market_context_prompt(
        symbol=symbol, ohlcv_features_df=features_df,
        recent_news_list=mock_news, num_recent_features=5 # Use last 5 data points from the fetched (already sliced) df
    )
    logger.debug(f"Formatted Market Context for {symbol}:\n{market_context_str}")

    full_prompt = get_trading_signal_prompt(market_context_str, symbol)
    logger.debug(f"Full Prompt for {symbol}:\n{full_prompt}")

    raw_llm_response = ollama_c.generate_text(prompt=full_prompt)

    if not raw_llm_response:
        logger.error(f"No response from Ollama for {symbol}. Cannot generate signal.")
        return None # Return the original parsed_signal_info

    logger.info(f"Raw response from Ollama for {symbol}: {raw_llm_response}")

    parsed_signal_info = parse_llama_response(raw_llm_response)

    if parsed_signal_info and parsed_signal_info.get('signal'):
        logger.info(f"Successfully parsed signal for {symbol}: {parsed_signal_info['signal']}. Reasoning: {parsed_signal_info.get('reasoning', 'N/A')}")

        signal_to_store = {
            "timestamp": latest_feature_timestamp, # This is the OHLCV bar's timestamp
            "symbol": symbol,
            "signal_type": f"AI_{ollama_c.default_model}",
            "signal_action": parsed_signal_info['signal'],
            "confidence": parsed_signal_info.get('confidence'),
            "reasoning": parsed_signal_info.get('reasoning'),
            "target_price": None,
            "stop_loss_price": None,
            "model_prompt": full_prompt,
            "raw_model_response": raw_llm_response,
            # Link to the specific feature record that triggered/preceded this signal
            "related_feature_timestamp": latest_feature_timestamp,
            "related_feature_symbol": symbol
        }

        with db_m:
            if db_m.insert_trading_signal(signal_to_store):
                logger.info(f"Successfully stored signal for {symbol} in trading_signals table.")
            else:
                logger.error(f"Failed to store signal for {symbol} in trading_signals table.")

        return parsed_signal_info
    else:
        logger.warning(f"Could not parse a valid signal from Ollama response for {symbol}.")
        return None # Return None if parsing failed


if __name__ == "__main__":
    logger.info("Starting main signal generator script execution...")

    # Ensure all required Python packages are available
    try:
        import ollama
        import pandas
        import yaml
        import requests
    except ImportError as e:
        logger.critical(f"Missing one or more critical Python packages: {e}. "
                        "Please ensure PyYAML, requests, pandas, and ollama are installed.")
        sys.exit(1)

    # Check config for mock mode to inform user & ensure signals table exists
    try:
        temp_cm, temp_db_m, _ = get_services()
        # Ensure trading_signals table exists - crucial for this script.
        with temp_db_m:
            temp_db_m.create_trading_signals_table()

        is_mocking = temp_cm.get('ollama_settings.mock_ollama_for_testing', False)
        if is_mocking:
            logger.warning("Ollama is configured for MOCKING. Actual API calls will NOT be made. Using mock responses from config.")
        else:
            logger.info("Ollama is configured for LIVE calls. Ensure Ollama server is running and the model is available.")
            logger.info(f"Target Ollama host: {temp_cm.get('ollama_settings.host')}, Model: {temp_cm.get('ollama_settings.model')}")
    except FileNotFoundError:
        logger.error("Main script cannot check mock status: config file not found. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during initial config check for mock status: {e}. Proceeding cautiously.")


    # Example: Generate signal for BTCUSD
    # This assumes `features_BTCUSD` table exists and has data from previous steps.
    # If not, fetch_features_for_symbol will return None and pipeline will stop for that symbol.
    symbol_to_test = "BTCUSD"
    logger.info(f"Attempting to generate trading signal for: {symbol_to_test}")

    # generate_signal_for_symbol now also stores the signal
    parsed_signal_dict = generate_signal_for_symbol(symbol_to_test)

    if parsed_signal_dict:
        print(f"\n--- Generated and Stored Signal for {symbol_to_test} ---")
        # Signal ID is not directly returned by generate_signal_for_symbol, would need another DB query to get it.
        # print(f"Signal ID (from DB): [Query DB to get last inserted ID for {symbol_to_test} if needed]")
        print(f"Signal Action: {parsed_signal_dict.get('signal')}")
        print(f"Reasoning: {parsed_signal_dict.get('reasoning', 'N/A')}")
        print(f"Confidence: {parsed_signal_dict.get('confidence', 'N/A')}")
    else:
        print(f"\n--- No valid signal generated or stored for {symbol_to_test} ---")
        logger.warning(f"Signal generation process for {symbol_to_test} did not yield a valid parsed signal for storage.")

    logger.info("Main signal generator script execution finished.")
