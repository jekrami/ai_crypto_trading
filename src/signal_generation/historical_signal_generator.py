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

class HistoricalSignalGenerator:
    def __init__(self, config_manager: ConfigManager, db_manager: DBManager, ollama_client: OllamaClient):
        self.cm = config_manager
        self.db_m = db_manager
        self.ollama_c = ollama_client
        logger.info("HistoricalSignalGenerator initialized.")

    def fetch_features_for_symbol_historical(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetches historical feature data for a symbol, optionally filtered by date."""
        table_name = f"features_{symbol.upper()}"
        query = f"SELECT * FROM {table_name}"

        # Date filtering (basic, assumes timestamp is unix epoch stored as INTEGER)
        # For more robust date filtering, ensure conversion or use SQL date functions if available/formatted.
        conditions = []
        params = []
        if start_date:
            try:
                start_ts = int(pd.Timestamp(start_date, tz='UTC').timestamp())
                conditions.append("timestamp >= ?")
                params.append(start_ts)
            except ValueError:
                logger.error(f"Invalid start_date format: {start_date}. Should be YYYY-MM-DD.")
                return None
        if end_date:
            try:
                end_ts = int(pd.Timestamp(end_date, tz='UTC').timestamp())
                conditions.append("timestamp <= ?")
                params.append(end_ts)
            except ValueError:
                logger.error(f"Invalid end_date format: {end_date}. Should be YYYY-MM-DD.")
                return None

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp ASC;"

        logger.info(f"Fetching historical features for {symbol} from {table_name} with query: {query} {params if params else ''}")

        try:
            pragma_query = f"PRAGMA table_info({table_name});"
            columns_info = self.db_m.fetch_data(pragma_query)
            if not columns_info:
                logger.error(f"Could not fetch column info for table {table_name}.")
                return None
            cols = [info[1] for info in columns_info]

            data = self.db_m.fetch_data(query, tuple(params))
            if not data:
                logger.warning(f"No historical feature data found for {symbol} matching criteria.")
                return None

            df = pd.DataFrame(data, columns=cols)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
            else:
                logger.error(f"Timestamp column missing for {symbol}. Cannot proceed.")
                return None

            # Convert numeric columns
            numeric_cols_to_try = ['open', 'high', 'low', 'close', 'volume'] + \
                                [col for col in df.columns if 'sma_' in col or 'rsi_' in col or 'atr_' in col or 'bb' in col or 'macd' in col]
            for col in numeric_cols_to_try:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.info(f"Successfully fetched {len(df)} historical feature rows for {symbol}.")
            return df
        except Exception as e:
            logger.error(f"Error fetching historical features for {symbol}: {e}", exc_info=True)
            return None

    def generate_and_store_signals(self,
                                   symbol: str,
                                   features_df: pd.DataFrame,
                                   process_every_n_bars: int = 1,
                                   # use_ollama_mock parameter is implicitly handled by OllamaClient via config
                                   num_bars_for_context: int = 10): # How many past bars to feed into prompt
        """
        Iterates through historical feature data, generates signals, and stores them.
        """
        if features_df.empty:
            logger.info(f"No features data for {symbol} to process for historical signals.")
            return 0

        signals_generated_count = 0
        logger.info(f"Generating historical signals for {symbol}, processing every {process_every_n_bars} bar(s). Total bars: {len(features_df)}")

        for i in range(len(features_df)):
            if (i + 1) % process_every_n_bars != 0: # +1 because index i is 0-based
                continue # Skip this bar

            # Data for the current bar for which signal is to be generated
            current_bar_data_for_signal_timestamp = features_df.iloc[i]
            current_signal_timestamp_unix = int(current_bar_data_for_signal_timestamp.name.timestamp())

            # Context data: includes data up to and including the current bar for signal generation
            # The prompt will use `num_bars_for_context` from this slice.
            context_end_idx = i + 1
            context_start_idx = max(0, context_end_idx - num_bars_for_context)
            historical_context_slice = features_df.iloc[context_start_idx:context_end_idx]

            if historical_context_slice.empty:
                logger.warning(f"Not enough historical data for context at index {i} for {symbol}. Skipping signal generation.")
                continue

            # Mock news (as in main_signal_generator) - can be enhanced later
            mock_news = [f"Historical news context for {symbol} around {current_bar_data_for_signal_timestamp.name}."]

            market_context_str = format_market_context_prompt(
                symbol=symbol,
                ohlcv_features_df=historical_context_slice, # Pass the slice for context
                recent_news_list=mock_news,
                num_recent_features=num_bars_for_context # Use all rows in the slice for the prompt
            )
            full_prompt = get_trading_signal_prompt(market_context_str, symbol)

            # OllamaClient will use its configured mock setting
            raw_llm_response = self.ollama_c.generate_text(prompt=full_prompt)

            if raw_llm_response:
                parsed_signal = parse_llama_response(raw_llm_response)
                if parsed_signal and parsed_signal.get('signal'):
                    signal_to_store = {
                        "timestamp": current_signal_timestamp_unix, # Timestamp of the bar signal is for
                        "symbol": symbol,
                        "signal_type": f"AI_{self.ollama_c.default_model}_Hist",
                        "signal_action": parsed_signal['signal'],
                        "confidence": parsed_signal.get('confidence'),
                        "reasoning": parsed_signal.get('reasoning'),
                        "model_prompt": full_prompt,
                        "raw_model_response": raw_llm_response,
                        "related_feature_timestamp": current_signal_timestamp_unix,
                        "related_feature_symbol": symbol
                    }
                    if self.db_m.insert_trading_signal(signal_to_store):
                        signals_generated_count += 1
                        logger.debug(f"Stored historical signal for {symbol} at {current_bar_data_for_signal_timestamp.name}: {parsed_signal['signal']}")
                    else:
                        logger.error(f"Failed to store historical signal for {symbol} at {current_bar_data_for_signal_timestamp.name}")

            if (i+1) % (process_every_n_bars * 10) == 0 : # Log progress every 10 processed signals (if n_bars=1)
                 logger.info(f"Progress for {symbol}: Processed up to bar {i+1}/{len(features_df)}. Signals stored: {signals_generated_count}")


        logger.info(f"Finished generating historical signals for {symbol}. Total signals stored: {signals_generated_count}")
        return signals_generated_count

def run_historical_signal_generation(symbols_list: List[str],
                                     start_date_str: Optional[str] = None,
                                     end_date_str: Optional[str] = None,
                                     process_frequency: int = 1,
                                     context_bars: int = 10):
    """
    Main function to orchestrate historical signal generation for multiple symbols.
    """
    logger.info(f"Starting historical signal generation for symbols: {symbols_list}")
    config_file_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    cm = ConfigManager(config_file_path=config_file_path)

    db_path = cm.get('database.path', "data/trading_system.db")
    db_full_path = os.path.join(PROJECT_ROOT, db_path) if not os.path.isabs(db_path) else db_path
    db_m = DBManager(db_path=db_full_path)

    ollama_c = OllamaClient(config_manager=cm) # Will use mock settings from config if true

    # Ensure trading_signals table exists
    with db_m:
        db_m.create_trading_signals_table()

    generator = HistoricalSignalGenerator(cm, db_m, ollama_c)
    total_signals_for_all_symbols = 0

    for symbol_item in symbols_list:
        with db_m: # Ensure connection for each symbol processing
            features = generator.fetch_features_for_symbol_historical(symbol_item, start_date_str, end_date_str)

        if features is not None and not features.empty:
            # The DB connection for storing signals will be managed by insert_trading_signal via its own db_m context
            # Or, if generator's db_m is passed and used, it needs to be within a context.
            # Let's assume insert_trading_signal handles its own context or the main one is sufficient.
            # For safety, let's wrap the generation and storing part in a DB context too.
            with db_m:
                num_signals = generator.generate_and_store_signals(symbol_item, features, process_frequency, context_bars)
                total_signals_for_all_symbols += num_signals
        else:
            logger.warning(f"No feature data to process for {symbol_item}. Skipping historical signal generation.")

    logger.info(f"Historical signal generation complete for all symbols. Total signals generated: {total_signals_for_all_symbols}")

    # DEBUG: Read back signals immediately to check if they were written from this script's perspective
    with db_m:
        for symbol_item in symbols_list:
            test_query = "SELECT COUNT(*) FROM trading_signals WHERE symbol = ? AND signal_type = ?;"
            signal_type_name = f"AI_{ollama_c.default_model}_Hist" # Match how it's stored
            count_result = db_m.fetch_data(test_query, (symbol_item, signal_type_name))
            if count_result and count_result[0]:
                logger.info(f"DEBUG READBACK for {symbol_item} ({signal_type_name}): Found {count_result[0][0]} signals in DB immediately after generation.")
            else:
                logger.error(f"DEBUG READBACK for {symbol_item} ({signal_type_name}): Found 0 signals or error reading back.")


if __name__ == "__main__":
    # This ensures that if this script is run directly, it executes the historical generation.
    # The `use_ollama_mock` parameter is effectively controlled by `ollama_settings.mock_ollama_for_testing` in settings.yaml
    # when OllamaClient is initialized.

    logger.info("--- Running Historical Signal Generator Directly ---")
    # Using a short list of symbols from dummy data (21 rows each)
    # Process every bar for testing.
    # Use last 5 bars for context to reduce prompt size with dummy data.
    run_historical_signal_generation(
        symbols_list=['BTCUSD', 'ETHUSD'],
        process_frequency=1, # Process every bar
        context_bars=5      # Use last 5 bars of the historical_slice for context
    )
    logger.info("--- Historical Signal Generator Direct Run Finished ---")

    # Example: Query to check stored signals (manual verification)
    # db_m_check = DBManager(os.path.join(PROJECT_ROOT, ConfigManager(os.path.join(PROJECT_ROOT, "config", "settings.yaml")).get('database.path')))
    # with db_m_check:
    #     print("\nSample signals from trading_signals for BTCUSD:")
    #     btc_sigs = db_m_check.fetch_data("SELECT timestamp, symbol, signal_action, reasoning FROM trading_signals WHERE symbol='BTCUSD' ORDER BY timestamp DESC LIMIT 5")
    #     for sig in btc_sigs if btc_sigs else []: print(sig)
    #     print("\nSample signals from trading_signals for ETHUSD:")
    #     eth_sigs = db_m_check.fetch_data("SELECT timestamp, symbol, signal_action, reasoning FROM trading_signals WHERE symbol='ETHUSD' ORDER BY timestamp DESC LIMIT 5")
    #     for sig in eth_sigs if eth_sigs else []: print(sig)

def main(symbols: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None,
         process_freq: int = 1, context_window: int = 5):
    """Main entry point for historical signal generation, callable from orchestrator."""
    if symbols is None:
        symbols = ['BTCUSD', 'ETHUSD'] # Default if none provided

    logger.info(f"Running Historical Signal Generation via main() for {symbols}...")
    run_historical_signal_generation(
        symbols_list=symbols,
        start_date_str=start_date,
        end_date_str=end_date,
        process_frequency=process_freq,
        context_bars=context_window
    )
    logger.info("Historical Signal Generation main() call complete.")
