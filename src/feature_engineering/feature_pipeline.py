import os
import sys
import pandas as pd
from typing import List, Optional, Dict, Any

# Setup paths
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /app/src
PROJECT_ROOT = os.path.dirname(SRC_DIR) # /app
sys.path.append(SRC_DIR)

from utils.db_manager import DBManager
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

# Feature engineering modules
from feature_engineering import technical_indicators as ti
from feature_engineering import ict_features # Placeholder
from feature_engineering import news_features # Placeholder

logger = setup_logger(__name__)

# --- Database Helper Functions ---
def get_db_manager() -> DBManager:
    """Initializes and returns a DBManager instance."""
    config_file_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    # Temp ConfigManager to get DB path
    temp_config_for_db_path = ConfigManager(config_file_path=config_file_path)
    db_path_from_config = temp_config_for_db_path.get('database.path', os.path.join("data", "trading_system.db"))

    db_full_path = db_path_from_config
    if not os.path.isabs(db_full_path):
        db_full_path = os.path.join(PROJECT_ROOT, db_full_path)

    db_dir = os.path.dirname(db_full_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir) # Ensure DB directory exists
        logger.info(f"Created database directory: {db_dir}")

    return DBManager(db_path=db_full_path)

def fetch_ohlcv_data(db_manager: DBManager, symbol: str) -> Optional[pd.DataFrame]:
    """Fetches OHLCV data for a symbol from the database and returns it as a DataFrame."""
    table_name = f"ohlcv_{symbol.upper()}"
    query = f"SELECT * FROM {table_name} ORDER BY timestamp ASC;"

    logger.info(f"Fetching OHLCV data for {symbol} from table {table_name}...")
    try:
        # Assuming db_manager.fetch_data returns list of tuples and column names are known
        # Or, if db_manager can return a DataFrame directly, that's better.
        # For now, let's assume it returns list of tuples.
        # We need column names. One way: PRAGMA table_info(table_name);

        # Simplified: fetch_data in current DBManager returns list of tuples.
        # We know the schema: timestamp, open, high, low, close, volume
        data = db_manager.fetch_data(query)
        if not data:
            logger.warning(f"No data found for {symbol} in table {table_name}.")
            return None

        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(data, columns=cols)

        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') # Assuming Unix timestamps
        df.set_index('timestamp', inplace=True)

        logger.info(f"Successfully fetched {len(df)} rows for {symbol}.")
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol}: {e}", exc_info=True)
        return None

def create_features_table(db_manager: DBManager, symbol: str, df_with_features: pd.DataFrame) -> None:
    """
    Creates a table for storing features if it doesn't exist.
    The table schema is derived from the DataFrame columns and dtypes.
    'timestamp' column from DataFrame index is expected.
    """
    table_name = f"features_{symbol.upper()}"

    # Prepare column definitions from DataFrame
    cols_with_types = []
    # Timestamp from index
    if isinstance(df_with_features.index, pd.DatetimeIndex):
        cols_with_types.append("timestamp INTEGER PRIMARY KEY") # Store Unix timestamp as INTEGER
    else:
        logger.error(f"DataFrame index is not a DatetimeIndex for {symbol}. Cannot create features table.")
        return

    for col_name, dtype in df_with_features.dtypes.items():
        sql_type = "REAL" # Default for numeric types
        if pd.api.types.is_integer_dtype(dtype):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "REAL"
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            sql_type = "TEXT"
        elif pd.api.types.is_datetime64_any_dtype(dtype): # Should not happen for value columns if index is timestamp
            sql_type = "INTEGER" # Store as Unix timestamp
        else:
            logger.warning(f"Unsupported dtype {dtype} for column {col_name} in {symbol}. Defaulting to TEXT.")
            sql_type = "TEXT"
        cols_with_types.append(f"\"{col_name}\" {sql_type}") # Quote column names

    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(cols_with_types)});"

    logger.info(f"Defining features table {table_name} with query: {create_table_query}")
    if db_manager.execute_query(create_table_query):
        logger.info(f"Table '{table_name}' checked/created successfully.")
    else:
        logger.error(f"Failed to create/check table '{table_name}'.")


def store_features(db_manager: DBManager, symbol: str, df_with_features: pd.DataFrame) -> None:
    """Stores the DataFrame with features into the corresponding features_{symbol} table."""
    table_name = f"features_{symbol.upper()}"

    if df_with_features.empty:
        logger.warning(f"DataFrame for {symbol} is empty. Nothing to store.")
        return

    # Prepare DataFrame for SQLite: reset index to have 'timestamp' as a column
    # and convert datetime to Unix timestamp (integer seconds)
    df_to_store = df_with_features.copy()
    if isinstance(df_to_store.index, pd.DatetimeIndex):
        df_to_store.reset_index(inplace=True)
        # Convert timestamp to Unix epoch seconds (integer) for PK
        df_to_store['timestamp'] = (df_to_store['timestamp'].astype(int) // 10**9).astype(int)
    else:
        logger.error("DataFrame index is not DatetimeIndex, 'timestamp' column might be missing or incorrect for storage.")
        # Attempt to find a 'timestamp' column if not from index
        if 'timestamp' not in df_to_store.columns:
            logger.error(f"'timestamp' column not found in DataFrame for {symbol}. Cannot store features.")
            return
        # If 'timestamp' column exists and is datetime, convert it
        if pd.api.types.is_datetime64_any_dtype(df_to_store['timestamp']):
             df_to_store['timestamp'] = (df_to_store['timestamp'].astype(int) // 10**9).astype(int)


    # Using INSERT OR REPLACE strategy:
    # Convert DataFrame to list of dictionaries for row-by-row insertion
    records = df_to_store.to_dict(orient='records')
    if not records:
        logger.warning(f"No records to store for {symbol} after processing DataFrame.")
        return

    # Dynamically create column names and placeholders for the query
    # Ensure column names are quoted if they contain special characters or are keywords
    columns = [f'"{col}"' for col in records[0].keys()]
    placeholders = ', '.join(['?'] * len(columns))
    insert_query = f"INSERT OR REPLACE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders});"

    logger.info(f"Storing {len(records)} feature rows for {symbol} into {table_name} using INSERT OR REPLACE.")

    success_count = 0
    fail_count = 0
    for record in records:
        params = tuple(record.values())
        if db_manager.execute_query(insert_query, params):
            success_count += 1
        else:
            fail_count += 1
            logger.debug(f"Failed to insert record: {record} for symbol {symbol}")

    if fail_count > 0:
        logger.warning(f"Failed to insert {fail_count} records for {symbol} into {table_name}.")
    logger.info(f"Successfully inserted/replaced {success_count} records for {symbol} in {table_name}.")


# --- Main Orchestration ---
def run_feature_engineering(symbol_list: Optional[List[str]] = None) -> None:
    """
    Main orchestration function for the feature engineering pipeline.
    Fetches OHLCV data, computes features, and stores them in the database.
    """
    logger.info("Starting feature engineering pipeline...")

    db_m = get_db_manager() # Use a single DBManager instance

    if symbol_list is None:
        # Discover symbols from ohlcv_* tables if symbol_list is not provided
        # This is a more advanced feature. For now, use a default or config.
        # Example: query "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ohlcv_%';"
        # For this subtask, let's use a fixed list if None.
        logger.info("No symbol list provided, using default: ['BTCUSD', 'ETHUSD']")
        symbol_list = ['BTCUSD', 'ETHUSD'] # Default symbols from dummy data

    with db_m: # Ensure DB connection is managed
        for symbol in symbol_list:
            logger.info(f"Processing symbol: {symbol}")

            # 1. Fetch OHLCV data
            ohlcv_df = fetch_ohlcv_data(db_m, symbol)
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning(f"Skipping feature engineering for {symbol} due to missing OHLCV data.")
                continue

            # Make a copy for feature addition to avoid modifying the original fetched df
            features_df = ohlcv_df.copy()

            # 2. Apply Technical Indicators
            logger.info(f"Calculating technical indicators for {symbol}...")
            try:
                features_df = ti.add_sma(features_df, length=20)
                features_df = ti.add_sma(features_df, length=50)
                features_df = ti.add_ema(features_df, length=20)
                features_df = ti.add_rsi(features_df, length=14)
                features_df = ti.add_macd(features_df, fast=12, slow=26, signal=9)
                features_df = ti.add_bollinger_bands(features_df, length=20, std=2)
                features_df = ti.add_atr(features_df, length=14)
                logger.info(f"Technical indicators added for {symbol}.")
            except Exception as e:
                logger.error(f"Error adding technical indicators for {symbol}: {e}", exc_info=True)
                # Optionally, continue with features_df that might have partial TIs, or skip storing.
                # For now, we proceed.

            # 3. (Placeholder) Apply ICT Features
            logger.info(f"Applying (placeholder) ICT features for {symbol}...")
            features_df = ict_features.find_fair_value_gaps(features_df)
            features_df = ict_features.identify_order_blocks(features_df)
            # ... any other ICT feature functions ...

            # 4. (Placeholder) Apply News Features
            # This would require fetching news data, potentially from `news_articles` table
            # For now, we'll just call the placeholder.
            logger.info(f"Applying (placeholder) News features for {symbol}...")
            # Dummy news_df for placeholder function signature
            # In a real scenario, fetch relevant news from DB:
            # news_table_data = db_m.fetch_data("SELECT published_at, title FROM news_articles ...")
            # sample_news_df = pd.DataFrame(news_table_data, columns=['published_at', 'title'])
            sample_news_df = pd.DataFrame({'published_at': pd.to_datetime([]), 'title': []}) # Empty DF

            sentiment_series = news_features.calculate_news_sentiment_score(sample_news_df, features_df)
            features_df['news_sentiment'] = sentiment_series # Assign placeholder scores

            # Remove original OHLCV columns if only features are to be stored,
            # or keep them if the features table should be self-contained with OHLCV + features.
            # Current approach: keep OHLCV in the features table too.
            # Ensure timestamp is the index before creating/storing.
            if not isinstance(features_df.index, pd.DatetimeIndex):
                 if 'timestamp' in features_df.columns:
                     features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
                     features_df.set_index('timestamp', inplace=True)
                 else:
                     logger.error(f"Timestamp information lost or incorrect for {symbol}. Cannot proceed with storing features.")
                     continue

            # Drop columns that are all NaNs, which can happen if TA calculations fail or data is too short
            # features_df.dropna(axis=1, how='all', inplace=True)
            # Be careful with this, as some TIs might have NaNs at the beginning.
            # Better to handle NaNs specifically if needed (e.g., fill, or ensure pandas_ta handles short periods gracefully by returning NaNs)

            # 5. Create features table (if not exists) and store features
            if not features_df.empty:
                logger.info(f"Preparing to store features for {symbol}. DataFrame shape: {features_df.shape}")
                create_features_table(db_m, symbol, features_df) # Create table based on final df schema
                store_features(db_m, symbol, features_df)
            else:
                logger.warning(f"No features generated or data became empty for {symbol}. Nothing to store.")

            logger.info(f"Finished processing symbol: {symbol}")

    logger.info("Feature engineering pipeline finished.")

if __name__ == "__main__":
    # Ensure required libraries are installed for the environment where this runs
    try:
        import pandas
        import pandas_ta
        import numpy
    except ImportError as e:
        logger.error(f"Missing critical dependency for feature_pipeline: {e}. Please install pandas, pandas-ta, numpy.")
        sys.exit(1)

    logger.info("Running Feature Engineering Pipeline directly for BTCUSD and ETHUSD...")
    # This assumes ohlcv_BTCUSD and ohlcv_ETHUSD tables exist from previous data ingestion steps
    run_feature_engineering(symbol_list=['BTCUSD', 'ETHUSD'])
    logger.info("Direct run of Feature Engineering Pipeline complete.")

    # Example: How to query the features table afterwards (manual check)
    # db_manager = get_db_manager()
    # with db_manager:
    #     btc_features = db_manager.fetch_data("SELECT * FROM features_BTCUSD LIMIT 5")
    #     if btc_features:
    #         print("\nSample BTCUSD features from DB:")
    #         for row in btc_features:
    #             print(row)

def main(symbols: Optional[List[str]] = None):
    """Main entry point for feature engineering pipeline, callable from orchestrator."""
    if symbols is None:
        symbols = ['BTCUSD', 'ETHUSD'] # Default if none provided
    logger.info(f"Running Feature Engineering Pipeline via main() for {symbols}...")
    run_feature_engineering(symbol_list=symbols)
    logger.info("Feature Engineering Pipeline main() call complete.")
    #     eth_features = db_manager.fetch_data("SELECT * FROM features_ETHUSD LIMIT 5")
    #     if eth_features:
    #         print("\nSample ETHUSD features from DB:")
    #         for row in eth_features:
    #             print(row)
