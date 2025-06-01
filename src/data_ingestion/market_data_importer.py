import os
import csv
import glob
from typing import List, Dict, Any

# Assuming utils are in src/utils and this script is in src/data_ingestion
# Adjust import paths if project structure differs or if using a different execution context
import sys
# Add src directory to Python path to allow direct imports
# This is often handled by setting PYTHONPATH or using a virtual environment installed in the project root
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /app/src
PROJECT_ROOT = os.path.dirname(SRC_DIR) # /app
sys.path.append(SRC_DIR)

from utils.db_manager import DBManager
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__) # Use module name for logger

DEFAULT_RAW_OHLCV_DIR = os.path.join(PROJECT_ROOT, "data", "raw_ohlcv") # Corrected path

class MarketDataImporter:
    def __init__(self, config_manager: ConfigManager, db_manager: DBManager):
        self.config_manager = config_manager
        self.db_manager = db_manager
        # Get path from config, default to DEFAULT_RAW_OHLCV_DIR (which is now project_root based)
        self.raw_data_dir = self.config_manager.get('data_paths.raw_ohlcv_dir',
                                                  os.path.join("data", "raw_ohlcv")) # Relative path from config

        # If the path from config is not absolute, join it with PROJECT_ROOT
        if not os.path.isabs(self.raw_data_dir):
            self.raw_data_dir = os.path.join(PROJECT_ROOT, self.raw_data_dir)


    def create_ohlcv_table(self, symbol: str) -> None:
        """
        Creates a table for OHLCV data for a given symbol if it doesn't exist.
        Table name will be ohlcv_{symbol}.
        """
        table_name = f"ohlcv_{symbol.replace('/', '_').upper()}" # Sanitize symbol for table name
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp INTEGER PRIMARY KEY,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL
        );
        """
        if self.db_manager.execute_query(query):
            logger.info(f"Table {table_name} checked/created successfully.")
        else:
            logger.error(f"Failed to create/check table {table_name}.")

    def parse_and_insert_csv_data(self, csv_file_path: str, symbol: str) -> None:
        """
        Parses OHLCV data from a CSV file and inserts it into the corresponding symbol's table.
        Uses INSERT OR IGNORE to avoid duplicate entries based on timestamp.
        """
        table_name = f"ohlcv_{symbol.replace('/', '_').upper()}"
        logger.info(f"Processing CSV file: {csv_file_path} for symbol {symbol} into table {table_name}")

        try:
            with open(csv_file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                rows_to_insert = []
                for row in reader:
                    try:
                        # Validate and convert data types
                        timestamp = int(float(row['timestamp'])) # Timestamps might be float in some CSVs
                        open_price = float(row['open'])
                        high_price = float(row['high'])
                        low_price = float(row['low'])
                        close_price = float(row['close'])
                        volume_val = float(row['volume'])
                        rows_to_insert.append(
                            (timestamp, open_price, high_price, low_price, close_price, volume_val)
                        )
                    except ValueError as ve:
                        logger.warning(f"Skipping row due to data conversion error in {csv_file_path}: {row} - {ve}")
                        continue
                    except KeyError as ke:
                        logger.warning(f"Skipping row due to missing column in {csv_file_path}: {row} - {ke}")
                        continue

                if not rows_to_insert:
                    logger.info(f"No valid data rows found in {csv_file_path}")
                    return

                # Use INSERT OR IGNORE to prevent issues with duplicate timestamps
                insert_query = f"""
                INSERT OR IGNORE INTO {table_name} (timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?);
                """

                inserted_count = 0
                for row_data in rows_to_insert:
                    if self.db_manager.execute_query(insert_query, row_data):
                        # This doesn't directly tell us if it was an INSERT or IGNORE,
                        # but execute_query returning True means the command ran.
                        # For more precise counting, one might need to check affected_rows if DB driver supports it.
                        pass # Assume success for now

                # A simple way to estimate insertions (not perfect due to IGNORE)
                # For a more accurate count of actual insertions vs ignored:
                # 1. Count rows before
                # 2. Execute all inserts
                # 3. Count rows after
                # This is more complex than needed for this example.
                # We can also fetch the last rowid if inserts are guaranteed.

                # Let's try to get a count of actual changes for better logging
                # This requires the cursor to be exposed or a method in DBManager
                # For now, we'll log the attempt.
                # A more robust way would be db_manager.execute_many_query() that returns rowcount.

                # Simplified approach for logging:
                # Check how many rows are in the table now, compared to before, for this specific data.
                # This is still tricky with IGNORE.
                # The key is that data gets in without error.
                logger.info(f"Attempted to insert {len(rows_to_insert)} rows from {csv_file_path} into {table_name}. "
                            f"Use SELECT COUNT(*) to verify actual new rows if needed.")

        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_file_path}")
        except Exception as e:
            logger.error(f"An error occurred while processing {csv_file_path}: {e}")

    def run_import(self) -> None:
        """
        Scans the raw data directory for '*-5.csv' files,
        creates tables, and ingests data.
        """
        logger.info(f"Starting market data import from directory: {self.raw_data_dir}")

        if not os.path.exists(self.raw_data_dir):
            logger.error(f"Raw OHLCV data directory not found: {self.raw_data_dir}")
            return

        # Using glob to find matching files. Path should be absolute or correctly relative.
        # Example pattern: /path/to/data/raw_ohlcv/*-5.csv
        search_pattern = os.path.join(self.raw_data_dir, "*-5.csv")
        logger.info(f"Scanning for files with pattern: {search_pattern}")

        csv_files = glob.glob(search_pattern)

        if not csv_files:
            logger.warning(f"No CSV files found matching pattern {search_pattern} in {self.raw_data_dir}")
            return

        logger.info(f"Found {len(csv_files)} CSV files to process: {csv_files}")

        with self.db_manager: # Ensure DB connection is managed (connect/disconnect)
            for csv_file_path in csv_files:
                filename = os.path.basename(csv_file_path)
                # Extract symbol, assuming format SYMBOL-TIMEFRAME.csv (e.g., BTCUSD-5.csv)
                parts = filename.split('-')
                if len(parts) > 1:
                    symbol = parts[0].upper() # e.g., BTCUSD
                    logger.info(f"Extracted symbol: {symbol} from filename: {filename}")

                    self.create_ohlcv_table(symbol)
                    self.parse_and_insert_csv_data(csv_file_path, symbol)
                else:
                    logger.warning(f"Could not extract symbol from filename: {filename}. Skipping.")

        logger.info("Market data import process finished.")

def main():
    """Main function to run the market data importer."""
    logger.info("Initializing market data importer...")

    # PROJECT_ROOT is already defined at the module level (/app)
    # SRC_DIR is /app/src

    config_file_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    # Ensure ConfigManager uses this path if not overridden
    temp_config_for_db_path = ConfigManager(config_file_path=config_file_path)
    db_path_from_config = temp_config_for_db_path.get('database.path', os.path.join("data","trading_system.db"))

    db_full_path = db_path_from_config
    if not os.path.isabs(db_full_path):
        db_full_path = os.path.join(PROJECT_ROOT, db_full_path)

    # Ensure the directory for the database exists
    db_dir = os.path.dirname(db_full_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        logger.info(f"Created database directory: {db_dir}")

    config_manager = ConfigManager(config_file_path=config_file_path)
    db_manager = DBManager(db_path=db_full_path)

    importer = MarketDataImporter(config_manager, db_manager)
    importer.run_import()

if __name__ == "__main__":
    # This allows the script to be run directly.
    # Ensure PYTHONPATH is set up correctly if running from a different directory,
    # or that the sys.path modifications at the top are effective. (sys.path.append(SRC_DIR))

    # Check if PyYAML is installed for ConfigManager
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is not installed. Please install it by running: pip install PyYAML")
        sys.exit(1)

    main()
