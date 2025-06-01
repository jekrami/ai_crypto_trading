# Placeholder for database management
# This module will handle connections and operations with the SQLite database.

import sqlite3
import os

# Assuming the database will be in the 'data' directory relative to the project root
# The actual path might be best managed via a configuration file
DEFAULT_DB_NAME = "trading_system.db"
DATA_DIR = "data"
# Construct the path relative to this file's location or project root
# For simplicity, let's assume the db_path will be passed or configured.
# If this script is in src/utils, then project_root is two levels up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, DATA_DIR, DEFAULT_DB_NAME)


class DBManager:
    def __init__(self, db_path=None):
        """
        Initializes the DBManager.
        :param db_path: Path to the SQLite database file.
                        Defaults to 'data/trading_system.db' relative to project root.
        """
        if db_path is None:
            self.db_path = DEFAULT_DB_PATH
        else:
            self.db_path = db_path

        # Ensure the data directory exists
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir)
                print(f"Created directory: {db_dir}")
            except OSError as e:
                print(f"Error creating directory {db_dir}: {e}")
                # Depending on requirements, might want to raise an error or log critical

        self.conn = None

    def connect(self):
        """Establishes a connection to the SQLite database."""
        if self.conn is None:
            try:
                self.conn = sqlite3.connect(self.db_path)
                print(f"Successfully connected to database: {self.db_path}")
            except sqlite3.Error as e:
                print(f"Error connecting to database {self.db_path}: {e}")
                # Potentially raise the error or handle it as per application requirements
                self.conn = None # Ensure conn is None if connection failed
        return self.conn

    def disconnect(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Disconnected from database.")

    def execute_query(self, query, params=None):
        """
        Executes a given SQL query (e.g., INSERT, UPDATE, DELETE).
        :param query: The SQL query string.
        :param params: Optional tuple of parameters to bind to the query.
        :return: True if execution was successful, False otherwise.
        """
        if not self.conn:
            print("Not connected to a database. Call connect() first.")
            return False

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            self.conn.commit()
            print(f"Query executed successfully: {query[:50]}...") # Log snippet of query
            return True
        except sqlite3.Error as e:
            print(f"Error executing query '{query[:50]}...': {e}")
            return False

    def fetch_data(self, query, params=None):
        """
        Fetches data from the database using a SELECT query.
        :param query: The SQL SELECT query string.
        :param params: Optional tuple of parameters to bind to the query.
        :return: A list of tuples representing the rows fetched, or None if an error occurs.
        """
        if not self.conn:
            print("Not connected to a database. Call connect() first.")
            return None

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            return rows
        except sqlite3.Error as e:
            print(f"Error fetching data with query '{query[:50]}...': {e}")
            return None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def create_trading_signals_table(self) -> None:
        """Creates the trading_signals table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS trading_signals (
            signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,                 -- Timestamp of the OHLCV bar this signal is based on
            symbol TEXT NOT NULL,                     -- Trading symbol, e.g., BTCUSD
            signal_type TEXT NOT NULL,                -- e.g., "AI_Llama3", "SMA_Cross_Random_5_10"
            generated_at DATETIME DEFAULT CURRENT_TIMESTAMP, -- When the signal was recorded in DB
            signal_action TEXT NOT NULL CHECK(signal_action IN ('BUY', 'SELL', 'HOLD')),
            confidence REAL,                          -- Confidence score of the signal (0.0 to 1.0)
            reasoning TEXT,                           -- Explanation from the model or strategy
            target_price REAL,                        -- Optional target price for the trade
            stop_loss_price REAL,                     -- Optional stop-loss price
            model_prompt TEXT,                        -- Full prompt sent to AI model, if applicable
            raw_model_response TEXT,                  -- Raw response from AI model, if applicable

            -- Columns for linking to the specific feature set that triggered this signal
            -- Replaces direct FK due to dynamic features_{symbol} table names
            related_feature_timestamp INTEGER, -- Corresponds to `timestamp` in `features_{symbol}`
            related_feature_symbol TEXT        -- Corresponds to the symbol part of `features_{symbol}`
        );
        """
        if self.execute_query(query):
            print("Table 'trading_signals' checked/created successfully.") # Or use logger
        else:
            print("Failed to create/check table 'trading_signals'.") # Or use logger

    def insert_trading_signal(self, signal_data: dict) -> bool:
        """
        Inserts a new trading signal into the trading_signals table.
        :param signal_data: A dictionary where keys are column names.
        :return: True if insertion was successful, False otherwise.
        """
        # Ensure all required fields are present if necessary, or let DB constraints handle it
        # For now, assume signal_data contains all necessary fields for insertion.

        columns = ', '.join(signal_data.keys())
        placeholders = ', '.join(['?'] * len(signal_data))
        query = f"INSERT INTO trading_signals ({columns}) VALUES ({placeholders})"

        try:
            if self.execute_query(query, tuple(signal_data.values())):
                # print(f"Trading signal inserted successfully: {signal_data.get('symbol')}, {signal_data.get('signal_action')}")
                return True
            else:
                # print(f"Failed to insert trading signal: {signal_data}")
                return False
        except Exception as e:
            # print(f"Exception during trading signal insertion: {e}")
            # Consider logging this exception with self.logger if available
            return False


# Example Usage (optional, can be removed or commented out)
if __name__ == "__main__":
    # This example assumes it's okay to create a dummy DB in the default location
    # For testing, it's often better to use an in-memory DB or a dedicated test DB path

    # Create a dummy data directory if it doesn't exist for the example
    example_db_dir = os.path.join(PROJECT_ROOT, DATA_DIR)
    if not os.path.exists(example_db_dir):
        os.makedirs(example_db_dir)
        print(f"Created example data directory: {example_db_dir}")

    db_manager = DBManager() # Uses default path: data/trading_system.db relative to project root

    # Create trading_signals table (idempotent)
    with db_manager as dbm_ctx:
        dbm_ctx.create_trading_signals_table()

    # Using context manager
    with DBManager() as dbm:
        if dbm.conn: # Check if connection was successful
            # Example: Create a table
            create_table_query = """
            CREATE TABLE IF NOT EXISTS example_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                value REAL
            );
            """
            dbm.execute_query(create_table_query)

            # Example: Insert data
            dbm.execute_query("INSERT INTO example_table (name, value) VALUES (?, ?)", ("sample_item", 123.45))
            dbm.execute_query("INSERT INTO example_table (name, value) VALUES (?, ?)", ("another_item", 67.89))

            # Example: Fetch data
            rows = dbm.fetch_data("SELECT * FROM example_table;")
            if rows:
                print("Fetched data:")
                for row in rows:
                    print(row)

            rows_filtered = dbm.fetch_data("SELECT * FROM example_table WHERE name = ?;", ("sample_item",))
            if rows_filtered:
                print("Fetched filtered data:")
                for row in rows_filtered:
                    print(row)
        else:
            print("Failed to connect to the database for example usage.")

    # Clean up the dummy database file created by the example, if desired
    # Be careful with this in a real scenario
    # if os.path.exists(DEFAULT_DB_PATH):
    #     print(f"Attempting to remove example database: {DEFAULT_DB_PATH}")
    #     os.remove(DEFAULT_DB_PATH)
    #     # Try to remove data directory if it's empty and was created by this script
    #     if not os.listdir(example_db_dir) and example_db_dir != os.path.join(PROJECT_ROOT, "data"): # Basic safety
    #         # os.rmdir(example_db_dir) # Be cautious with rmdir
    #         pass
    #     print(f"Example database {DEFAULT_DB_PATH} removed.")
