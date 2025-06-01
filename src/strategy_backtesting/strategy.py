import pandas as pd
import random
from typing import Dict, Any, Optional, List

# Configure logger if needed
# from utils.logger import setup_logger
# logger = setup_logger(__name__)

class BaseStrategy:
    """
    Base class for all trading strategies.
    """
    def __init__(self, strategy_name: str = "BaseStrategy", strategy_params: Optional[Dict[str, Any]] = None):
        """
        :param strategy_name: Name of the strategy.
        :param strategy_params: Dictionary of parameters specific to the strategy.
        """
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params if strategy_params is not None else {}
        # print(f"Initialized {self.strategy_name} with params: {self.strategy_params}") # Optional: for debugging

    def generate_signal(self, current_data_point: pd.Series, historical_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates a trading signal based on the current data point and historical data.
        This method MUST be implemented by subclasses.

        :param current_data_point: A pandas Series representing the current row of market data (OHLCV + features).
                                   The index of the Series is the column name, e.g., current_data_point['close'].
        :param historical_data: Pandas DataFrame containing all historical data up to (and including)
                                the current_data_point. Timestamps are expected to be the index.
        :return: A dictionary like {'action': 'BUY'/'SELL'/'HOLD', 'reasoning': 'optional explanation',
                                    'signal_type': 'strategy_name_params'}
                 or None if no signal is generated.
        """
        raise NotImplementedError("Subclasses must implement generate_signal()")

    def get_signal_type_name(self) -> str:
        """
        Returns a descriptive name for the signal type, often incorporating key parameters.
        Example: "SMA_Cross_10_30"
        """
        return self.strategy_name


class RandomStrategy(BaseStrategy):
    """
    A simple strategy that generates random BUY, SELL, or HOLD signals.
    Useful for testing the backtesting engine.
    """
    def __init__(self, strategy_params: Optional[Dict[str, Any]] = None):
        super().__init__(strategy_name="RandomStrategy", strategy_params=strategy_params)
        self.possible_actions = ['BUY', 'SELL', 'HOLD']
        # Optional: Seed for reproducibility during testing
        if 'random_seed' in self.strategy_params:
            random.seed(self.strategy_params['random_seed'])

    def generate_signal(self, current_data_point: pd.Series, historical_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        action = random.choice(self.possible_actions)
        return {
            'action': action,
            'reasoning': f'Random decision made at {current_data_point.name if current_data_point.name else "current time"}.',
            'signal_type': self.get_signal_type_name()
        }


class SmaCrossStrategy(BaseStrategy):
    """
    A strategy based on Simple Moving Average (SMA) crossovers.
    """
    def __init__(self, short_window: int = 10, long_window: int = 30, strategy_params: Optional[Dict[str, Any]] = None):
        # Merge default params with any provided strategy_params
        base_params = {'short_window': short_window, 'long_window': long_window}
        if strategy_params:
            base_params.update(strategy_params)

        super().__init__(strategy_name=f"SmaCrossStrategy_{short_window}_{long_window}", strategy_params=base_params)

        self.short_window = self.strategy_params['short_window']
        self.long_window = self.strategy_params['long_window']

        # Column names for SMAs - these must exist in the input data for generate_signal
        self.sma_short_col = f"sma_{self.short_window}" # Assuming feature pipeline names them this way
        self.sma_long_col = f"sma_{self.long_window}"

        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window for SMA Crossover strategy.")

    def get_signal_type_name(self) -> str:
        return f"SMA_Cross_{self.short_window}_{self.long_window}"

    def generate_signal(self, current_data_point: pd.Series, historical_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        # Ensure historical_data has enough rows to look back for previous SMAs
        # We need at least 2 rows in the relevant slice of historical_data to see a cross (current and previous)
        if len(historical_data) < 2:
            # print(f"{self.strategy_name}: Not enough historical data to check for crossover (need at least 2 rows, got {len(historical_data)}). Holding.")
            return {'action': 'HOLD', 'reasoning': 'Insufficient historical data for SMA crossover.', 'signal_type': self.get_signal_type_name()}

        # Check if required SMA columns exist in current_data_point (and thus in historical_data)
        if not all(col in current_data_point for col in [self.sma_short_col, self.sma_long_col]):
            # print(f"{self.strategy_name}: Required SMA columns ({self.sma_short_col}, {self.sma_long_col}) not in data. Holding.")
            return {'action': 'HOLD', 'reasoning': f'Missing SMA columns: {self.sma_short_col} or {self.sma_long_col}.', 'signal_type': self.get_signal_type_name()}

        # Get current and previous SMA values from the historical_data
        # current_data_point is effectively historical_data.iloc[-1]
        current_sma_short = current_data_point[self.sma_short_col]
        current_sma_long = current_data_point[self.sma_long_col]

        # Previous data point is historical_data.iloc[-2]
        previous_data_point = historical_data.iloc[-2]
        previous_sma_short = previous_data_point[self.sma_short_col]
        previous_sma_long = previous_data_point[self.sma_long_col]

        # Check for NaN values (can happen at the start of series or if data is too short)
        if pd.isna(current_sma_short) or pd.isna(current_sma_long) or \
           pd.isna(previous_sma_short) or pd.isna(previous_sma_long):
            # print(f"{self.strategy_name}: NaN values in SMA data. Holding.")
            return {'action': 'HOLD', 'reasoning': 'NaN values in SMA data, cannot determine crossover.', 'signal_type': self.get_signal_type_name()}

        action = 'HOLD'
        reasoning = 'No SMA crossover detected.' # More specific default reasoning

        # Bullish crossover: Short SMA crosses above Long SMA
        if previous_sma_short <= previous_sma_long and current_sma_short > current_sma_long:
            action = 'BUY'
            reasoning = f'Bullish crossover: {self.sma_short_col} ({current_sma_short:.2f}) crossed above {self.sma_long_col} ({current_sma_long:.2f}).'

        # Bearish crossover: Short SMA crosses below Long SMA
        elif previous_sma_short >= previous_sma_long and current_sma_short < current_sma_long:
            action = 'SELL'
            reasoning = f'Bearish crossover: {self.sma_short_col} ({current_sma_short:.2f}) crossed below {self.sma_long_col} ({current_sma_long:.2f}).'

        return {'action': action, 'reasoning': reasoning, 'signal_type': self.get_signal_type_name()}


if __name__ == '__main__':
    print("--- Testing Strategies ---")

    # Create dummy historical data for testing
    data_dict = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00', '2023-01-01 12:00', '2023-01-01 13:00']),
        'close': [100, 105, 95, 110],
        'sma_1': [100, 102, 98, 108], # Short SMA (e.g. period 1 for simplicity)
        'sma_2': [100, 101, 100, 102]  # Long SMA (e.g. period 2 for simplicity)
    }
    sample_hist_df = pd.DataFrame(data_dict).set_index('timestamp')

    # Test RandomStrategy
    print("\n-- RandomStrategy Test --")
    random_strat = RandomStrategy(strategy_params={'random_seed': 42})
    for i in range(len(sample_hist_df)):
        current_dp = sample_hist_df.iloc[i]
        hist_up_to_current = sample_hist_df.iloc[:i+1]
        signal = random_strat.generate_signal(current_dp, hist_up_to_current)
        print(f"Time: {current_dp.name}, Close: {current_dp['close']}, Signal: {signal}")

    # Test SmaCrossStrategy
    print("\n-- SmaCrossStrategy Test --")
    # For this test, make sure sma_short_col and sma_long_col match the dummy data
    # SmaCrossStrategy will use 'sma_1' and 'sma_2' based on its init params if we set short_window=1, long_window=2
    sma_cross_strat = SmaCrossStrategy(short_window=1, long_window=2)
                                       # This will look for 'sma_1' and 'sma_2' columns

    # Manually create data that shows a crossover for SmaCrossStrategy
    sma_data = {
        'timestamp': pd.to_datetime([
            '2023-01-01 10:00', '2023-01-01 11:00', '2023-01-01 12:00',
            '2023-01-01 13:00', '2023-01-01 14:00'
        ]),
        'close':             [100, 101, 102, 103, 104],
        sma_cross_strat.sma_short_col: [99,  100, 103, 104, 103], # Short SMA values
        sma_cross_strat.sma_long_col:  [101, 101, 102, 102, 105]  # Long SMA values
        # Previous: 99 <= 101 (no cross) -> Current: 100 <= 101 (no cross) -> HOLD
        # Previous: 100 <= 101 (no cross) -> Current: 103 > 102 (BULLISH CROSS) -> BUY
        # Previous: 103 > 102 (no cross) -> Current: 104 > 102 (no cross) -> HOLD
        # Previous: 104 > 102 (no cross) -> Current: 103 < 105 (BEARISH CROSS) -> SELL
    }
    sma_test_df = pd.DataFrame(sma_data).set_index('timestamp')

    print("SMA Test DataFrame:")
    print(sma_test_df)

    for i in range(len(sma_test_df)):
        current_dp = sma_test_df.iloc[i]
        hist_up_to_current = sma_test_df.iloc[:i+1]

        # Skip first point if strategy needs lookback
        if i == 0 and isinstance(sma_cross_strat, SmaCrossStrategy): # SmaCross needs at least 2 points
            print(f"Time: {current_dp.name}, Close: {current_dp['close']}, Signal: Initializing (HOLD or No Signal)")
            continue

        signal = sma_cross_strat.generate_signal(current_dp, hist_up_to_current)
        print(f"Time: {current_dp.name}, Close: {current_dp['close']}, Signal: {signal}")

    print("\n-- SmaCrossStrategy Test with NaNs --")
    nan_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00', '2023-01-01 12:00']),
        'close': [100, 101, 102],
        sma_cross_strat.sma_short_col: [pd.NA, 100, 101], # Using actual column names from strategy
        sma_cross_strat.sma_long_col:  [pd.NA, 99, 100]
    }
    nan_test_df = pd.DataFrame(nan_data).set_index('timestamp')
    print("SMA Test DataFrame with NaNs:")
    print(nan_test_df)
    for i in range(len(nan_test_df)):
        current_dp = nan_test_df.iloc[i]
        hist_up_to_current = nan_test_df.iloc[:i+1]
        if i == 0 and isinstance(sma_cross_strat, SmaCrossStrategy):
            print(f"Time: {current_dp.name}, Close: {current_dp['close']}, Signal: Initializing (HOLD or No Signal)")
            continue
        signal = sma_cross_strat.generate_signal(current_dp, hist_up_to_current)
        print(f"Time: {current_dp.name}, Close: {current_dp['close']}, Signal: {signal}")


class HistoricalAiSignalStrategy(BaseStrategy):
    """
    A strategy that uses pre-generated AI signals from the database.
    """
    def __init__(self, db_manager: Any, symbol: str,
                 signal_type_filter: str = "AI_Llama3_Hist",
                 strategy_params: Optional[Dict[str, Any]] = None):

        super().__init__(strategy_name=f"HistoricalAiSignalStrategy_{symbol}_{signal_type_filter}",
                         strategy_params=strategy_params)
        self.db_m_passed_in = db_manager # Keep a reference if needed for other things, but don't use for _load_historical_signals directly for this test
        self.symbol = symbol.upper()
        self.signal_type_filter = signal_type_filter
        self.ai_signals_df = self._load_historical_signals() # This will now use its own DBManager

        if self.ai_signals_df.empty:
            print(f"Warning: No historical AI signals found for {self.symbol} with type '{self.signal_type_filter}'. Strategy will always HOLD.")

    def _get_fresh_db_manager(self) -> Any: # Actual return type is DBManager but Any to avoid circular import if db_manager itself imports strategy
        """Helper to get a fresh DBManager instance. Assumes utils.db_manager.DBManager exists."""
        # This assumes DBManager can be imported and path is okay.
        # This is a bit of a hack to ensure fresh connection for reading.
        # Proper solution might involve better DB session management at higher level.
        from utils.db_manager import DBManager
        from utils.config_manager import ConfigManager # Needed by DBManager default path logic
        import os

        # Simplified: assumes DB path can be found via a default ConfigManager if needed by DBManager's init
        # Ideally, the DB path should be consistently passed or globally accessible.
        # For this test, let's assume default path logic in DBManager works or path is hardcoded if no CM.
        # The main scripts usually set up DBManager with a path from ConfigManager.
        # We need to replicate that here if we want a truly independent instance.
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_file_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
        cm = ConfigManager(config_file_path=config_file_path)
        db_path_from_config = cm.get('database.path', "data/trading_system.db")
        db_full_path = db_path_from_config
        if not os.path.isabs(db_full_path):
            db_full_path = os.path.join(PROJECT_ROOT, db_full_path)

        # Speculative delay for sandbox FS
        import time
        time.sleep(0.1) # Wait 100ms

        print(f"DEBUG HistoricalAiSignalStrategy: _get_fresh_db_manager() connecting to DB at: {db_full_path}")
        return DBManager(db_path=db_full_path)


    def _load_historical_signals(self) -> pd.DataFrame:
        """Loads historical AI signals from the trading_signals table using a fresh DBManager."""

        fresh_db_m = self._get_fresh_db_manager()

        # DEBUG: Fetch ALL signals first to see what's in the table from this connection's perspective
        all_signals_debug_query = "SELECT timestamp, symbol, signal_type, signal_action FROM trading_signals ORDER BY timestamp ASC;"
        all_signals_data = []
        with fresh_db_m:
            print(f"DEBUG HistoricalAiSignalStrategy: Attempting to fetch ALL signals for diagnostics from DB path {fresh_db_m.db_path}...")
            all_signals_data = fresh_db_m.fetch_data(all_signals_debug_query)
            if all_signals_data:
                print(f"DEBUG HistoricalAiSignalStrategy: Diagnostic fetch found {len(all_signals_data)} total signals in trading_signals.")
                # for row_idx, sig_row in enumerate(all_signals_data[:5]): # Print first 5
                #     print(f"DEBUG HistoricalAiSignalStrategy: AllSignals Row {row_idx}: {sig_row}")
            else:
                print("DEBUG HistoricalAiSignalStrategy: Diagnostic fetch found 0 total signals in trading_signals.")

        # Original targeted query
        query = """
        SELECT timestamp, signal_action, reasoning
        FROM trading_signals
        WHERE symbol = ? AND signal_type = ?
        ORDER BY timestamp ASC;
        """
        params = (self.symbol, self.signal_type_filter)

        raw_data_from_db = []
        with fresh_db_m:
            raw_data_from_db = fresh_db_m.fetch_data(query, params)

        print(f"DEBUG HistoricalAiSignalStrategy (fresh DB conn): Fetched {len(raw_data_from_db) if raw_data_from_db else 0} targeted signals from DB for {self.symbol} / {self.signal_type_filter}")

        if not raw_data_from_db:
            return pd.DataFrame(columns=['signal_action', 'reasoning']) # Empty DataFrame with expected columns

        df = pd.DataFrame(raw_data_from_db, columns=['timestamp', 'signal_action', 'reasoning'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True) # Assuming Unix timestamps
        df.set_index('timestamp', inplace=True)
        # print(f"Loaded {len(df)} historical signals.")
        return df

    def generate_signal(self, current_data_point: pd.Series, historical_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        # current_data_point.name should be the timestamp if index is DatetimeIndex
        current_timestamp = pd.Timestamp(current_data_point.name, tz='UTC') if isinstance(current_data_point.name, pd.Timestamp) else pd.to_datetime(current_data_point.name, unit='s', utc=True)


        if not self.ai_signals_df.empty and current_timestamp in self.ai_signals_df.index:
            signal_row = self.ai_signals_df.loc[current_timestamp]
            action = signal_row['signal_action']
            reasoning = signal_row['reasoning']
            # print(f"Signal found for {current_timestamp}: {action}")
            return {
                'action': action,
                'reasoning': reasoning if pd.notna(reasoning) else "AI signal from history.",
                'signal_type': self.get_signal_type_name()
            }
        else:
            # print(f"No AI signal for {current_timestamp}. Holding.")
            return {
                'action': 'HOLD',
                'reasoning': f'No AI signal available for {current_timestamp}.',
                'signal_type': self.get_signal_type_name()
            }
