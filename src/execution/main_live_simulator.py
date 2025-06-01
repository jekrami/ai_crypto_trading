import os
import sys
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

# Setup paths
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /app/src
PROJECT_ROOT = os.path.dirname(SRC_DIR) # /app
sys.path.append(SRC_DIR) # Add src to path to allow imports from utils, etc.

from utils.db_manager import DBManager
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

from execution.simulated_exchange import SimulatedExchange
from strategy_backtesting.strategy import BaseStrategy, RandomStrategy # Import specific strategies needed
# For risk management modules, ensure they are importable
from risk_management import position_sizer
from risk_management import order_modifiers

logger = setup_logger(__name__)

class LiveSimulator:
    def __init__(self,
                 config_manager: ConfigManager,
                 db_manager: DBManager, # Not directly used in this version, but good for future state loading/saving
                 strategy_instance: BaseStrategy,
                 symbol: str,
                 initial_ohlcv_data: pd.DataFrame):
        """
        Initializes the Live Simulator.

        :param config_manager: Instance of ConfigManager for settings.
        :param db_manager: Instance of DBManager (can be used for logging trades to DB later).
        :param strategy_instance: An instantiated strategy object.
        :param symbol: The trading symbol, e.g., "BTCUSD".
        :param initial_ohlcv_data: DataFrame containing OHLCV and feature data, indexed by timestamp.
        """
        self.cm = config_manager
        self.db_m = db_manager # Store for potential future use (e.g. logging trades to system DB)
        self.strategy = strategy_instance
        self.symbol = symbol.upper()

        # Parse symbol into base and quote assets
        assets = SimulatedExchange()._get_assets_from_symbol(self.symbol) # Use helper from SimEx
        if not assets:
            raise ValueError(f"Could not parse symbol '{self.symbol}' in LiveSimulator.")
        self.asset1, self.asset2 = assets # e.g., BTC, USD

        # Ensure data is sorted by timestamp if not already
        self.ohlcv_data = initial_ohlcv_data.sort_index() if isinstance(initial_ohlcv_data.index, pd.DatetimeIndex) else initial_ohlcv_data.sort_values(by='timestamp')


        # Initialize SimulatedExchange
        sim_ex_defaults = self.cm.get('simulated_exchange_defaults', {})
        initial_balances = sim_ex_defaults.get('initial_balances', {"USD": 10000.0, self.asset1: 0.0})
        # Ensure quote asset (asset2) is in initial_balances if not USD
        if self.asset2 not in initial_balances: initial_balances[self.asset2] = initial_balances.get("USD", 10000.0)

        sim_fee = sim_ex_defaults.get('fee_percent', 0.001)
        self.sim_exchange = SimulatedExchange(initial_balances=initial_balances, fee_percent=sim_fee)
        logger.info(f"SimulatedExchange initialized for LiveSimulator. Balances: {self.sim_exchange.balances}")

        self.open_position: Optional[Dict[str, Any]] = None # e.g., {'units': X, 'entry_price': Y, 'stop_loss': Z, 'take_profit': A, 'side': 'LONG'/'SHORT'}

        # Load risk management defaults (these are also loaded by Backtester, consider centralizing if DRY needed)
        self.risk_per_trade_percent = self.cm.get('risk_management_defaults.risk_per_trade_percent', 0.01)
        self.default_sl_config = self.cm.get('risk_management_defaults.default_stop_loss', {'type': 'PERCENTAGE', 'value': 0.02})
        self.default_tp_config = self.cm.get('risk_management_defaults.default_take_profit', {'type': 'RISK_REWARD_RATIO', 'value': 1.5})
        self.asset_price_increment = self.cm.get(f'risk_management_defaults.{self.asset1.lower()}_price_increment', 0.01)

        logger.info(f"LiveSimulator for {self.symbol} initialized with strategy {self.strategy.strategy_name}.")

    def _get_current_equity(self, current_price_asset1_in_asset2: float) -> float:
        """Calculates current total equity in terms of asset2 (quote currency)."""
        equity = self.sim_exchange.get_balance(self.asset2)
        asset1_balance = self.sim_exchange.get_balance(self.asset1)
        if asset1_balance > 0:
            equity += asset1_balance * current_price_asset1_in_asset2
        return equity

    def run_simulation(self):
        logger.info(f"Starting live simulation for {self.symbol} with {len(self.ohlcv_data)} data points.")

        if self.ohlcv_data.empty:
            logger.warning("No OHLCV data provided to simulator. Exiting run_simulation.")
            return

        for current_bar_tuple in self.ohlcv_data.itertuples(): # itertuples is efficient
            # Convert namedtuple to Series to match strategy.generate_signal expectations
            current_bar_data = pd.Series(current_bar_tuple._asdict(), index=current_bar_tuple._fields)
            # Pandas itertuples includes 'Index' as first field, which is the timestamp here
            current_timestamp = current_bar_data.Index
            current_price = current_bar_data.close # Use close price of current bar for decisions/execution

            logger.debug(f"Processing bar: Timestamp={current_timestamp}, Price={current_price}")

            # --- 1. Manage Open Position (Check SL/TP) ---
            exit_trade_result = None
            if self.open_position:
                pos_side = self.open_position['side']
                sl_price = self.open_position['stop_loss']
                tp_price = self.open_position['take_profit']
                exit_event_price = None # Price at which SL/TP is triggered
                exit_reason = None

                if pos_side == "LONG":
                    if current_bar_data.low <= sl_price:
                        exit_event_price, exit_reason = sl_price, "STOP_LOSS"
                    elif tp_price and current_bar_data.high >= tp_price:
                        exit_event_price, exit_reason = tp_price, "TAKE_PROFIT"
                # elif pos_side == "SHORT": # TODO: Implement short selling logic
                #     pass

                if exit_reason and exit_event_price is not None:
                    logger.info(f"{exit_reason} triggered for {pos_side} position at price {exit_event_price} (Bar: L={current_bar_data.low}, H={current_bar_data.high})")
                    exit_signal_action = "SELL" if pos_side == "LONG" else "BUY" # Opposite action to close
                    exit_trade_result = self.sim_exchange.execute_market_order(
                        self.symbol, exit_signal_action, self.open_position['units'],
                        exit_event_price, # Execute at SL/TP price (simplification)
                        current_timestamp
                    )
                    logger.info(f"Position closed via {exit_reason}: {exit_trade_result}")
                    if exit_trade_result and exit_trade_result['status'] == 'FILLED':
                        self.open_position = None
                    else:
                        logger.error(f"Failed to execute {exit_reason} order: {exit_trade_result.get('reason')}")
                        # Critical error, portfolio state might be inconsistent. May need to halt or handle.

            # --- 2. Generate New Signal & Process (if no SL/TP exit occurred on this bar) ---
            if not exit_trade_result or (exit_trade_result and exit_trade_result['status'] != 'FILLED'):
                # Get historical data up to the *previous* bar for signal generation,
                # as current bar's data (esp. close) isn't fully known until it closes.
                # For simplicity in this example, strategy uses data up to and including current_bar_data,
                # which implies signal is generated at close of current_bar and action for next.
                # This is a common simplification but be aware of lookahead bias if not careful.
                # Let's assume strategy acts on current bar's close, for execution on "next tick" (simulated here as current_price).

                # Find index of current_bar_tuple.Index in self.ohlcv_data.index
                current_bar_idx = self.ohlcv_data.index.get_loc(current_timestamp)
                historical_slice = self.ohlcv_data.iloc[:current_bar_idx + 1]

                strategy_signal_dict = self.strategy.generate_signal(current_bar_data, historical_slice)
                action = strategy_signal_dict['action'] if strategy_signal_dict else 'HOLD'

                logger.debug(f"Strategy signal: {action} at {current_timestamp}")

                # --- 3. Process Signal (Entry Logic - LONG only for now) ---
                if action == 'BUY' and not self.open_position:
                    entry_price_candidate = current_price # Use current bar's close as entry price

                    sl_price = order_modifiers.calculate_stop_loss(
                        entry_price_candidate, "LONG", self.default_sl_config,
                        atr_value=current_bar_data.get('atr_14'), # Assumes 'atr_14' from features
                        recent_low=current_bar_data.low # Simplistic: use current bar's low
                    )
                    if sl_price is None:
                        logger.warning(f"BUY signal at {current_timestamp}: Could not calculate SL. Skipping trade.")
                    else:
                        tp_price = order_modifiers.calculate_take_profit(
                            entry_price_candidate, sl_price, "LONG", self.default_tp_config,
                            atr_value=current_bar_data.get('atr_14')
                        )
                        sl_distance = abs(entry_price_candidate - sl_price)

                        if sl_distance < 1e-6: # Avoid zero or tiny SL distance
                            logger.warning(f"BUY signal at {current_timestamp}: SL distance too small ({sl_distance}). Skipping trade.")
                        else:
                            current_equity = self._get_current_equity(current_price)
                            units_to_buy = position_sizer.calculate_position_size(
                                current_equity, self.risk_per_trade_percent,
                                sl_distance, entry_price_candidate, self.asset_price_increment
                            )
                            logger.info(f"BUY signal: Attempting to buy {units_to_buy} {self.asset1} at ~{entry_price_candidate}. SL: {sl_price}, TP: {tp_price}, Equity: {current_equity}")

                            if units_to_buy > 0:
                                entry_trade_result = self.sim_exchange.execute_market_order(
                                    self.symbol, 'BUY', units_to_buy,
                                    entry_price_candidate, # Execute at current_price (close of bar)
                                    current_timestamp
                                )
                                logger.info(f"Entry trade execution result: {entry_trade_result}")
                                if entry_trade_result['status'] == 'FILLED':
                                    self.open_position = {
                                        'units': entry_trade_result['filled_quantity_asset1'],
                                        'entry_price': entry_trade_result['price'],
                                        'stop_loss': sl_price,
                                        'take_profit': tp_price,
                                        'side': 'LONG'
                                    }
                                    logger.info(f"Position opened: {self.open_position}")
                                else:
                                    logger.warning(f"Failed to open BUY position: {entry_trade_result.get('reason')}")
                            else:
                                logger.info(f"BUY signal: Position size calculated to zero or less. No trade placed.")

                elif action == 'SELL' and self.open_position and self.open_position['side'] == 'LONG':
                    # Signal to close existing LONG position (not SL/TP hit)
                    logger.info(f"SELL signal received to close existing LONG position at {current_timestamp}.")
                    close_trade_result = self.sim_exchange.execute_market_order(
                        self.symbol, 'SELL', self.open_position['units'],
                        current_price, # Exit at current bar's close
                        current_timestamp
                    )
                    logger.info(f"Manual position close result: {close_trade_result}")
                    if close_trade_result['status'] == 'FILLED':
                        self.open_position = None
                # elif action == 'SELL' and not self.open_position: # TODO: Open SHORT position
                #     pass

            # Log portfolio value at each step (optional, can be done by backtester if this is a live exec engine)
            # current_portfolio_value = self._get_current_equity(current_price)
            # logger.debug(f"End of bar {current_timestamp}: Portfolio Value ~{current_portfolio_value:.2f} {self.asset2}, Balances: {self.sim_exchange.balances}")

        logger.info(f"Live simulation for {self.symbol} finished.")
        logger.info(f"Final Balances: {self.sim_exchange.balances}")
        logger.info(f"Total Trades Executed: {len(self.sim_exchange.trade_history)}")


if __name__ == "__main__":
    logger.info("--- Starting Main Live Simulator Test ---")

    try:
        # Standard setup to get config and DB access (though DB not heavily used by simulator itself yet)
        config_file_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
        cm = ConfigManager(config_file_path=config_file_path)

        db_path_from_config = cm.get('database.path', "data/trading_system.db")
        db_full_path = db_path_from_config
        if not os.path.isabs(db_full_path): db_full_path = os.path.join(PROJECT_ROOT, db_full_path)
        db_m = DBManager(db_path=db_full_path)

        # Fetch feature data for BTCUSD (should be the 21-row set)
        symbol_to_simulate = "BTCUSD"
        features_df: Optional[pd.DataFrame] = None
        with db_m:
            # Need to query features table similar to how main_backtest_runner does
            table_name = f"features_{symbol_to_simulate.upper()}"
            query = f"SELECT * FROM {table_name} ORDER BY timestamp ASC;"
            pragma_query = f"PRAGMA table_info({table_name});"
            columns_info = db_m.fetch_data(pragma_query)
            if not columns_info:
                logger.critical(f"Could not fetch column info for table {table_name}. Exiting.")
                sys.exit(1)
            cols = [info[1] for info in columns_info]
            data = db_m.fetch_data(query)
            if not data:
                logger.critical(f"No feature data for {symbol_to_simulate}. Exiting.")
                sys.exit(1)

            features_df = pd.DataFrame(data, columns=cols)
            if 'timestamp' in features_df.columns:
                features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], unit='s')
                features_df.set_index('timestamp', inplace=True)
            else:
                logger.critical("Timestamp data missing or not set as index. Exiting.")
                sys.exit(1)

            # Convert numeric columns
            numeric_cols_to_try = ['open', 'high', 'low', 'close', 'volume'] + \
                                  [col for col in features_df.columns if 'sma_' in col or 'rsi_' in col or 'atr_' in col] # Add more as needed
            for col in numeric_cols_to_try:
                if col in features_df.columns:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')


        if features_df is None or features_df.empty:
            logger.critical(f"Failed to load feature data for {symbol_to_simulate}. Exiting.")
            sys.exit(1)

        logger.info(f"Loaded {len(features_df)} rows of feature data for {symbol_to_simulate} for simulation.")

        # Initialize Strategy (RandomStrategy for testing)
        # Seed for reproducibility in tests
        strategy = RandomStrategy(strategy_params={'random_seed': 42})
        logger.info(f"Using strategy: {strategy.strategy_name}")

        # Initialize and Run Simulator
        simulator = LiveSimulator(
            config_manager=cm,
            db_manager=db_m,
            strategy_instance=strategy,
            symbol=symbol_to_simulate,
            initial_ohlcv_data=features_df
        )
        simulator.run_simulation()

        print("\n--- Simulation Results ---")
        print("Final Balances:")
        for asset, balance in simulator.sim_exchange.balances.items():
            print(f"  {asset}: {balance:.8f}")

        print("\nTrade History:")
        if simulator.sim_exchange.trade_history:
            trades_df = pd.DataFrame(simulator.sim_exchange.trade_history)
            print(trades_df.to_string())
        else:
            print("No trades were executed during the simulation.")

    except FileNotFoundError as fnf_err:
        logger.critical(f"Essential file not found: {fnf_err}. Cannot run simulation.")
    except Exception as e:
        logger.critical(f"An error occurred during the simulation setup or run: {e}", exc_info=True)

    logger.info("--- Main Live Simulator Test Finished ---")
