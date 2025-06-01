import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .strategy import BaseStrategy # Assuming strategy.py is in the same directory
from utils.config_manager import ConfigManager # Added import

# Configure logger if needed
# from utils.logger import setup_logger
# logger = setup_logger(__name__)

class Backtester:
    def __init__(self,
                 strategy: BaseStrategy,
                 historical_data_df: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 transaction_fee_percent: float = 0.001, # 0.1%
                 slippage_percent: float = 0.0005,      # 0.05%
                 symbol_col: str = 'symbol', # If data contains multiple symbols, not used yet
                 price_col: str = 'close',   # Price column to use for execution
                 timestamp_col: str = 'timestamp', # Expected to be index, or a column
                 config_manager: Optional[ConfigManager] = None # Added to load risk configs
                ):

        self.strategy = strategy
        self.historical_data_df = historical_data_df.copy() # Work on a copy
        self.initial_capital = initial_capital
        self.transaction_fee_percent = transaction_fee_percent
        self.slippage_percent = slippage_percent

        self.symbol_col = symbol_col
        self.price_col = price_col
        self.timestamp_col = timestamp_col

        self.portfolio_history: List[Dict[str, Any]] = []
        self.trades_history: List[Dict[str, Any]] = []

        # Load risk management defaults from config or use hardcoded if CM not provided
        self.risk_per_trade_percent = 0.01 # Default
        self.default_sl_config = {'type': 'PERCENTAGE', 'value': 0.02} # Default
        self.default_tp_config = {'type': 'RISK_REWARD_RATIO', 'value': 1.5} # Default
        self.asset_price_increment = 0.01 # Default, e.g. for BTC/USD or ETH/USD

        if config_manager:
            self.risk_per_trade_percent = config_manager.get('risk_management_defaults.risk_per_trade_percent', self.risk_per_trade_percent)
            self.default_sl_config = config_manager.get('risk_management_defaults.default_stop_loss', self.default_sl_config)
            self.default_tp_config = config_manager.get('risk_management_defaults.default_take_profit', self.default_tp_config)
            # Example for loading asset-specific increment, assuming symbol is known at init or passed
            # current_symbol_for_config = "btc" # This needs to be dynamic if backtester handles multiple symbols
            # self.asset_price_increment = config_manager.get(f'risk_management_defaults.{current_symbol_for_config}_price_increment', self.asset_price_increment)

        # For simplicity, asset_price_increment is kept generic for now.
        # It might be better to pass it based on the specific asset being backtested.

        # Import risk management modules directly. Relies on PYTHONPATH including 'src'.
        from risk_management import position_sizer
        from risk_management import order_modifiers
        self.position_sizer = position_sizer
        self.order_modifiers = order_modifiers

        # Ensure data has a proper timestamp index if specified as index, or column exists
        if self.timestamp_col == self.historical_data_df.index.name:
            if not isinstance(self.historical_data_df.index, pd.DatetimeIndex):
                try:
                    self.historical_data_df.index = pd.to_datetime(self.historical_data_df.index)
                except Exception as e:
                    raise ValueError(f"Failed to convert index '{self.timestamp_col}' to DatetimeIndex: {e}")
        elif self.timestamp_col not in self.historical_data_df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in historical data.")
        else: # Timestamp is a column, try to convert it
             if not pd.api.types.is_datetime64_any_dtype(self.historical_data_df[self.timestamp_col]):
                try:
                    self.historical_data_df[self.timestamp_col] = pd.to_datetime(self.historical_data_df[self.timestamp_col])
                except Exception as e:
                    raise ValueError(f"Failed to convert column '{self.timestamp_col}' to datetime: {e}")

        # print(f"Backtester initialized for strategy: {self.strategy.strategy_name}")
        # print(f"Data shape: {self.historical_data_df.shape}, Initial Capital: {self.initial_capital}")

    def run_backtest(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Check moved to __init__ effectively by direct import. If not found, __init__ fails.
        # if not self.position_sizer or not self.order_modifiers:
        #     raise ImportError("Risk management modules (position_sizer, order_modifiers) are not available. Backtest cannot run.")

        self.portfolio_history = []
        self.trades_history = []

        cash = self.initial_capital
        current_position_units = 0.0
        open_position_details: Optional[Dict[str, Any]] = None # Store SL/TP with position

        for i in range(len(self.historical_data_df)):
            current_data_point = self.historical_data_df.iloc[i]
            current_timestamp = current_data_point.name if isinstance(self.historical_data_df.index, pd.DatetimeIndex) else current_data_point[self.timestamp_col]
            current_price = current_data_point[self.price_col] # Used for portfolio valuation and as potential entry/exit

            exit_reason = None # To mark if SL/TP was hit

            # 1. Check for SL/TP if position is open
            if open_position_details:
                pos_side = open_position_details['side']
                sl_price = open_position_details['stop_loss_price']
                tp_price = open_position_details['take_profit_price']

                if pos_side == "LONG":
                    if current_data_point['low'] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "STOP_LOSS"
                    elif tp_price and current_data_point['high'] >= tp_price:
                        exit_price = tp_price
                        exit_reason = "TAKE_PROFIT"
                elif pos_side == "SHORT":
                    # if current_data_point['high'] >= sl_price: exit_reason = "STOP_LOSS"; exit_price = sl_price
                    # elif tp_price and current_data_point['low'] <= tp_price: exit_reason = "TAKE_PROFIT"; exit_price = tp_price
                    pass

                if exit_reason:
                    execution_price_sl_tp = exit_price * (1 - self.slippage_percent if pos_side == "LONG" else 1 + self.slippage_percent)
                    revenue = current_position_units * execution_price_sl_tp
                    fee = revenue * self.transaction_fee_percent
                    cash += (revenue - fee)

                    self.trades_history.append({
                        'timestamp': current_timestamp, 'action': f'EXIT_{pos_side}_{exit_reason}',
                        'price': execution_price_sl_tp, 'units': current_position_units,
                        'fee': fee, 'reasoning': exit_reason, 'signal_type': open_position_details['signal_type']
                    })
                    current_position_units = 0.0
                    open_position_details = None


            # 2. Generate signal and act if no position was closed by SL/TP in this bar
            if not exit_reason and not open_position_details:
                historical_slice_for_strategy = self.historical_data_df.iloc[:i+1]
                signal_decision = self.strategy.generate_signal(current_data_point, historical_slice_for_strategy)

                if signal_decision and signal_decision.get('action') != 'HOLD':
                    action = signal_decision['action']
                    reasoning = signal_decision.get('reasoning', 'N/A')
                    signal_type = signal_decision.get('signal_type', self.strategy.strategy_name)

                    entry_price = current_price

                    if action == 'BUY':
                        position_side = "LONG"
                        sl_price = self.order_modifiers.calculate_stop_loss(
                            entry_price, position_side, self.default_sl_config,
                            atr_value=current_data_point.get('atr_14'),
                            recent_low=current_data_point.get('low')
                        )
                        if sl_price is None:
                            pass
                        else:
                            tp_price = self.order_modifiers.calculate_take_profit(
                                entry_price, sl_price, position_side, self.default_tp_config,
                                atr_value=current_data_point.get('atr_14')
                            )
                            sl_distance = abs(entry_price - sl_price)
                            if sl_distance < 1e-6 :
                                pass
                            else:
                                units_to_trade = self.position_sizer.calculate_position_size(
                                    cash,
                                    self.risk_per_trade_percent,
                                    sl_distance, entry_price, self.asset_price_increment
                                )
                                if units_to_trade > 0:
                                    execution_price_entry = entry_price * (1 + self.slippage_percent)
                                    cost = units_to_trade * execution_price_entry
                                    fee = cost * self.transaction_fee_percent
                                    if cash >= cost + fee:
                                        cash -= (cost + fee)
                                        current_position_units = units_to_trade
                                        open_position_details = {
                                            'side': position_side, 'entry_price': execution_price_entry,
                                            'units': units_to_trade, 'stop_loss_price': sl_price,
                                            'take_profit_price': tp_price, 'signal_type': signal_type
                                        }
                                        self.trades_history.append({
                                            'timestamp': current_timestamp, 'action': 'BUY_ENTRY',
                                            'price': execution_price_entry, 'units': units_to_trade,
                                            'fee': fee, 'reasoning': reasoning, 'signal_type': signal_type,
                                            'sl_price': sl_price, 'tp_price': tp_price
                                        })

                    # elif action == 'SELL': # Attempt to open SHORT (not fully implemented for brevity)
                    #     pass


            # 3. Update portfolio value for this period
            current_asset_value = current_position_units * current_price
            portfolio_value = cash + current_asset_value

            self.portfolio_history.append({
                'timestamp': current_timestamp, 'cash': cash,
                'position_units': current_position_units, 'asset_value': current_asset_value,
                'portfolio_value': portfolio_value, 'sl_hit': (exit_reason == "STOP_LOSS"),
                'tp_hit': (exit_reason == "TAKE_PROFIT"),
                'signal_action_taken': signal_decision.get('action') if signal_decision and open_position_details and not exit_reason else ('HOLD' if not exit_reason else exit_reason)
            })

        return self.portfolio_history, self.trades_history

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        if not self.portfolio_history:
            # print("Portfolio history is empty. Run backtest first or no trades were made.")
            return {"error": "Portfolio history empty."}

        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df.set_index('timestamp', inplace=True)

        final_portfolio_value = portfolio_df['portfolio_value'].iloc[-1]
        total_pnl = final_portfolio_value - self.initial_capital
        total_pnl_percent = (total_pnl / self.initial_capital) * 100

        # Max Drawdown
        portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = portfolio_df['portfolio_value'] - portfolio_df['peak']
        portfolio_df['drawdown_percent'] = (portfolio_df['drawdown'] / portfolio_df['peak']) * 100 # Drawdown as % of peak
        max_drawdown_abs = portfolio_df['drawdown'].min() # Most negative drawdown value
        max_drawdown_percent = portfolio_df['drawdown_percent'].min() if not portfolio_df['drawdown_percent'].empty else 0


        # Number of Trades
        num_trades = len(self.trades_history)

        # Basic P&L from trades (sum of P&L per trade) - more complex if not exiting full positions
        # For now, total P&L from portfolio value is simpler and more robust.

        metrics = {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": final_portfolio_value,
            "total_pnl_absolute": total_pnl,
            "total_pnl_percentage": total_pnl_percent,
            "max_drawdown_absolute": max_drawdown_abs,
            "max_drawdown_percentage": max_drawdown_percent,
            "number_of_trades": num_trades,
            # "sharpe_ratio": None, # Placeholder
            # "win_rate": None, # Placeholder
        }
        # print(f"Performance Metrics: {metrics}")
        return metrics

if __name__ == '__main__':
    print("--- Testing Backtester ---")

    # Create dummy historical data (ensure it has columns strategy might need, e.g., SMAs)
    sma_short_window = 2 # For SmaCrossStrategy test
    sma_long_window = 3  # For SmaCrossStrategy test
    sma_short_col_name = f"sma_{sma_short_window}"
    sma_long_col_name = f"sma_{sma_long_window}"

    data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00', '2023-01-01 12:00',
                                     '2023-01-01 13:00', '2023-01-01 14:00', '2023-01-01 15:00']),
        'open':  [100, 101, 102, 103, 104, 105],
        'high':  [102, 103, 104, 105, 106, 107],
        'low':   [99,  100, 101, 102, 103, 104],
        'close': [101, 102, 100, 104, 105, 103], # Some price movement
        'volume':[1000,1000,1000,1000,1000,1000],
        # Dummy SMAs for SmaCrossStrategy test
        sma_short_col_name: [100.0, 101.5, 101.0, 102.0, 104.5, 104.0], # Short SMA
        sma_long_col_name:  [100.0, 100.5, 101.0, 101.33,102.33,104.0] # Long SMA
        # SMA_2: 100, 101.5, 101, 102, 104.5, 104
        # SMA_3: 100, 100.5, 101, 101.33, 102.33, 104
        # P S2<=S3, C S2>S3: BUY  (101.5 > 100.5 at T1)
        # P S2>=S3, C S2<S3: SELL (101 < 101 at T2, but 101.5 > 100.5 prev implies short was above long)
        # P S2<=S3, C S2>S3: BUY  (102 > 101.33 at T3)
        # P S2>=S3, C S2<S3: SELL (104 == 104 at T5, but 104.5 > 102.33 prev implies short was above long) - needs careful cross check
    }
    sample_features_df = pd.DataFrame(data).set_index('timestamp')

    print("Sample Features DataFrame for Backtest:")
    print(sample_features_df)

    # Test with RandomStrategy
    print("\n-- Backtest with RandomStrategy --")
    random_strat_bt = RandomStrategy(strategy_params={'random_seed': 42})
    backtester_random = Backtester(strategy=random_strat_bt, historical_data_df=sample_features_df.copy())
    portfolio_hist_rand, trades_hist_rand = backtester_random.run_backtest()
    metrics_rand = backtester_random.calculate_performance_metrics()

    print("Random Strategy Metrics:", metrics_rand)
    # print("Random Portfolio History (last 5):", pd.DataFrame(portfolio_hist_rand).tail())
    # print("Random Trades History:", pd.DataFrame(trades_hist_rand))


    # Test with SmaCrossStrategy
    print("\n-- Backtest with SmaCrossStrategy --")
    # Ensure sma_short_col and sma_long_col in SmaCrossStrategy match column names here
    sma_cross_strat_bt = SmaCrossStrategy(short_window=sma_short_window, long_window=sma_long_window)
    backtester_sma = Backtester(strategy=sma_cross_strat_bt, historical_data_df=sample_features_df.copy())
    portfolio_hist_sma, trades_hist_sma = backtester_sma.run_backtest()
    metrics_sma = backtester_sma.calculate_performance_metrics()

    print("SMA Cross Strategy Metrics:", metrics_sma)
    # print("SMA Portfolio History (last 5):", pd.DataFrame(portfolio_hist_sma).tail())
    # print("SMA Trades History:", pd.DataFrame(trades_hist_sma))

    if not trades_hist_sma:
        print("NOTE: SMA Cross Strategy did not produce any trades with this dummy data. This might be due to data length or SMA values not crossing as expected by logic.")
    else:
        print("SMA Trades History:", pd.DataFrame(trades_hist_sma))
        print("SMA Portfolio History (last 5 entries):")
        print(pd.DataFrame(portfolio_hist_sma).tail().to_string())
