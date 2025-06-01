import numpy as np
from typing import Optional

# Configure logger if needed
# from utils.logger import setup_logger
# logger = setup_logger(__name__)

def calculate_position_size(
    account_equity: float,
    risk_per_trade_percent: float,
    stop_loss_distance_value: float,
    entry_price: float, # Added for context, though not directly used in simplest form
    asset_price_increment: float = 0.01
) -> float:
    """
    Calculates the position size in units of the asset.

    :param account_equity: Current total equity of the account.
    :param risk_per_trade_percent: Maximum percentage of equity to risk on this trade (e.g., 0.01 for 1%).
    :param stop_loss_distance_value: The monetary distance from entry price to stop-loss price per unit of asset.
                                     This should always be a positive value.
    :param entry_price: The price at which the asset is to be bought or sold.
    :param asset_price_increment: Smallest tradable unit of the asset's price (e.g., 0.01 for BTC/USD).
                                  Used for potential rounding of position size, though not strictly implemented here yet.
    :return: Number of units of the asset to trade. Returns 0.0 if inputs are invalid or risk is too high.
    """

    if account_equity <= 0:
        # print("Warning: Account equity is zero or negative. Cannot calculate position size.") # Or log
        return 0.0

    if not (0 < risk_per_trade_percent < 1.0): # Should be a fraction, e.g., 0.01 for 1%
        # print(f"Warning: risk_per_trade_percent ({risk_per_trade_percent}) should be between 0 and 1 (exclusive). Cannot calculate position size.")
        return 0.0

    if stop_loss_distance_value <= 0:
        # print(f"Warning: Stop-loss distance ({stop_loss_distance_value}) must be positive. Cannot calculate position size.")
        return 0.0

    if entry_price <= 0: # Sanity check for entry price
        # print(f"Warning: Entry price ({entry_price}) must be positive. Cannot calculate position size.")
        return 0.0

    # Calculate the total monetary amount to risk on this trade
    amount_to_risk = account_equity * risk_per_trade_percent

    # Calculate position size in units of the asset
    # This is how many units you can buy/sell such that if the price moves by stop_loss_distance_value, you lose amount_to_risk
    position_size_units = amount_to_risk / stop_loss_distance_value

    # Optional: Adjust for minimum tradable units or lot sizes.
    # For example, if asset can only be traded in multiples of `asset_price_increment` for its *value*,
    # or if units must be integers (e.g. shares).
    # For crypto, units can often be fractional.
    # If asset_price_increment related to unit size (e.g. 1 share, 0.001 BTC):
    # position_size_units = np.floor(position_size_units / asset_price_increment) * asset_price_increment
    # The current `asset_price_increment` seems more related to price ticks than unit sizes.
    # For now, we return the float value. Refinement depends on broker/exchange rules.
    # A common rounding for units might be to a certain number of decimal places.
    # E.g., round down to 8 decimal places for BTC:
    # position_size_units = np.floor(position_size_units * 1e8) / 1e8

    if position_size_units <= 0: # Should not happen if inputs are valid, but as a safeguard
        return 0.0

    # Consider if the cost of this position size exceeds available equity (for leveraged scenarios, this is more complex)
    # For simple spot trading, cost = position_size_units * entry_price.
    # This function primarily focuses on risk, not affordability directly. Affordability is checked by backtester.

    return position_size_units

if __name__ == '__main__':
    print("--- Testing calculate_position_size ---")

    equity = 10000.0
    risk_pct = 0.01  # 1% risk per trade

    # Example 1: Standard scenario
    entry_1 = 20000.0
    sl_distance_1 = 200.0 # e.g. SL at 19800 for a long, or 20200 for a short
    print(f"\nTest 1: Equity=${equity}, Risk={risk_pct*100}%, Entry=${entry_1}, SL Distance=${sl_distance_1}")
    units_1 = calculate_position_size(equity, risk_pct, sl_distance_1, entry_1)
    print(f"  Calculated units: {units_1:.8f}") # Amount to risk = 100. Units = 100 / 200 = 0.5 units
    assert abs(units_1 - 0.5) < 1e-9

    # Example 2: Tighter stop loss
    sl_distance_2 = 50.0
    print(f"\nTest 2: Equity=${equity}, Risk={risk_pct*100}%, Entry=${entry_1}, SL Distance=${sl_distance_2}")
    units_2 = calculate_position_size(equity, risk_pct, sl_distance_2, entry_1)
    print(f"  Calculated units: {units_2:.8f}") # Amount to risk = 100. Units = 100 / 50 = 2.0 units
    assert abs(units_2 - 2.0) < 1e-9

    # Example 3: Invalid SL distance (zero)
    sl_distance_3 = 0.0
    print(f"\nTest 3: Equity=${equity}, Risk={risk_pct*100}%, Entry=${entry_1}, SL Distance=${sl_distance_3}")
    units_3 = calculate_position_size(equity, risk_pct, sl_distance_3, entry_1)
    print(f"  Calculated units: {units_3:.8f}") # Should be 0
    assert abs(units_3 - 0.0) < 1e-9

    # Example 4: Invalid risk_pct (too high)
    risk_pct_invalid = 1.5
    print(f"\nTest 4: Equity=${equity}, Risk={risk_pct_invalid*100}%, Entry=${entry_1}, SL Distance=${sl_distance_1}")
    units_4 = calculate_position_size(equity, risk_pct_invalid, sl_distance_1, entry_1)
    print(f"  Calculated units: {units_4:.8f}") # Should be 0
    assert abs(units_4 - 0.0) < 1e-9

    # Example 5: Negative equity
    equity_invalid = -100
    print(f"\nTest 5: Equity=${equity_invalid}, Risk={risk_pct*100}%, Entry=${entry_1}, SL Distance=${sl_distance_1}")
    units_5 = calculate_position_size(equity_invalid, risk_pct, sl_distance_1, entry_1)
    print(f"  Calculated units: {units_5:.8f}") # Should be 0
    assert abs(units_5 - 0.0) < 1e-9

    # Example 6: Small equity, standard risk leading to small position size
    equity_small = 100.0
    sl_distance_6 = 10.0 # e.g. for a lower priced asset
    entry_6 = 50.0
    print(f"\nTest 6: Equity=${equity_small}, Risk={risk_pct*100}%, Entry=${entry_6}, SL Distance=${sl_distance_6}")
    units_6 = calculate_position_size(equity_small, risk_pct, sl_distance_6, entry_6)
    print(f"  Calculated units: {units_6:.8f}") # Amount to risk = 1. Units = 1 / 10 = 0.1 units
    assert abs(units_6 - 0.1) < 1e-9

    print("\nAll tests seem to pass based on assertions.")
