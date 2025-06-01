import numpy as np
from typing import Dict, Any, Optional, Literal

# Configure logger if needed
# from utils.logger import setup_logger
# logger = setup_logger(__name__)

PositionSide = Literal["LONG", "SHORT"]

def calculate_stop_loss(
    entry_price: float,
    position_side: PositionSide,
    stop_loss_config: Dict[str, Any],
    atr_value: Optional[float] = None,
    recent_low: Optional[float] = None,  # For a long position, could be recent swing low
    recent_high: Optional[float] = None  # For a short position, could be recent swing high
) -> Optional[float]:
    """
    Calculates the stop-loss price based on the configuration.

    :param entry_price: The price at which the position was/will be entered.
    :param position_side: "LONG" or "SHORT".
    :param stop_loss_config: Dictionary defining SL type and value.
                             Examples: {'type': 'PERCENTAGE', 'value': 0.02} (2%)
                                       {'type': 'ATR', 'value': 1.5} (1.5x ATR)
                                       {'type': 'FIXED_VALUE', 'value': 50.0} ($50 price distance)
                                       {'type': 'RECENT_SWING', 'offset_percent': 0.001} (0.1% beyond swing)
    :param atr_value: Current ATR value, required if type is 'ATR'.
    :param recent_low: Recent low price, potentially used if type is 'RECENT_SWING' for longs.
    :param recent_high: Recent high price, potentially used if type is 'RECENT_SWING' for shorts.
    :return: Calculated stop-loss price, or None if calculation is not possible.
    """
    if entry_price <= 0:
        # print("Warning: Entry price must be positive for SL calculation.") # Or log
        return None

    sl_type = stop_loss_config.get('type', '').upper()
    sl_value = stop_loss_config.get('value')

    if sl_value is None:
        # print(f"Warning: No 'value' provided in stop_loss_config for type {sl_type}.")
        return None

    stop_loss_price: Optional[float] = None

    if sl_type == 'PERCENTAGE':
        if not (0 < sl_value < 1.0): # Percentage value should be like 0.02 for 2%
            # print(f"Warning: Percentage SL value ({sl_value}) should be a fraction (e.g., 0.02 for 2%).")
            return None
        if position_side == "LONG":
            stop_loss_price = entry_price * (1 - sl_value)
        else: # SHORT
            stop_loss_price = entry_price * (1 + sl_value)

    elif sl_type == 'ATR':
        if atr_value is None or atr_value <= 0:
            # print("Warning: ATR type SL requires a positive atr_value.")
            return None
        if sl_value <= 0: # Multiplier for ATR
            # print("Warning: ATR multiplier 'value' must be positive for SL calculation.")
            return None

        distance = atr_value * sl_value
        if position_side == "LONG":
            stop_loss_price = entry_price - distance
        else: # SHORT
            stop_loss_price = entry_price + distance

    elif sl_type == 'FIXED_VALUE': # Fixed price distance
        if sl_value <= 0:
            # print("Warning: FIXED_VALUE SL 'value' (distance) must be positive.")
            return None
        if position_side == "LONG":
            stop_loss_price = entry_price - sl_value
        else: # SHORT
            stop_loss_price = entry_price + sl_value

    elif sl_type == 'RECENT_SWING':
        offset_percent = stop_loss_config.get('offset_percent', 0.001) # Default 0.1% buffer
        if position_side == "LONG":
            if recent_low is None or recent_low >= entry_price:
                # print("Warning: Valid recent_low below entry_price required for RECENT_SWING SL on LONG.")
                return None # Fallback or use another SL type if this happens
            stop_loss_price = recent_low * (1 - offset_percent)
        else: # SHORT
            if recent_high is None or recent_high <= entry_price:
                # print("Warning: Valid recent_high above entry_price required for RECENT_SWING SL on SHORT.")
                return None
            stop_loss_price = recent_high * (1 + offset_percent)
    else:
        # print(f"Warning: Unknown stop-loss type: {sl_type}")
        return None

    # Ensure SL price is not negative or zero, and respects position side (e.g. SL for long must be < entry)
    if stop_loss_price is not None and stop_loss_price <= 0:
        # print(f"Warning: Calculated SL price ({stop_loss_price}) is zero or negative. Invalid SL.")
        return None
    if position_side == "LONG" and stop_loss_price is not None and stop_loss_price >= entry_price:
        # print(f"Warning: Calculated SL price ({stop_loss_price}) for LONG is not below entry price ({entry_price}). Invalid SL.")
        return None
    if position_side == "SHORT" and stop_loss_price is not None and stop_loss_price <= entry_price:
        # print(f"Warning: Calculated SL price ({stop_loss_price}) for SHORT is not above entry price ({entry_price}). Invalid SL.")
        return None

    return stop_loss_price


def calculate_take_profit(
    entry_price: float,
    stop_loss_price: Optional[float], # Required if TP is based on Risk/Reward
    position_side: PositionSide,
    take_profit_config: Dict[str, Any],
    atr_value: Optional[float] = None # For potential ATR-based TP
) -> Optional[float]:
    """
    Calculates the take-profit price.

    :param entry_price: The price at which the position was/will be entered.
    :param stop_loss_price: The calculated stop-loss price. Required for 'RISK_REWARD_RATIO'.
    :param position_side: "LONG" or "SHORT".
    :param take_profit_config: Dictionary defining TP type and value.
                               Examples: {'type': 'RISK_REWARD_RATIO', 'value': 1.5} (1.5:1 R:R)
                                         {'type': 'PERCENTAGE', 'value': 0.03} (3% from entry)
                                         {'type': 'ATR_MULTIPLE_FROM_ENTRY', 'value': 3.0} (3x ATR from entry)
    :param atr_value: Current ATR value, required if type is 'ATR_MULTIPLE_FROM_ENTRY'.
    :return: Calculated take-profit price, or None.
    """
    if entry_price <= 0:
        return None

    tp_type = take_profit_config.get('type', '').upper()
    tp_value = take_profit_config.get('value')

    if tp_value is None or tp_value <= 0: # Value must be positive for all types
        # print(f"Warning: No 'value' or non-positive 'value' provided in take_profit_config for type {tp_type}.")
        return None

    take_profit_price: Optional[float] = None

    if tp_type == 'PERCENTAGE':
        if not (0 < tp_value < 1.0): # e.g. 0.03 for 3%
             # print(f"Warning: Percentage TP value ({tp_value}) should be a fraction (e.g., 0.03 for 3%).")
            return None
        if position_side == "LONG":
            take_profit_price = entry_price * (1 + tp_value)
        else: # SHORT
            take_profit_price = entry_price * (1 - tp_value)

    elif tp_type == 'RISK_REWARD_RATIO':
        if stop_loss_price is None:
            # print("Warning: Stop-loss price must be provided for RISK_REWARD_RATIO based TP.")
            return None

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit <= 1e-9: # Avoid division by zero or tiny risk
            # print("Warning: Risk per unit (entry - SL) is zero or too small for R:R TP.")
            return None

        reward_per_unit = risk_per_unit * tp_value # tp_value is the R:R multiplier, e.g., 1.5

        if position_side == "LONG":
            take_profit_price = entry_price + reward_per_unit
        else: # SHORT
            take_profit_price = entry_price - reward_per_unit

    elif tp_type == 'ATR_MULTIPLE_FROM_ENTRY': # Similar to ATR SL, but for TP
        if atr_value is None or atr_value <= 0:
            # print("Warning: ATR type TP requires a positive atr_value.")
            return None

        distance = atr_value * tp_value
        if position_side == "LONG":
            take_profit_price = entry_price + distance
        else: # SHORT
            take_profit_price = entry_price - distance
    else:
        # print(f"Warning: Unknown take-profit type: {tp_type}")
        return None

    # Ensure TP price is not negative or zero, and respects position side
    if take_profit_price is not None and take_profit_price <= 0:
        # print(f"Warning: Calculated TP price ({take_profit_price}) is zero or negative. Invalid TP.")
        return None
    if position_side == "LONG" and take_profit_price is not None and take_profit_price <= entry_price:
        # print(f"Warning: Calculated TP price ({take_profit_price}) for LONG is not above entry price ({entry_price}). Invalid TP.")
        return None
    if position_side == "SHORT" and take_profit_price is not None and take_profit_price >= entry_price:
        # print(f"Warning: Calculated TP price ({take_profit_price}) for SHORT is not below entry price ({entry_price}). Invalid TP.")
        return None

    return take_profit_price


if __name__ == '__main__':
    print("--- Testing Order Modifiers ---")
    entry = 100.0
    atr = 2.0
    recent_l = 95.0
    recent_h = 105.0

    # Test Stop Loss
    print("\n-- Stop Loss Tests --")
    sl_conf_pct = {'type': 'PERCENTAGE', 'value': 0.02} # 2% SL
    sl_long_pct = calculate_stop_loss(entry, "LONG", sl_conf_pct) # Expected: 100 * (1-0.02) = 98
    sl_short_pct = calculate_stop_loss(entry, "SHORT", sl_conf_pct) # Expected: 100 * (1+0.02) = 102
    print(f"Percentage SL Long (2%): {sl_long_pct} (Expected: 98)")
    assert abs(sl_long_pct - 98.0) < 1e-9
    print(f"Percentage SL Short (2%): {sl_short_pct} (Expected: 102)")
    assert abs(sl_short_pct - 102.0) < 1e-9

    sl_conf_atr = {'type': 'ATR', 'value': 1.5} # 1.5 * ATR SL
    sl_long_atr = calculate_stop_loss(entry, "LONG", sl_conf_atr, atr_value=atr) # Expected: 100 - 1.5*2 = 97
    sl_short_atr = calculate_stop_loss(entry, "SHORT", sl_conf_atr, atr_value=atr) # Expected: 100 + 1.5*2 = 103
    print(f"ATR SL Long (1.5xATR={atr}): {sl_long_atr} (Expected: 97)")
    assert abs(sl_long_atr - 97.0) < 1e-9
    print(f"ATR SL Short (1.5xATR={atr}): {sl_short_atr} (Expected: 103)")
    assert abs(sl_short_atr - 103.0) < 1e-9

    sl_conf_swing = {'type': 'RECENT_SWING', 'offset_percent': 0.01} # 1% beyond swing
    sl_long_swing = calculate_stop_loss(entry, "LONG", sl_conf_swing, recent_low=recent_l) # Expected: 95 * (1-0.01) = 94.05
    sl_short_swing = calculate_stop_loss(entry, "SHORT", sl_conf_swing, recent_high=recent_h) # Expected: 105 * (1+0.01) = 106.05
    print(f"Recent Swing SL Long (1% offset from {recent_l}): {sl_long_swing} (Expected: 94.05)")
    assert abs(sl_long_swing - 94.05) < 1e-9
    print(f"Recent Swing SL Short (1% offset from {recent_h}): {sl_short_swing} (Expected: 106.05)")
    assert abs(sl_short_swing - 106.05) < 1e-9


    # Test Take Profit
    print("\n-- Take Profit Tests --")
    tp_conf_pct = {'type': 'PERCENTAGE', 'value': 0.03} # 3% TP
    tp_long_pct = calculate_take_profit(entry, None, "LONG", tp_conf_pct) # Expected: 100 * (1+0.03) = 103
    tp_short_pct = calculate_take_profit(entry, None, "SHORT", tp_conf_pct) # Expected: 100 * (1-0.03) = 97
    print(f"Percentage TP Long (3%): {tp_long_pct} (Expected: 103)")
    assert abs(tp_long_pct - 103.0) < 1e-9
    print(f"Percentage TP Short (3%): {tp_short_pct} (Expected: 97)")
    assert abs(tp_short_pct - 97.0) < 1e-9

    # R:R TP needs a SL price. Let's use sl_long_atr = 97 (risk = 3)
    tp_conf_rr = {'type': 'RISK_REWARD_RATIO', 'value': 2.0} # 2:1 R:R
    tp_long_rr = calculate_take_profit(entry, sl_long_atr, "LONG", tp_conf_rr) # Risk = 100-97=3. Reward = 3*2=6. TP = 100+6=106
    print(f"R:R TP Long (2:1, SL=97, Risk=3): {tp_long_rr} (Expected: 106)")
    assert abs(tp_long_rr - 106.0) < 1e-9

    # R:R TP for short, use sl_short_atr = 103 (risk = 3)
    tp_short_rr = calculate_take_profit(entry, sl_short_atr, "SHORT", tp_conf_rr) # Risk = 103-100=3. Reward = 3*2=6. TP = 100-6=94
    print(f"R:R TP Short (2:1, SL=103, Risk=3): {tp_short_rr} (Expected: 94)")
    assert abs(tp_short_rr - 94.0) < 1e-9

    print("\nAll tests seem to pass based on assertions.")
