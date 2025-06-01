import pandas as pd
import pandas_ta as ta # type: ignore
from typing import List, Optional

# Initialize logger if needed (e.g., from src.utils.logger)
# For simplicity in this module, direct print/logging or rely on caller's logger.
# import logging
# logger = logging.getLogger(__name__) # Or get a pre-configured logger

def _validate_df_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """Helper to check for required columns."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # logger.error(f"Missing required columns: {missing_cols}")
        print(f"Error: Missing required columns for indicator calculation: {missing_cols}")
        return False
    return True

def add_sma(df: pd.DataFrame, length: int = 20, close_col: str = 'close', column_name_prefix: Optional[str] = 'sma') -> pd.DataFrame:
    """Adds Simple Moving Average (SMA) to the DataFrame."""
    if not _validate_df_columns(df, [close_col]):
        return df
    if df.empty:
        # print(f"Warning: DataFrame is empty. Skipping SMA calculation for length {length}.") # Optional: less verbose
        return df

    col_name = f"{column_name_prefix}_{length}" if column_name_prefix else f"sma_{length}"
    try:
        df.ta.sma(length=length, close=close_col, append=True, col_names=(col_name,))
        if col_name not in df.columns:
            df[col_name] = pd.NA
    except Exception as e:
        print(f"Error calculating SMA {col_name}: {e}")
        if col_name not in df.columns: df[col_name] = pd.NA
    return df

def add_ema(df: pd.DataFrame, length: int = 20, close_col: str = 'close', column_name_prefix: Optional[str] = 'ema') -> pd.DataFrame:
    """Adds Exponential Moving Average (EMA) to the DataFrame."""
    if not _validate_df_columns(df, [close_col]):
        return df
    if df.empty:
        # print(f"Warning: DataFrame is empty. Skipping EMA calculation for length {length}.")
        return df

    col_name = f"{column_name_prefix}_{length}" if column_name_prefix else f"ema_{length}"
    try:
        df.ta.ema(length=length, close=close_col, append=True, col_names=(col_name,))
        if col_name not in df.columns:
            df[col_name] = pd.NA
    except Exception as e:
        print(f"Error calculating EMA {col_name}: {e}")
        if col_name not in df.columns: df[col_name] = pd.NA
    return df

def add_rsi(df: pd.DataFrame, length: int = 14, close_col: str = 'close', column_name_prefix: Optional[str] = 'rsi') -> pd.DataFrame:
    """Adds Relative Strength Index (RSI) to the DataFrame."""
    if not _validate_df_columns(df, [close_col]):
        return df
    if df.empty:
        # print(f"Warning: DataFrame is empty. Skipping RSI calculation for length {length}.")
        return df

    col_name = f"{column_name_prefix}_{length}" if column_name_prefix else f"rsi_{length}"
    try:
        df.ta.rsi(length=length, close=close_col, append=True, col_names=(col_name,))
        if col_name not in df.columns:
            df[col_name] = pd.NA
    except Exception as e:
        print(f"Error calculating RSI {col_name}: {e}")
        if col_name not in df.columns: df[col_name] = pd.NA
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
             close_col: str = 'close',
             macd_col_prefix: Optional[str] = 'macd',      # For MACD line: MACD_12_26_9
             signal_col_prefix: Optional[str] = 'macds',   # For Signal line: MACDS_12_26_9
             hist_col_prefix: Optional[str] = 'macdh') -> pd.DataFrame:   # For Histogram: MACDH_12_26_9
    """Adds MACD, MACD Signal, and MACD Histogram to the DataFrame."""
    if not _validate_df_columns(df, [close_col]):
        return df
    if df.empty:
        # print(f"Warning: DataFrame is empty. Skipping MACD calculation for slow length {slow}.")
        return df

    base_name = f"{fast}_{slow}_{signal}"
    # Default pandas-ta names are like MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    # We allow customization via prefixes.
    macd_final_col = f"{macd_col_prefix}_{base_name}" if macd_col_prefix else f"MACD_{base_name}"
    hist_final_col = f"{hist_col_prefix}_{base_name}" if hist_col_prefix else f"MACDh_{base_name}"
    signal_final_col = f"{signal_col_prefix}_{base_name}" if signal_col_prefix else f"MACDs_{base_name}"

    # The col_names tuple for df.ta.macd should be (macd, hist, signal)
    col_names_ordered = (macd_final_col, hist_final_col, signal_final_col)

    try:
        df.ta.macd(fast=fast, slow=slow, signal=signal, close=close_col, append=True, col_names=col_names_ordered)
        for col in col_names_ordered: # Check each column
            if col not in df.columns: df[col] = pd.NA
    except Exception as e:
        print(f"Error calculating MACD for {base_name}: {e}")
        for col in col_names_ordered:
            if col not in df.columns: df[col] = pd.NA
    return df

def add_bollinger_bands(df: pd.DataFrame, length: int = 20, std: float = 2.0,
                        close_col: str = 'close',
                        bbl_col_prefix: Optional[str] = 'bbl',    # Lower Band
                        bbm_col_prefix: Optional[str] = 'bbm',    # Middle Band
                        bbu_col_prefix: Optional[str] = 'bbu',    # Upper Band
                        bba_col_prefix: Optional[str] = 'bbb',    # Bandwidth (pandas-ta default: BBB_length_std)
                        bbb_col_prefix: Optional[str] = 'bbp'     # Percent B (pandas-ta default: BBP_length_std)
                        ) -> pd.DataFrame:
    """Adds Bollinger Bands (Lower, Middle, Upper, Bandwidth, Percent) to the DataFrame."""
    if not _validate_df_columns(df, [close_col]):
        return df
    if df.empty:
        # print(f"Warning: DataFrame is empty. Skipping Bollinger Bands for length {length}.")
        return df

    base_name = f"{length}_{str(std).replace('.', 'p')}"
    # Standard pandas-ta names: BBL_l_s, BBM_l_s, BBU_l_s, BBB_l_s, BBP_l_s
    l_col = f"{bbl_col_prefix}_{base_name}" if bbl_col_prefix else f"BBL_{base_name}"
    m_col = f"{bbm_col_prefix}_{base_name}" if bbm_col_prefix else f"BBM_{base_name}"
    u_col = f"{bbu_col_prefix}_{base_name}" if bbu_col_prefix else f"BBU_{base_name}"
    band_col = f"{bba_col_prefix}_{base_name}" if bba_col_prefix else f"BBB_{base_name}"
    perc_col = f"{bbb_col_prefix}_{base_name}" if bbb_col_prefix else f"BBP_{base_name}"

    col_names_ordered = (l_col, m_col, u_col, band_col, perc_col)

    try:
        df.ta.bbands(length=length, std=std, close=close_col, append=True, col_names=col_names_ordered)
        for col in col_names_ordered:
            if col not in df.columns: df[col] = pd.NA
    except Exception as e:
        print(f"Error calculating Bollinger Bands for {base_name}: {e}")
        for col in col_names_ordered:
            if col not in df.columns: df[col] = pd.NA
    return df

def add_atr(df: pd.DataFrame, length: int = 14,
            high_col: str = 'high', low_col: str = 'low', close_col: str = 'close',
            column_name_prefix: Optional[str] = 'atr') -> pd.DataFrame:
    """Adds Average True Range (ATR) to the DataFrame."""
    required = [high_col, low_col, close_col]
    if not _validate_df_columns(df, required):
        return df
    if df.empty:
        # print(f"Warning: DataFrame is empty. Skipping ATR calculation for length {length}.")
        return df

    col_name = f"{column_name_prefix}_{length}" if column_name_prefix else f"atr_{length}"
    try:
        df.ta.atr(length=length, high=high_col, low=low_col, close=close_col, append=True, col_names=(col_name,))
        if col_name not in df.columns:
            df[col_name] = pd.NA
    except Exception as e:
        print(f"Error calculating ATR {col_name}: {e}")
        if col_name not in df.columns: df[col_name] = pd.NA
    return df

# Example Usage (can be run directly if this file is executed)
if __name__ == '__main__':
    # Create a sample DataFrame (replace with actual data loading)
    data = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'] * 10), # 50 days
        'open': [100, 101, 102, 103, 104] * 10,
        'high': [105, 106, 107, 108, 109] * 10,
        'low': [99, 100, 101, 102, 103] * 10,
        'close': [101, 102, 103, 104, 105] * 10,
        'volume': [1000, 1100, 1200, 1300, 1400] * 10
    }
    sample_df = pd.DataFrame(data)
    sample_df['close'] = sample_df['close'] + pd.Series(range(50)) * 0.1 # Add some variation
    sample_df['high'] = sample_df['high'] + pd.Series(range(50)) * 0.15
    sample_df['low'] = sample_df['low'] - pd.Series(range(50)) * 0.15


    print("Original DataFrame:")
    print(sample_df.head())

    # Add indicators
    sample_df = add_sma(sample_df.copy(), length=5) # Use copy to avoid modifying original in sequential calls for testing
    sample_df = add_ema(sample_df.copy(), length=5)
    sample_df = add_rsi(sample_df.copy(), length=5) # Short length for small sample
    sample_df = add_macd(sample_df.copy(), fast=5, slow=10, signal=3) # Short lengths for small sample
    sample_df = add_bollinger_bands(sample_df.copy(), length=5, std=1.5) # Short length
    sample_df = add_atr(sample_df.copy(), length=5)

    print("\nDataFrame with Indicators:")
    # print(sample_df.head())
    print(sample_df.tail(10).to_string()) # Print more rows to see indicator values

    # Test custom column names
    custom_df = pd.DataFrame(data) # Fresh df
    custom_df['close'] = custom_df['close'] + pd.Series(range(50)) * 0.12
    custom_df = add_sma(custom_df, length=7, column_name_prefix='my_sma')
    custom_df = add_rsi(custom_df, length=7, column_name_prefix='my_rsi')
    print("\nDataFrame with Custom Indicator Names:")
    print(custom_df.tail(10).to_string())

    # Test edge cases: empty df
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    empty_df = add_sma(empty_df, length=5)
    print("\nSMA on empty DF:", empty_df)

    # Test edge cases: df too short
    short_df = sample_df.head(3).copy()
    short_df = add_sma(short_df, length=5) # length 5 on 3 rows
    print("\nSMA on short DF (3 rows, length 5):")
    print(short_df)

    # Test missing columns
    missing_col_df = sample_df[['open', 'high', 'low', 'volume']].copy()
    missing_col_df = add_sma(missing_col_df, length=5) # 'close' is missing
    print("\nSMA on DF with missing 'close':")
    print(missing_col_df.head())
