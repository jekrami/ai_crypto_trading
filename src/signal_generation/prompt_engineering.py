import pandas as pd
from typing import List, Optional, Union

# Configure logger if needed
# from utils.logger import setup_logger
# logger = setup_logger(__name__)

def format_market_context_prompt(symbol: str,
                                 ohlcv_features_df: pd.DataFrame,
                                 recent_news_list: Optional[List[str]] = None,
                                 num_recent_features: int = 5) -> str:
    """
    Formats a prompt string summarizing market context including recent features and news.

    :param symbol: The trading symbol (e.g., "BTCUSD").
    :param ohlcv_features_df: Pandas DataFrame with features, indexed by timestamp.
                               Assumes columns are 'open', 'high', 'low', 'close', 'volume', and other feature columns.
    :param recent_news_list: Optional list of recent news headlines or summaries.
    :param num_recent_features: Number of recent feature rows (time periods) to include.
    :return: A formatted string summarizing the market context.
    """
    if not isinstance(ohlcv_features_df.index, pd.DatetimeIndex):
        # logger.warning("OHLCV features DataFrame index is not a DatetimeIndex. Timestamp info might be lost.")
        # Attempt to convert if it's a 'timestamp' column, otherwise formatting might be poor.
        if 'timestamp' in ohlcv_features_df.columns:
            try:
                ohlcv_features_df = ohlcv_features_df.set_index(pd.to_datetime(ohlcv_features_df['timestamp']))
                # logger.info("Converted 'timestamp' column to DatetimeIndex for context formatting.")
            except Exception as e:
                # logger.error(f"Failed to convert 'timestamp' column to DatetimeIndex: {e}")
                pass # Proceed with current index if conversion fails

    # Select the last N rows for the prompt
    if num_recent_features > 0 and not ohlcv_features_df.empty:
        recent_features_df = ohlcv_features_df.tail(num_recent_features)
        # Convert DataFrame to string, consider formatting options for clarity
        # df_to_string = recent_features_df.to_string(float_format="%.2f") # Control float precision
        # For LLMs, a more structured format like CSV or selected columns might be better
        # For now, use a simple string representation.
        # Limit columns to key ones if too many features, or select representative ones.

        # Example: Select a subset of columns for brevity in the prompt, if df has many features
        cols_to_show = ['open', 'high', 'low', 'close', 'volume'] # Base OHLCV
        # Add some common TA indicators if they exist
        for ta_col in ['sma_20', 'sma_50', 'rsi_14', 'macd_12_26_9', 'bbu_20_2p0', 'bbl_20_2p0', 'atr_14']: # Example TA cols
            if ta_col in recent_features_df.columns:
                cols_to_show.append(ta_col)

        # Ensure selected columns exist to avoid KeyError
        cols_to_show = [col for col in cols_to_show if col in recent_features_df.columns]

        if not cols_to_show: # Fallback if no standard columns found
            df_to_string = recent_features_df.to_string(float_format="%.2f")
        else:
            df_to_string = recent_features_df[cols_to_show].to_string(float_format="%.2f")

        # Could also convert index to string for more readable dates if not already
        # df_to_string = recent_features_df[cols_to_show].rename_axis("datetime_utc").to_csv(float_format="%.2f")

    else:
        df_to_string = "No feature data available or num_recent_features is zero."

    context_parts = [f"Market context for {symbol}:"]
    context_parts.append(f"Recent Features (last {num_recent_features} periods, most recent last):")
    context_parts.append(df_to_string)

    if recent_news_list:
        context_parts.append("\nRecent News:")
        for i, news_item in enumerate(recent_news_list):
            context_parts.append(f"- [{i+1}] {news_item}")

    return "\n".join(context_parts)


def get_trading_signal_prompt(market_context_summary: str,
                              symbol: str,
                              trading_style_guidance: str = "Focus on short-term price action and technical patterns. Consider standard candlestick patterns, support/resistance levels, and momentum indicators visible in the provided data.") -> str:
    """
    Constructs the final prompt asking for a trading signal.

    :param market_context_summary: String containing the summary of market features and news.
    :param symbol: The trading symbol (e.g., "BTCUSD").
    :param trading_style_guidance: Specific instructions or style for the AI to adopt.
    :return: The full prompt string for the LLM.
    """

    prompt = f"""{market_context_summary}

Guidance: {trading_style_guidance}

Question: Based ONLY on the information provided above, what is the recommended trading action (BUY, SELL, or HOLD) for {symbol} for the next short-term period (e.g., next 1 to 4 hours)?
Your response MUST be structured as follows:
Signal: [BUY, SELL, or HOLD]
Reasoning: [A brief reasoning for your decision, limited to one or two sentences. Focus on key drivers from the data.]
"""
    # Removed "Confidence:" part from prompt as per subtask requirements (confidence is None for now)
    return prompt.strip()


if __name__ == '__main__':
    # Create a sample features DataFrame (mimicking output from feature_pipeline)
    sample_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00', '2023-01-01 12:00',
                                     '2023-01-01 13:00', '2023-01-01 14:00', '2023-01-01 15:00']),
        'open': [100, 101, 102, 103, 104, 105],
        'high': [105, 106, 107, 108, 109, 110],
        'low': [99, 100, 101, 102, 103, 104],
        'close': [101, 102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500],
        'sma_20': [98.0, 99.5, 100.5, 101.5, 102.5, 103.5],
        'rsi_14': [60.0, 62.0, 65.0, 68.0, 70.0, 72.0],
        'another_feature': [1,0,1,0,1,0] # Example other feature
    }
    sample_features_df = pd.DataFrame(sample_data).set_index('timestamp')

    print("--- Test: format_market_context_prompt ---")
    # Test without news
    context_no_news = format_market_context_prompt("BTCUSD", sample_features_df, num_recent_features=3)
    print("Context (no news, 3 features):")
    print(context_no_news)

    # Test with news
    sample_news = [
        "Bitcoin reaches new yearly high amid institutional interest.",
        "Ethereum upgrade 'Cancun' successfully deployed on testnet.",
        "Global markets show volatility due to inflation fears."
    ]
    context_with_news = format_market_context_prompt("ETHUSD", sample_features_df.tail(2), recent_news_list=sample_news, num_recent_features=2)
    print("\nContext (with news, 2 features):")
    print(context_with_news)

    print("\n--- Test: get_trading_signal_prompt ---")
    final_prompt = get_trading_signal_prompt(context_with_news, "ETHUSD")
    print("Final prompt to LLM:")
    print(final_prompt)

    custom_guidance = "You are a conservative trader. Prioritize capital preservation. Only recommend BUY or SELL if there is very strong evidence."
    final_prompt_custom = get_trading_signal_prompt(context_no_news, "BTCUSD", trading_style_guidance=custom_guidance)
    print("\nFinal prompt to LLM (custom guidance):")
    print(final_prompt_custom)

    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']).set_index(pd.to_datetime([]))
    context_empty_df = format_market_context_prompt("XYZUSD", empty_df)
    print("\nContext (empty features DF):")
    print(context_empty_df)
    final_prompt_empty = get_trading_signal_prompt(context_empty_df, "XYZUSD")
    print("\nFinal prompt (empty features DF):")
    print(final_prompt_empty)
