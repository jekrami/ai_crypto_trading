import re
from typing import Dict, Optional, Union

# Configure logger if needed
# from utils.logger import setup_logger
# logger = setup_logger(__name__)

def parse_llama_response(response_text: Optional[str]) -> Optional[Dict[str, Union[str, None]]]:
    """
    Parses the text response from Ollama to extract trading signal and reasoning.

    Expected format in response_text:
    Signal: [BUY|SELL|HOLD]
    Reasoning: [Some text]

    :param response_text: The raw text string from the Ollama model.
    :return: A dictionary with "signal", "confidence" (None for now), and "reasoning",
             or None if a signal cannot be reliably extracted.
    """
    if not response_text or not response_text.strip():
        # logger.warning("Received empty or None response_text for parsing.")
        return None

    # Case-insensitive search for "Signal: BUY", "Signal: SELL", or "Signal: HOLD"
    signal_match = re.search(r"Signal:\s*(BUY|SELL|HOLD)", response_text, re.IGNORECASE)

    extracted_signal: Optional[str] = None
    if signal_match:
        extracted_signal = signal_match.group(1).upper()
    else:
        # Fallback: If "Signal: " prefix is missing, look for BUY, SELL, or HOLD as a whole word on a line,
        # possibly surrounded by minimal other text or as the primary content.
        # This is more lenient but could be prone to false positives if these words appear in other contexts.
        # Prioritize lines starting with the signal word.
        lines = response_text.strip().split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped == "BUY" or line_stripped == "SELL" or line_stripped == "HOLD":
                 extracted_signal = line_stripped
                 break
            # More lenient regex for a line containing just the signal:
            # Useful if the model just says "BUY." or "The signal is SELL."
            # This searches for the signal word potentially with some leading/trailing simple punctuation.
            # Example: "BUY.", "SELL", "Action: HOLD"
            # This regex is very basic, more sophisticated patterns might be needed for robustness.
            # For now, keep it simple. If the model is well-prompted, Signal: X is best.
            # Let's stick to the primary "Signal: X" or standalone word for now.
            # Add more robust fallback if needed based on observed model behavior.

    if not extracted_signal:
        # logger.warning(f"Could not extract a valid signal (BUY/SELL/HOLD) from response: '{response_text[:200]}...'")
        return None

    # Try to extract reasoning
    reasoning_match = re.search(r"Reasoning:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
    extracted_reasoning: Optional[str] = None
    if reasoning_match:
        extracted_reasoning = reasoning_match.group(1).strip()
    else:
        # Fallback for reasoning: if "Signal: ..." was found, take text after it as potential reasoning,
        # if it's not another "Signal:" line etc.
        # This is very basic. Good prompting is key for model to provide reasoning clearly.
        if signal_match: # Only if "Signal: X" was found
            text_after_signal = response_text[signal_match.end():].strip()
            # Avoid picking up a new "Signal:" line if model outputs multiple blocks
            if text_after_signal and not re.match(r"Signal:\s*(BUY|SELL|HOLD)", text_after_signal, re.IGNORECASE):
                extracted_reasoning = text_after_signal.split('\n')[0].strip() # Take first line after signal
                if not extracted_reasoning: # If first line was empty
                     # Take up to a certain number of characters or sentences if more complex logic is desired
                     extracted_reasoning = text_after_signal[:200].strip() + "..." if len(text_after_signal) > 200 else text_after_signal

        if not extracted_reasoning:
            # logger.info(f"Could not extract explicit reasoning from response: '{response_text[:200]}...'")
            pass # Reasoning is optional for the return dict if not found

    return {
        "signal": extracted_signal,
        "confidence": None,  # Placeholder as per requirements
        "reasoning": extracted_reasoning
    }

if __name__ == '__main__':
    print("--- Test: parse_llama_response ---")

    test_responses = [
        "Signal: BUY. Reasoning: The market shows strong upward momentum.",
        "Signal: SELL \nReasoning: Based on the indicators, a sell is advised.",
        "Signal: HOLD\n\nReasoning: Market is consolidating, hold position.",
        "signal: buy\nreasoning: looks good.",
        "SIGNAL: Sell. \n This is because the RSI is overbought.", # Reasoning not prefixed
        "Buy", # Fallback test (not ideal, depends on if we enable lenient signal search)
        "The signal is BUY. The reasoning is that the trend is up.", # Complex, needs better regex if common
        "Recommendation: BUY. Justification: Positive outlook.", # Not matching current regex
        "No clear signal. Market is choppy.", # No signal
        "Signal: UNKNOWN. Reasoning: Not enough data.", # Invalid signal
        None,
        "",
        "Signal: BUY. Reasoning: RSI is oversold and MACD is crossing bullishly.",
        "Signal: SELL. Reasoning: Price broke key support and BBands are expanding downwards.",
        "Signal: HOLD. Reasoning: Market is in a tight range, awaiting clearer breakout. ATR is low."
    ]

    expected_fallback_enabled = False # Controls if we expect "Buy" alone to work

    for i, resp_text in enumerate(test_responses):
        print(f"\nTest Case {i+1}:")
        print(f"Input Response Text: '{resp_text}'")
        parsed_result = parse_llama_response(resp_text)
        if parsed_result:
            print(f"  Signal: {parsed_result['signal']}")
            print(f"  Reasoning: {parsed_result['reasoning']}")
        else:
            # Specific check for the "Buy" test case if lenient parsing is off
            if resp_text == "Buy" and not expected_fallback_enabled:
                 print(f"  Parsed: {parsed_result} (Correctly no signal for standalone 'Buy' if lenient parsing is off)")
            else:
                 print(f"  Parsed: {parsed_result}")

    print("\n--- Specific Reasoning Fallback Test ---")
    reasoning_test_1 = "Signal: BUY. The indicators are all positive and news sentiment is good."
    # This tests if text after "Signal: BUY." is captured if "Reasoning:" prefix is missing.
    parsed_reasoning_test_1 = parse_llama_response(reasoning_test_1)
    print(f"Input: '{reasoning_test_1}'")
    if parsed_reasoning_test_1:
        print(f"  Signal: {parsed_reasoning_test_1['signal']}")
        print(f"  Reasoning: {parsed_reasoning_test_1['reasoning']}")
    else:
        print(f"  Parsed: {parsed_reasoning_test_1}")

    reasoning_test_2 = "Signal: SELL" # No reasoning provided
    parsed_reasoning_test_2 = parse_llama_response(reasoning_test_2)
    print(f"Input: '{reasoning_test_2}'")
    if parsed_reasoning_test_2:
        print(f"  Signal: {parsed_reasoning_test_2['signal']}")
        print(f"  Reasoning: {parsed_reasoning_test_2['reasoning']}") # Should be None
    else:
        print(f"  Parsed: {parsed_reasoning_test_2}")
