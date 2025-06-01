import sys
import os

# Ensure 'src' is in the Python path for direct imports of modules
# This assumes orchestrator_example.py is in the project root, and 'src' is a subdirectory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR_PATH not in sys.path:
    sys.path.append(SRC_DIR_PATH)

# Attempt to import main functions from modules
# These modules might need slight adjustments to make their main logic callable via a main() function
# if they currently only use `if __name__ == "__main__":` for direct execution.

try:
    from data_ingestion import market_data_importer
except ImportError:
    print("Could not import market_data_importer. Ensure it has a callable main function and src is in PYTHONPATH.")
    market_data_importer = None

try:
    from data_ingestion import news_importer
except ImportError:
    print("Could not import news_importer. Ensure it has a callable main function.")
    news_importer = None

try:
    from feature_engineering import feature_pipeline
except ImportError:
    print("Could not import feature_pipeline. Ensure it has a callable main function.")
    feature_pipeline = None

try:
    from signal_generation import historical_signal_generator
except ImportError:
    print("Could not import historical_signal_generator. Ensure it has a callable main function.")
    historical_signal_generator = None

try:
    from strategy_backtesting import main_backtest_runner
except ImportError:
    print("Could not import main_backtest_runner. Ensure it has a callable main function.")
    main_backtest_runner = None


def run_workflow_a(symbols=['BTCUSD', 'ETHUSD']):
    """
    Executes Workflow A: Initial Data Setup & Full Historical Backtest Preparation.
    """
    print("\n--- Starting Workflow A: Initial Data Setup & Full Historical Backtest Preparation ---")

    print("\nStep 1: Importing market data (OHLCV)...")
    if market_data_importer and hasattr(market_data_importer, 'main'):
        try:
            market_data_importer.main()
            print("Market data import finished.")
        except Exception as e:
            print(f"Error running market_data_importer.main(): {e}")
    else:
        print("Skipping market data import (module or main function not available). Run manually: python src/data_ingestion/market_data_importer.py")

    print("\nStep 2: Importing news data...")
    if news_importer and hasattr(news_importer, 'main'):
        try:
            news_importer.main()
            print("News import finished.")
        except Exception as e:
            print(f"Error running news_importer.main(): {e}")
    else:
        print("Skipping news import (module or main function not available). Run manually: python src/data_ingestion/news_importer.py")

    print("\nStep 3: Generating features...")
    if feature_pipeline and hasattr(feature_pipeline, 'run_feature_engineering'): # Assuming run_feature_engineering is the main logic
        try:
            # The feature_pipeline's main execution is run_feature_engineering(symbol_list)
             # Its main() wrapper takes `symbols`
            feature_pipeline.main(symbols=symbols) # Corrected: use 'symbols'
            print(f"Feature generation finished for {symbols}.") # This line was over-indented
        except Exception as e:
            print(f"Error running feature_pipeline.main(): {e}") # Corrected to call main()
    else:
        print(f"Skipping feature generation (module or main function not available). Run manually: python src/feature_engineering/feature_pipeline.py")

    print("\nStep 4: Generating historical AI signals (mocked by default based on config)...")
    if historical_signal_generator and hasattr(historical_signal_generator, 'run_historical_signal_generation'):
        try:
            # Call the main() wrapper function for consistency
            historical_signal_generator.main(
                symbols=symbols,
                process_freq=1, # Process every bar for the 21-row dummy data
                context_window=5      # Use 5 bars of context
            )
            print(f"Historical signal generation finished for {symbols}.")
        except Exception as e:
            print(f"Error running historical_signal_generator.run_historical_signal_generation(): {e}")
    else:
        print(f"Skipping historical signal generation (module or main function not available). Run manually: python src/signal_generation/historical_signal_generator.py")

    print("\nStep 5: Running backtest with Historical AI Strategy via direct call...")
    if main_backtest_runner and hasattr(main_backtest_runner, 'run_backtest_for_workflow'):
        try:
            # Assuming the ConfigManager might be needed by the workflow function if not None
            # For this call, let's pass None to use its internal default.
            # Or, a cm instance could be created here / passed down if run_workflow_a took it.
            main_backtest_runner.run_backtest_for_workflow(
                symbol_to_test=symbols[0] if symbols else 'BTCUSD', # Use first symbol from the list
                strategy_key='AI_HISTORICAL'
                # config_manager_instance=None # Optional: pass a CM if needed by the function's design
            )
            print("Backtest (called by orchestrator) finished.")
        except Exception as e:
            print(f"Error running main_backtest_runner.run_backtest_for_workflow(): {e}")
    else:
        print("Skipping backtest run (module or run_backtest_for_workflow function not available).")
        print("To run manually: python src/strategy_backtesting/main_backtest_runner.py")

    print("\n--- Workflow A conceptual run complete. ---")

def run_workflow_b(symbol='BTCUSD'):
    """
    Executes Workflow B: Simulated Live Trading.
    """
    print(f"\n--- Starting Workflow B: Simulated Live Trading for {symbol} ---")
    # This workflow directly calls the main_live_simulator.py script
    # Ensure src.execution.main_live_simulator has a callable main() or its core logic wrapped.
    # For now, this is conceptual.
    print("This workflow involves running src/execution/main_live_simulator.py.")
    print("Ensure feature data is up-to-date for the symbol before running.")
    print("Example manual run: python src/execution/main_live_simulator.py")
    # try:
    #     from execution import main_live_simulator # Assuming it has a main()
    #     main_live_simulator.main(symbol=symbol)
    # except ImportError:
    #     print("Could not import main_live_simulator to run Workflow B automatically.")
    # except Exception as e:
    #     print(f"Error running main_live_simulator: {e}")
    print("\n--- Workflow B conceptual run complete. ---")


if __name__ == "__main__":
    # This orchestrator is primarily for showing the workflow.
    # It requires the individual scripts to be runnable and their main functions importable.

    # For Workflow A, we need to ensure data is populated first.
    # The data generation steps (1-3) are critical prerequisites for steps 4 and 5.

    print("Orchestrator Example: Running conceptual Workflow A.")
    print("Note: This orchestrator will attempt to call main functions of modules.")
    print("Ensure `src` is in PYTHONPATH if running from project root, or adjust imports.")
    print("Some underlying scripts might need to be modified to expose a callable main() function if they only use if __name__ == '__main__'.")

    run_workflow_a(symbols=['BTCUSD']) # Run for one symbol to reduce output verbosity for test

    # To run workflow B (Simulated Live Trading)
    # run_workflow_b(symbol='BTCUSD')

    print("\nOrchestrator example finished.")
