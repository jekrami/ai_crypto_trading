import os
import sys
import ollama
import requests # For requests.exceptions.ConnectionError
from typing import Dict, Any, Optional, List

# Setup paths
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /app/src
PROJECT_ROOT = os.path.dirname(SRC_DIR) # /app
sys.path.append(SRC_DIR)

from utils.config_manager import ConfigManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

class OllamaClient:
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        if config_manager:
            self.cm = config_manager
        else:
            # Fallback to default config path if not provided (e.g. for standalone use)
            config_file_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
            self.cm = ConfigManager(config_file_path=config_file_path)

        self.host = self.cm.get('ollama_settings.host', "http://127.0.0.1:11434")
        self.default_model = self.cm.get('ollama_settings.model', "llama3")
        self.default_timeout = self.cm.get('ollama_settings.default_timeout_seconds', 60)

        self.mock_ollama = self.cm.get('ollama_settings.mock_ollama_for_testing', False)
        self.mock_response_text = self.cm.get(
            'ollama_settings.mock_response_text',
            "Signal: HOLD. Reasoning: Market conditions are neutral, providing no clear buy or sell signal. (Mocked)"
        )

        try:
            # Do not initialize self.client if mocking, to avoid connection attempt if server is down
            if not self.mock_ollama:
                 self.client = ollama.Client(host=self.host, timeout=self.default_timeout)
                 # Attempt a listing of models to verify connection, but handle failure gracefully.
                 try:
                    self.client.list()
                    logger.info(f"Successfully connected to Ollama host at {self.host}")
                 except (requests.exceptions.ConnectionError, ollama.ResponseError) as e:
                    logger.warning(f"Could not connect to Ollama host at {self.host} during init. "
                                   f"Client initialized but connection seems down. Error: {e}")
                    # Client is initialized, but calls will likely fail if server remains down.
                    # This allows the app to start but logs the issue.
            else:
                self.client = None # No client needed if mocking all responses
                logger.info("Ollama client initialized in MOCK mode. Will use mock responses.")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}", exc_info=True)
            self.client = None # Ensure client is None if initialization fails for any reason

    def generate_text(self,
                      prompt: str,
                      model_name: Optional[str] = None,
                      system_message: str = "You are a helpful financial assistant providing trading insights.",
                      options: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generates text using the Ollama API, with a timeout.
        Handles mock responses and API errors.
        """
        if self.mock_ollama:
            logger.warning(f"MOCKING OLLAMA: Returning predefined mock response for model '{model_name or self.default_model}'.")
            logger.debug(f"Mocked prompt (first 200 chars): {prompt[:200]}")
            return self.mock_response_text

        if not self.client:
            logger.error("Ollama client is not initialized or connection failed. Cannot generate text.")
            return None

        target_model = model_name or self.default_model
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': prompt}
        ]

        try:
            logger.info(f"Sending prompt to Ollama model '{target_model}' (timeout: {self.default_timeout}s). Prompt (first 200 chars): {prompt[:200]}...")

            # Note: ollama.Client already has a timeout set during __init__
            # If you need per-request timeout different from client's default,
            # ollama library might not support it directly in client.chat().
            # The timeout set on ollama.Client applies to the HTTP request.
            response = self.client.chat(
                model=target_model,
                messages=messages,
                options=options # e.g., {"temperature": 0.7}
            )

            response_content = response.get('message', {}).get('content')
            if response_content:
                logger.info(f"Received response from Ollama model '{target_model}'.")
                logger.debug(f"Full Ollama response: {response_content}")
                return response_content.strip()
            else:
                logger.error(f"Ollama response for model '{target_model}' did not contain message content. Full response: {response}")
                return None

        except ollama.ResponseError as e:
            logger.error(f"Ollama API ResponseError for model '{target_model}': {e.status_code} - {e.error}")
            if e.status_code == 404:
                 logger.error(f"Model '{target_model}' not found. Please ensure it is pulled in Ollama.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama API ConnectionError: Failed to connect to Ollama server at {self.host}. Details: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while communicating with Ollama model '{target_model}': {e}", exc_info=True)

        return None

if __name__ == "__main__":
    logger.info("Testing OllamaClient...")

    # This requires config/settings.yaml to be set up correctly relative to PROJECT_ROOT
    # Test with default config (mock_ollama_for_testing: false)
    print("\n--- Test Case 1: Attempting Live Connection (assuming mock_ollama_for_testing=false in config) ---")
    # To force a specific mode for this test, we can temporarily alter the config values:
    # Create a ConfigManager instance for this test
    test_config_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    if not os.path.exists(test_config_path):
         logger.error(f"Test Case Aborted: Config file not found at {test_config_path}")
    else:
        # Test 1: Live connection (assuming Ollama might be running)
        # Ensure mock_ollama_for_testing is false for this part of the test if you have Ollama
        # For automated testing where Ollama isn't guaranteed, this will show connection error or success
        cfg_live = ConfigManager(test_config_path)
        original_mock_setting = cfg_live.get('ollama_settings.mock_ollama_for_testing', False)

        # Temporarily set mock_ollama_for_testing to False for this part of the test
        # THIS IS DIFFICULT TO DO WITHOUT MODIFYING THE FILE ON DISK.
        # The ConfigManager reads the file. So, for a true test of both paths,
        # one would typically run with the file configured for mock=false, then mock=true.
        # For this __main__, we'll just use whatever is in the file first.

        logger.info(f"Using 'mock_ollama_for_testing: {original_mock_setting}' from config file.")
        client_live_or_config_mock = OllamaClient(cfg_live)
        prompt_example = "What is the weather like today?"
        if client_live_or_config_mock.client or client_live_or_config_mock.mock_ollama : # Check if client could be used or is mocked
            response_live = client_live_or_config_mock.generate_text(prompt_example)
            if response_live:
                print(f"Response from Ollama (or config-defined mock): {response_live}")
            else:
                print("No response or error during live/config-mock test.")
        else:
             print("Ollama client not initialized for live/config-mock test (e.g. connection error on init and not mocking).")


        # Test 2: Forced Mocking (overriding any file setting via direct class attribute modification)
        print("\n--- Test Case 2: Forced Mocking ---")
        # We need a new client instance or to modify the existing one's mock settings.
        # For simplicity, let's create a new one and force its mock attributes for this test block.
        cfg_mock_test = ConfigManager(test_config_path) # Re-read
        client_forced_mock = OllamaClient(cfg_mock_test)
        client_forced_mock.mock_ollama = True # Force mocking
        client_forced_mock.mock_response_text = "Signal: SELL. Reasoning: This is a forced mock response for testing. (Mocked)"

        response_mock = client_forced_mock.generate_text("Another prompt for the forced mock.")
        if response_mock:
            print(f"Response from Forced Mock Ollama: {response_mock}")
        else:
            print("No response from forced mock (this shouldn't happen if configured right).")

    logger.info("OllamaClient test finished.")
