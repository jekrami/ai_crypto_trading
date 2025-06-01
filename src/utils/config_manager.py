# Placeholder for configuration management
# This module will be responsible for loading and providing access to configuration settings.

import yaml # Or import json, configparser depending on the chosen format

CONFIG_DIR = "config"
DEFAULT_CONFIG_FILE = "settings.yaml" # Or .ini, .json

class ConfigManager:
    def __init__(self, config_file_path=None):
        if config_file_path is None:
            self.config_file_path = os.path.join(CONFIG_DIR, DEFAULT_CONFIG_FILE)
        else:
            self.config_file_path = config_file_path
        self.config = self.load_config()

    def load_config(self):
        """Loads the configuration from the specified file."""
        try:
            with open(self.config_file_path, 'r') as f:
                # For YAML
                config_data = yaml.safe_load(f)
                # For JSON:
                # import json
                # config_data = json.load(f)
                # For INI:
                # import configparser
                # parser = configparser.ConfigParser()
                # parser.read(f)
                # config_data = {section: dict(parser.items(section)) for section in parser.sections()}
            return config_data
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {self.config_file_path}")
            # Potentially load default settings or raise an error
            return {} # Return empty dict or handle appropriately
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}

    def get(self, key, default=None):
        """Retrieves a configuration value for a given key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

# Example Usage (optional)
if __name__ == "__main__":
    import os # Required for the example usage to run standalone if paths are relative
    # This assumes the script is run from the project root or that paths are adjusted.
    # For robust testing, you might need to adjust CWD or use absolute paths.
    print(f"Current CWD: {os.getcwd()}") # Check current working directory

    # Attempt to load config, assuming config/settings.yaml exists relative to where this is run
    # This example might require `config/settings.yaml` to be created first.
    # And `pyyaml` to be installed (`pip install pyyaml`)

    # To make this example work if run directly from src/utils:
    # Adjust path to be relative to project root for the config file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    example_config_path = os.path.join(project_root, CONFIG_DIR, DEFAULT_CONFIG_FILE)

    # Ensure the example config file exists for this test to pass
    if not os.path.exists(os.path.join(project_root, CONFIG_DIR)):
        os.makedirs(os.path.join(project_root, CONFIG_DIR))
    if not os.path.exists(example_config_path):
        with open(example_config_path, 'w') as f:
            yaml.dump({'database': {'path': 'data/example.db'}, 'api_keys': {'cryptopanic': 'YOUR_API_KEY'}}, f)
            print(f"Created example config file: {example_config_path}")

    config_manager = ConfigManager(config_file_path=example_config_path)

    db_path = config_manager.get('database.path', 'default_db.sqlite')
    api_key = config_manager.get('api_keys.cryptopanic', 'default_api_key')
    non_existent = config_manager.get('does.not.exist', 'fallback_value')

    print(f"Database Path: {db_path}")
    print(f"Cryptopanic API Key: {api_key}")
    print(f"Non-existent key: {non_existent}")

    # Clean up example config file
    # os.remove(example_config_path)
    # if not os.listdir(os.path.join(project_root, CONFIG_DIR)):
    #     os.rmdir(os.path.join(project_root, CONFIG_DIR))
