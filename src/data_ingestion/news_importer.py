import os
import sys
import json
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Add src directory to Python path for direct imports
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /app/src
PROJECT_ROOT = os.path.dirname(SRC_DIR) # /app
sys.path.append(SRC_DIR)

from utils.db_manager import DBManager
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Placeholder for your actual API key, remind user to change it
CRYPTOPANIC_API_KEY_PLACEHOLDER = "YOUR_API_KEY_HERE"

class NewsImporter:
    def __init__(self, config_manager: ConfigManager, db_manager: DBManager):
        self.config_manager = config_manager
        self.db_manager = db_manager
        self.api_key = self.config_manager.get('api_configs.cryptopanic.api_key')
        self.base_url = self.config_manager.get('api_configs.cryptopanic.base_url', "https://cryptopanic.com/api/v1/")

    def create_news_table(self) -> None:
        """Creates the news_articles table if it doesn't exist."""
        table_name = "news_articles"
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            source_domain TEXT,
            published_at TEXT,
            kind TEXT,
            created_at_api TEXT,
            raw_currencies_data TEXT,
            ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        if self.db_manager.execute_query(query):
            logger.info(f"Table '{table_name}' checked/created successfully.")
        else:
            logger.error(f"Failed to create/check table '{table_name}'.")

    def get_mock_news_data(self) -> List[Dict[str, Any]]:
        """Returns a list of mock news articles for testing."""
        logger.info("Using mocked news data because API key is a placeholder or missing.")
        return [
            {
                "id": "mock_1696128000_1", # Using timestamp + index for potential uniqueness
                "kind": "news",
                "domain": "cointelegraph.com",
                "title": "Bitcoin Price Hits New All-Time High - Or Does It? (Mocked)",
                "published_at": "2023-10-01T12:00:00Z",
                "created_at": "2023-10-01T11:55:00Z", # API's created_at
                "url": "https://cointelegraph.com/news/mock-btc-ath",
                "source": {"title": "CoinTelegraph", "region": "en", "domain": "cointelegraph.com"},
                "currencies": [{"code": "BTC", "title": "Bitcoin", "slug": "bitcoin", "url": "..."}],
                "votes": {"liked": 100, "disliked": 5, "positive": 80, "negative": 10, "important": 90, "saved": 50, "comments": 20}
            },
            {
                "id": "mock_1696128000_2",
                "kind": "media",
                "domain": "decrypt.co",
                "title": "Ethereum Devs Announce Major Upgrade Timeline (Mocked)",
                "published_at": "2023-10-01T14:30:00Z",
                "created_at": "2023-10-01T14:25:00Z",
                "url": "https://decrypt.co/news/mock-eth-upgrade",
                "source": {"title": "Decrypt", "region": "en", "domain": "decrypt.co"},
                "currencies": [{"code": "ETH", "title": "Ethereum", "slug": "ethereum", "url": "..."}],
                "votes": {"liked": 120, "disliked": 2, "positive": 100, "negative": 5, "important": 95, "saved": 60, "comments": 25}
            },
             { # Duplicate URL to test INSERT OR IGNORE
                "id": "mock_1696128000_3",
                "kind": "news",
                "domain": "decrypt.co",
                "title": "Ethereum Devs Announce Major Upgrade Timeline - AGAIN (Mocked)",
                "published_at": "2023-10-01T14:35:00Z", # Slightly different time
                "created_at": "2023-10-01T14:30:00Z",
                "url": "https://decrypt.co/news/mock-eth-upgrade", # SAME URL
                "source": {"title": "Decrypt", "region": "en", "domain": "decrypt.co"},
                "currencies": [{"code": "ETH", "title": "Ethereum", "slug": "ethereum", "url": "..."}],
                "votes": {"liked": 10, "disliked": 0, "positive": 8, "negative": 1, "important": 9, "saved": 5, "comments": 2}
            }
        ]

    def fetch_news_from_cryptopanic_api(self, auth_token: str, **kwargs) -> List[Dict[str, Any]]:
        """Fetches news from Cryptopanic API."""
        if not auth_token or auth_token == CRYPTOPANIC_API_KEY_PLACEHOLDER:
            logger.warning("Cryptopanic API key is a placeholder or missing. Using mocked data.")
            return self.get_mock_news_data()

        # Example: Fetching general news, public endpoint often doesn't need auth_token for basic posts
        # but check API docs. Some filters might require it.
        # For 'posts' endpoint: https://cryptopanic.com/api/v1/posts/?auth_token=YOUR_API_KEY&public=true
        # We'll assume for now we are fetching general posts.
        # Adjust 'public=true' or other filters as needed.

        params = {"auth_token": auth_token, "public": "true"} # Basic params
        # Add any other filters passed via kwargs, e.g., currencies, kind, region, filter
        # Example: params.update({"currencies": "BTC,ETH", "filter": "rising"})
        params.update(kwargs)

        request_url = f"{self.base_url.rstrip('/')}/posts/"

        logger.info(f"Fetching news from Cryptopanic API: {request_url} with params: { {k:v for k,v in params.items() if k != 'auth_token'} }")

        try:
            response = requests.get(request_url, params=params, timeout=10)
            response.raise_for_status()  # Raises HTTPError for 4XX/5XX status codes

            data = response.json()
            if "results" in data and isinstance(data["results"], list):
                logger.info(f"Successfully fetched {len(data['results'])} news items from API.")
                return data["results"]
            else:
                logger.warning(f"No 'results' list found in API response or it's not a list. Full response: {data}")
                return []
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred during API request: {req_err}")
        except json.JSONDecodeError as json_err:
            logger.error(f"Error decoding JSON response: {json_err} - Response: {response.text if 'response' in locals() else 'N/A'}")
        return [] # Return empty list on error

    def insert_news_data(self, news_items: List[Dict[str, Any]]) -> int:
        """Inserts a list of news items into the news_articles table."""
        table_name = "news_articles"
        insert_query = f"""
        INSERT OR IGNORE INTO {table_name}
        (id, title, url, source_domain, published_at, kind, created_at_api, raw_currencies_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """

        inserted_count = 0
        for item in news_items:
            try:
                # Ensure 'id' exists and is suitable. If not, generate one or use URL.
                # Cryptopanic 'id' is usually a number, but can change or might not be globally unique over time.
                # Let's use 'slug' or a combination if 'id' isn't reliable or always present.
                # For now, assuming API 'id' is usable or that URL's uniqueness handles conflicts.
                # The mock data uses string IDs like "mock_123". Actual API might have integer IDs.
                # The DB schema has 'id' as TEXT PRIMARY KEY.

                api_post_id = item.get("id") # This is usually an integer from the API for posts.
                if api_post_id is None and item.get("slug"): # Fallback to slug if id is missing
                    api_post_id = item.get("slug")
                if api_post_id is None: # If still none, this item cannot be inserted with current PK
                    logger.warning(f"Skipping item due to missing 'id' and 'slug': {item.get('title')}")
                    continue

                # Convert numeric ID to string if schema expects TEXT
                api_post_id_str = str(api_post_id)

                currencies_json = json.dumps(item.get("currencies")) if item.get("currencies") else None

                # API 'domain' vs 'source.domain'. Let's prefer 'source.domain' if available
                source_domain_val = item.get("domain") # Default
                if item.get("source") and isinstance(item["source"], dict) and item["source"].get("domain"):
                    source_domain_val = item["source"]["domain"]

                # API `created_at` for `created_at_api` field
                created_at_api_val = item.get("created_at", item.get("published_at")) # Fallback to published_at

                params = (
                    api_post_id_str,
                    item.get("title"),
                    item.get("url"),
                    source_domain_val,
                    item.get("published_at"), # Assumed to be ISO 8601 string from API
                    item.get("kind"),
                    created_at_api_val,
                    currencies_json
                )
                if self.db_manager.execute_query(insert_query, params):
                    # This doesn't tell us if it was an INSERT or IGNORE.
                    # To get precise count, would need to check changes/rowcount from cursor.
                    # For this example, we'll assume it implies an attempt.
                    # A more accurate way is to use `cursor.rowcount` if db_manager exposes it.
                    inserted_count += 1 # This counts attempts, not actual inserts.
                                        # For actual inserts, a SELECT COUNT before/after or checking cursor.lastrowid (for non-IGNORE)
                                        # or cursor.rowcount (driver dependent) is needed.
                                        # Given `INSERT OR IGNORE`, this is an optimistic count.
                else:
                    logger.warning(f"Failed to execute insert for news item ID {api_post_id_str}, URL {item.get('url')}")

            except Exception as e:
                logger.error(f"Error processing news item for DB insertion: {item.get('title')} - {e}", exc_info=True)

        # For a more accurate count of *newly inserted* rows:
        # This is complex with INSERT OR IGNORE. A common pattern is to use
        # `cursor.execute(...); self.conn.commit(); count = cursor.rowcount`
        # but `db_manager.execute_query` hides the cursor.
        # If `db_manager.execute_query` could return rowcount, that would be ideal.
        # For now, this 'inserted_count' is the number of successful executions of INSERT OR IGNORE.
        if inserted_count > 0:
             logger.info(f"Attempted to insert/ignore {inserted_count} news items.")
        else:
             logger.info("No news items were processed for insertion (or all failed execution).")
        return inserted_count # Return number of attempted inserts

    def run_news_import(self) -> None:
        """Main logic for fetching and storing news data."""
        logger.info("Starting news data import process...")

        if not self.api_key:
            logger.error("Cryptopanic API key is not configured. Please set 'api_configs.cryptopanic.api_key' in settings.")
            # Decide if to proceed with mock data or exit. Task asks to use mock if placeholder.
            if self.api_key != CRYPTOPANIC_API_KEY_PLACEHOLDER: # If it's empty but not the placeholder
                 return # Exit if truly empty

        # If API key is the placeholder, fetch_news_from_cryptopanic_api will use mock data.
        # If API key is set to something else (presumably a real key), it will attempt a live call.

        with self.db_manager: # Manages connect/disconnect
            self.create_news_table()

            # Example: Fetch news (can add filters like currencies='BTC,ETH', kind='news')
            news_items = self.fetch_news_from_cryptopanic_api(auth_token=self.api_key)

            if news_items:
                self.insert_news_data(news_items)
            else:
                logger.info("No news items fetched or returned from API/mock.")

        logger.info("News data import process finished.")


def main():
    logger.info("Initializing news importer...")

    config_file_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")

    # Temporary ConfigManager to get DB path, as DBManager needs it for its own directory creation
    temp_config_for_db_path = ConfigManager(config_file_path=config_file_path)
    db_path_from_config = temp_config_for_db_path.get('database.path', os.path.join("data", "trading_system.db"))

    db_full_path = db_path_from_config
    if not os.path.isabs(db_full_path):
        db_full_path = os.path.join(PROJECT_ROOT, db_full_path)

    db_dir = os.path.dirname(db_full_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        logger.info(f"Created database directory: {db_dir}")

    # Main ConfigManager and DBManager instances
    config_manager = ConfigManager(config_file_path=config_file_path)
    db_manager = DBManager(db_path=db_full_path)

    importer = NewsImporter(config_manager, db_manager)
    importer.run_news_import()

if __name__ == "__main__":
    # Ensure PyYAML and requests are installed
    try:
        import yaml
        import requests
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Please install PyYAML and requests: pip install PyYAML requests")
        sys.exit(1)

    main()
