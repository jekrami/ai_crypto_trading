import copy
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

# Configure logger if needed
# from utils.logger import setup_logger
# logger = setup_logger(__name__)

class SimulatedExchange:
    def __init__(self, initial_balances: Optional[Dict[str, float]] = None,
                 fee_percent: float = 0.001):
        """
        Initializes the simulated exchange.

        :param initial_balances: Dict of initial asset balances, e.g., {"USD": 10000.0, "BTC": 0.0}.
        :param fee_percent: Trading fee as a percentage (e.g., 0.001 for 0.1%).
        """
        if initial_balances is None:
            self.balances = {"USD": 10000.0, "BTC": 0.0, "ETH": 0.0} # Default if none provided
        else:
            self.balances = copy.deepcopy(initial_balances)

        self.fee_percent = fee_percent
        self.trade_history: List[Dict[str, Any]] = []
        self.open_orders: Dict[str, Dict[str, Any]] = {} # For future use with limit orders
        self.next_order_id = 1

        # print(f"SimulatedExchange initialized. Balances: {self.balances}, Fee: {self.fee_percent*100}%") # Optional debug

    def get_balance(self, asset_code: str) -> float:
        """Returns the current balance of a given asset."""
        return self.balances.get(asset_code.upper(), 0.0)

    def _get_assets_from_symbol(self, symbol: str) -> Optional[Tuple[str, str]]:
        """
        Parses a trading symbol like "BTCUSD" into base asset (BTC) and quote asset (USD).
        Assumes symbol format ASSET1ASSET2 where ASSET2 is typically the quote currency.
        More robust parsing might be needed for varied symbol formats.
        """
        # Common quote currencies
        quote_currencies = ["USD", "USDT", "USDC", "EUR", "GBP", "JPY", "BTC", "ETH"]
        symbol_upper = symbol.upper()

        for qc in quote_currencies:
            if symbol_upper.endswith(qc):
                asset1 = symbol_upper[:-len(qc)]
                asset2 = qc
                if asset1: # Ensure asset1 is not empty
                    return asset1, asset2

        # Fallback for 3-letter base assets if common quotes don't match (e.g. BTC/DAI -> BTCDAI)
        if len(symbol_upper) > 3 : # Crude split if no common quote found, e.g. ABCXYZ -> ABC, XYZ
            asset1 = symbol_upper[:3]
            asset2 = symbol_upper[3:]
            # print(f"Warning: Using fallback symbol split for {symbol}: ({asset1}, {asset2})")
            return asset1, asset2

        # print(f"Error: Could not reliably parse symbol '{symbol}' into asset1 and asset2.")
        return None


    def execute_market_order(self,
                             symbol: str,
                             order_type: str, # "BUY" or "SELL" (asset1)
                             quantity_asset1: float,
                             current_market_price_asset2_per_asset1: float,
                             timestamp: Any # Can be datetime, int timestamp, or string
                             ) -> Dict[str, Any]:
        """
        Executes a market order on the simulated exchange.

        :param symbol: Trading symbol, e.g., "BTCUSD".
        :param order_type: "BUY" (buy asset1 with asset2) or "SELL" (sell asset1 for asset2).
        :param quantity_asset1: The amount of asset1 to buy or sell.
        :param current_market_price_asset2_per_asset1: Current market price (asset2 per unit of asset1).
        :param timestamp: Timestamp of the order execution.
        :return: A dictionary with order execution status and details.
        """
        assets = self._get_assets_from_symbol(symbol)
        if not assets:
            return {'status': 'REJECTED', 'reason': f'Invalid symbol format: {symbol}', 'timestamp': timestamp}
        asset1, asset2 = assets

        order_type = order_type.upper()
        execution_price = current_market_price_asset2_per_asset1 # No exchange-level slippage here yet

        if quantity_asset1 <= 1e-9: # Effectively zero quantity
             return {'status': 'REJECTED', 'reason': 'Order quantity is too small or zero.', 'timestamp': timestamp}


        trade_details: Dict[str, Any] = {
            'order_id': self.next_order_id,
            'timestamp': timestamp,
            'symbol': symbol,
            'type': order_type,
            'price': execution_price,
            'quantity_asset1': quantity_asset1,
        }

        if order_type == "BUY":
            cost_asset2 = quantity_asset1 * execution_price
            fee_asset2 = cost_asset2 * self.fee_percent
            total_cost_asset2 = cost_asset2 + fee_asset2

            if self.get_balance(asset2) >= total_cost_asset2:
                self.balances[asset2] -= total_cost_asset2
                self.balances[asset1] = self.get_balance(asset1) + quantity_asset1

                trade_details.update({
                    'status': 'FILLED',
                    'total_cost_asset2': total_cost_asset2,
                    'fee_asset2': fee_asset2,
                    'filled_quantity_asset1': quantity_asset1
                })
                # print(f"SimEx: BUY {quantity_asset1:.4f} {asset1} for {total_cost_asset2:.2f} {asset2} (Price: {execution_price:.2f}, Fee: {fee_asset2:.2f})")
            else:
                trade_details.update({'status': 'REJECTED', 'reason': f'Insufficient {asset2} funds.'})
                # print(f"SimEx: BUY REJECTED - Insufficient {asset2}. Need {total_cost_asset2:.2f}, Have {self.get_balance(asset2):.2f}")

        elif order_type == "SELL":
            if self.get_balance(asset1) >= quantity_asset1:
                proceeds_asset2 = quantity_asset1 * execution_price
                fee_asset2 = proceeds_asset2 * self.fee_percent
                net_proceeds_asset2 = proceeds_asset2 - fee_asset2

                self.balances[asset1] -= quantity_asset1
                self.balances[asset2] = self.get_balance(asset2) + net_proceeds_asset2

                trade_details.update({
                    'status': 'FILLED',
                    'net_proceeds_asset2': net_proceeds_asset2,
                    'fee_asset2': fee_asset2,
                    'filled_quantity_asset1': quantity_asset1
                })
                # print(f"SimEx: SELL {quantity_asset1:.4f} {asset1} for {net_proceeds_asset2:.2f} {asset2} (Price: {execution_price:.2f}, Fee: {fee_asset2:.2f})")
            else:
                trade_details.update({'status': 'REJECTED', 'reason': f'Insufficient {asset1} funds.'})
                # print(f"SimEx: SELL REJECTED - Insufficient {asset1}. Need {quantity_asset1:.4f}, Have {self.get_balance(asset1):.4f}")
        else:
            trade_details.update({'status': 'REJECTED', 'reason': f'Invalid order type: {order_type}'})

        if trade_details['status'] == 'FILLED':
            self.trade_history.append(trade_details)
            self.next_order_id +=1

        return trade_details

if __name__ == '__main__':
    print("--- Testing SimulatedExchange ---")

    initial_bals = {"USD": 10000.0, "BTC": 1.0}
    exchange = SimulatedExchange(initial_balances=initial_bals, fee_percent=0.001) # 0.1% fee

    print(f"Initial Balances: USD={exchange.get_balance('USD')}, BTC={exchange.get_balance('BTC')}")

    # Test BUY order (BTC with USD)
    print("\nTest 1: BUY BTC with USD")
    buy_symbol = "BTCUSD" # Buy BTC (asset1) using USD (asset2)
    buy_qty_btc = 0.1
    btc_price_usd = 20000.0
    ts1 = datetime.now(timezone.utc)

    # Expected: Cost = 0.1 * 20000 = 2000 USD. Fee = 2000 * 0.001 = 2 USD. Total Cost = 2002 USD.
    # USD Balance: 10000 - 2002 = 7998 USD. BTC Balance: 1.0 + 0.1 = 1.1 BTC.
    buy_result = exchange.execute_market_order(buy_symbol, "BUY", buy_qty_btc, btc_price_usd, ts1)
    print(f"Buy Result: {buy_result}")
    print(f"Balances after BUY: USD={exchange.get_balance('USD')}, BTC={exchange.get_balance('BTC')}")
    assert buy_result['status'] == 'FILLED'
    assert abs(exchange.get_balance('USD') - 7998.0) < 1e-9
    assert abs(exchange.get_balance('BTC') - 1.1) < 1e-9

    # Test SELL order (BTC for USD)
    print("\nTest 2: SELL BTC for USD")
    sell_qty_btc = 0.05
    btc_price_usd_sell = 21000.0 # Price went up
    ts2 = datetime.now(timezone.utc)

    # Expected: Proceeds = 0.05 * 21000 = 1050 USD. Fee = 1050 * 0.001 = 1.05 USD. Net Proceeds = 1048.95 USD.
    # USD Balance: 7998 + 1048.95 = 9046.95 USD. BTC Balance: 1.1 - 0.05 = 1.05 BTC.
    sell_result = exchange.execute_market_order(buy_symbol, "SELL", sell_qty_btc, btc_price_usd_sell, ts2)
    print(f"Sell Result: {sell_result}")
    print(f"Balances after SELL: USD={exchange.get_balance('USD')}, BTC={exchange.get_balance('BTC')}")
    assert sell_result['status'] == 'FILLED'
    assert abs(exchange.get_balance('USD') - 9046.95) < 1e-9
    assert abs(exchange.get_balance('BTC') - 1.05) < 1e-9

    # Test Insufficient Funds for BUY
    print("\nTest 3: Insufficient USD for BUY")
    buy_qty_btc_large = 1.0 # Cost = 1.0 * 21000 = 21000. Fee = 21. Total = 21021. Have 9046.95 USD.
    buy_fail_result = exchange.execute_market_order(buy_symbol, "BUY", buy_qty_btc_large, btc_price_usd_sell, datetime.now(timezone.utc))
    print(f"Buy Fail Result: {buy_fail_result}")
    print(f"Balances after failed BUY: USD={exchange.get_balance('USD')}, BTC={exchange.get_balance('BTC')}")
    assert buy_fail_result['status'] == 'REJECTED'
    assert abs(exchange.get_balance('USD') - 9046.95) < 1e-9 # Unchanged
    assert abs(exchange.get_balance('BTC') - 1.05) < 1e-9   # Unchanged

    # Test Insufficient Funds for SELL
    print("\nTest 4: Insufficient BTC for SELL")
    sell_qty_btc_large = 2.0 # Have 1.05 BTC
    sell_fail_result = exchange.execute_market_order(buy_symbol, "SELL", sell_qty_btc_large, btc_price_usd_sell, datetime.now(timezone.utc))
    print(f"Sell Fail Result: {sell_fail_result}")
    print(f"Balances after failed SELL: USD={exchange.get_balance('USD')}, BTC={exchange.get_balance('BTC')}")
    assert sell_fail_result['status'] == 'REJECTED'
    assert abs(exchange.get_balance('USD') - 9046.95) < 1e-9 # Unchanged
    assert abs(exchange.get_balance('BTC') - 1.05) < 1e-9   # Unchanged

    print("\nTrade History:")
    for trade in exchange.trade_history:
        print(trade)
    assert len(exchange.trade_history) == 2

    # Test symbol parsing
    print("\nTest 5: Symbol Parsing")
    assert exchange._get_assets_from_symbol("ETHUSDT") == ("ETH", "USDT")
    assert exchange._get_assets_from_symbol("XRPBTC") == ("XRP", "BTC")
    assert exchange._get_assets_from_symbol("DOGEUSD") == ("DOGE", "USD")
    assert exchange._get_assets_from_symbol("btcusd") == ("BTC", "USD")
    assert exchange._get_assets_from_symbol("ADAEUR") == ("ADA", "EUR")
    # Fallback for unknown quote, assumes 3-letter base
    # print(exchange._get_assets_from_symbol("SOMECOINXYZ")) # Should be ("SOM", "ECOINXYZ") by current logic
    # assert exchange._get_assets_from_symbol("SOMECOINXYZ") == ("SOM", "ECOINXYZ") # Example of a more complex case

    print("\nAll tests seem to pass based on assertions.")
