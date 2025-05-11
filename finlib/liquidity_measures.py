import numpy as np


def quoted_spread(asks: np.ndarray, bids: np.ndarray) -> float:
    """Calculate the quoted bid-ask spread.

    Args:
        asks (array-like): Array of ask prices
        bids (array-like): Array of bid prices

    Returns:
        float: Mean-quoted spread (ask - bid)

    Raises:
        ValueError: If asks or bids are empty or have different lengths
        TypeError: If asks or bids cannot be converted to numpy arrays
    """
    try:
        if len(asks) == 0 or len(bids) == 0:
            raise ValueError("Input arrays cannot be empty")

        if len(asks) != len(bids):
            raise ValueError("Ask and bid arrays must have the same length")

        return float(np.mean(asks - bids))
    except TypeError as e:
        raise TypeError("Could not convert input to numpy array") from e


def effective_spread(prices: np.ndarray, bids: np.ndarray, asks: np.ndarray, trade_directions: np.ndarray) -> float:
    """Calculate the effective spread based on trade prices and directions.

    Args:
        prices (array-like): Array of transaction prices
        bids (array-like): Array of bid prices
        asks (array-like): Array of ask prices
        trade_directions (array-like): Array of trade directions (+1 for buyer-initiated, -1 for seller-initiated)

    Returns:
        float: Mean effective spread

    Raises:
        ValueError: If input arrays are empty or have different lengths
        TypeError: If inputs cannot be converted to numpy arrays
    """
    try:
        if len(prices) == 0 or len(bids) == 0 or len(asks) == 0 or len(trade_directions) == 0:
            raise ValueError("Input arrays cannot be empty")

        if not (len(prices) == len(bids) == len(asks) == len(trade_directions)):
            raise ValueError("All input arrays must have the same length")

        mid_prices = (bids + asks) / 2
        return float(np.mean(2 * trade_directions * (prices - mid_prices)))
    except TypeError as e:
        raise TypeError("Could not convert input to numpy array") from e


def test_quoted_spread():
    asks = np.array([101.5, 102.0, 102.5])
    bids = np.array([100.0, 100.5, 101.0])
    expected_spread = 1.5
    assert abs(quoted_spread(asks, bids) - expected_spread) < 1e-5


def test_effective_spread():
    prices = np.array([101.5, 102.0, 102.5])
    bids = np.array([100.0, 100.5, 101.0])
    asks = np.array([102.0, 102.5, 103.0])
    trade_directions = np.array([1, -1, 1])
    # Expected effective spread calculation
    mid_prices = (bids + asks) / 2
    expected_spread = np.mean(2 * trade_directions * (prices - mid_prices))
    assert abs(effective_spread(prices, bids, asks, trade_directions) - expected_spread) < 1e-5

if __name__ == "__main__":
    test_quoted_spread()
    test_effective_spread()