import numpy as np


def tick_direction(prices: np.ndarray) -> np.ndarray:
    """Calculate trade direction using tick test.

    Args:
        prices (np.ndarray): Array of transaction prices

    Returns:
        np.ndarray: Array of trade directions (1: uptick, -1: downtick, 0: undefined)

    Raises:
        TypeError: If prices cannot be converted to numpy array
        ValueError: If prices array is empty
    """
    if not isinstance(prices, np.ndarray):
        raise TypeError("Prices must be a numpy array")
    if len(prices) == 0:
        raise ValueError("Prices array cannot be empty")

    n = len(prices)
    direction = np.zeros(n, dtype=int)

    for i in range(1, n):
        if prices[i] > prices[i - 1]:
            direction[i] = 1  # uptick
        elif prices[i] < prices[i - 1]:
            direction[i] = -1  # downtick
        else:  # zero tick
            if i >= 2:
                if prices[i - 1] > prices[i - 2]:
                    direction[i] = 1  # zero-uptick
                elif prices[i - 1] < prices[i - 2]:
                    direction[i] = -1  # zero-downtick
                else:
                    direction[i] = direction[i - 1]  # propagate previous if two consecutive zeros
            else:
                direction[i] = direction[i - 1]  # propagate initial tick

    direction[0] = 0  # undefined for the first trade
    return direction


def lee_ready_direction(prices: np.ndarray, bids: np.ndarray, asks: np.ndarray) -> np.ndarray:
    """Calculate trade direction using Lee-Ready algorithm.

    Args:
        prices (np.ndarray): Array of transaction prices
        bids (np.ndarray): Array of bid prices
        asks (np.ndarray): Array of ask prices

    Returns:
        np.ndarray: Array of trade directions (1: buy, -1: sell)

    Raises:
        TypeError: If inputs cannot be converted to numpy arrays
        ValueError: If input arrays are empty or have different lengths
    """
    if not all(isinstance(x, np.ndarray) for x in [prices, bids, asks]):
        raise TypeError("All inputs must be numpy arrays")
    if len(prices) == 0 or len(bids) == 0 or len(asks) == 0:
        raise ValueError("Input arrays cannot be empty")
    if not (len(prices) == len(bids) == len(asks)):
        raise ValueError("All input arrays must have the same length")

    n = len(prices)
    mid_prices = (bids + asks) / 2
    direction = np.zeros(n, dtype=int)

    tick_directions = tick_direction(prices)

    for i in range(n):
        if prices[i] > mid_prices[i]:
            direction[i] = 1
        elif prices[i] < mid_prices[i]:
            direction[i] = -1
        else:
            direction[i] = tick_directions[i]

    return direction


def test_tick_test_direction():
    prices = np.array([100, 101, 101, 100, 100, 99, 100])
    expected_directions = np.array([0, 1, 1, -1, -1, -1, 1])
    calculated_directions = tick_direction(prices)
    assert np.array_equal(calculated_directions, expected_directions), \
        f"Expected {expected_directions}, but got {calculated_directions}"
    print("Test passed.")


def test_lee_ready_direction():
    prices = np.array([100, 101, 100, 99, 100])
    bids = np.array([99.5, 100.5, 99.5, 98.5, 99.5])
    asks = np.array([100.5, 101.5, 100.5, 99.5, 100.5])
    expected_directions = np.array([0, 1, -1, -1, 1])
    calculated_directions = lee_ready_direction(prices, bids, asks)
    assert np.array_equal(calculated_directions, expected_directions), \
        f"Expected {expected_directions}, but got {calculated_directions}"
    print("Lee-Ready Test passed.")


if __name__ == "__main__":
    test_tick_test_direction()
    test_lee_ready_direction()
