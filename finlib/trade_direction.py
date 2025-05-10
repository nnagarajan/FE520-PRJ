import numpy as np


def tick_test_direction(prices: np.ndarray) -> np.ndarray:
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
    n = len(prices)
    mid_prices = (bids + asks) / 2
    direction = np.zeros(n, dtype=int)

    tick_direction = tick_test_direction(prices)

    for i in range(n):
        if prices[i] > mid_prices[i]:
            direction[i] = 1
        elif prices[i] < mid_prices[i]:
            direction[i] = -1
        else:
            direction[i] = tick_direction[i]

    return direction


def test_tick_test_direction():
    prices = np.array([100, 101, 101, 100, 100, 99, 100])
    expected_directions = np.array([0, 1, 1, -1, -1, -1, 1])
    calculated_directions = tick_test_direction(prices)
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
