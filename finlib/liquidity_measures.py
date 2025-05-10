import numpy as np


def quoted_spread(asks, bids):
    return float(np.mean(asks - bids))


def effective_spread(prices, bids, asks, trade_directions):
    mid_prices = (bids + asks) / 2
    return float(np.mean(2 * trade_directions * (prices - mid_prices)))
