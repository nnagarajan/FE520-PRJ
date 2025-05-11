"""
Investment metrics calculation module providing common financial analysis tools
including CAGR, ROI, Sharpe Ratio, Volatility calculations and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cagr(start_value, end_value, periods):
    """Calculate Compound Annual Growth Rate (CAGR).

    Args:
        start_value (float): Initial investment value
        end_value (float): Final investment value
        periods (float): Number of periods (usually years)

    Returns:
        float: CAGR as a decimal, or error message if calculation fails

    Raises:
        ValueError: If any input is not positive
    """
    try:
        # Validate inputs are positive
        if start_value <= 0 or end_value <= 0 or periods <= 0:
            raise ValueError("All inputs must be positive numbers.")
        # Calculate CAGR using the standard formula
        return (end_value / start_value) ** (1 / periods) - 1
    except Exception as e:
        return f"CAGR calculation error: {e}"

def roi(gain, cost):
    """Calculate Return on Investment (ROI).

    Args:
        gain (float): Total value of investment including gains
        cost (float): Initial cost of investment

    Returns:
        float: ROI as a decimal, or error message if calculation fails

    Raises:
        ZeroDivisionError: If cost is zero
    """
    try:
        if cost == 0:
            raise ZeroDivisionError("Cost of investment cannot be zero.")
        # Calculate ROI as (gain - cost) / cost
        return (gain - cost) / cost
    except Exception as e:
        return f"ROI calculation error: {e}"

def sharpe_ratio(returns, risk_free_rate=0.01):
    """Calculate Sharpe Ratio for a set of returns.

    Args:
        returns (array-like): Array of investment returns
        risk_free_rate (float, optional): Risk-free rate. Defaults to 0.01

    Returns:
        float: Sharpe ratio, or error message if calculation fails

    Raises:
        ValueError: If standard deviation of returns is zero
    """
    try:
        # Calculate excess returns over risk-free rate
        excess_returns = np.array(returns) - risk_free_rate
        # Calculate standard deviation of returns
        std_dev = np.std(returns)
        if std_dev == 0:
            raise ValueError("Standard deviation of returns is zero.")
        # Calculate Sharpe ratio
        return np.mean(excess_returns) / std_dev
    except Exception as e:
        return f"Sharpe Ratio error: {e}"

def calculate_volatility(prices):
    """Calculate volatility from a series of prices.

    Args:
        prices (pd.Series): Time series of prices

    Returns:
        float: Volatility measure (standard deviation of log returns),
              or error message if calculation fails
    """
    try:
        # Calculate log returns
        returns = np.log(prices / prices.shift(1)).dropna()
        # Return standard deviation of returns as volatility measure
        return np.std(returns)
    except Exception as e:
        return f"Volatility calculation error: {e}"

def plot_price_and_returns(prices):
    """Create a visualization of prices and returns.

    Args:
        prices (pd.Series): Time series of prices

    Displays:
        A figure with two subplots showing prices and log returns
    """
    try:
        # Calculate log returns from prices
        returns = np.log(prices / prices.shift(1)).dropna()
        # Create subplot figure
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        # Plot prices in top subplot
        axs[0].plot(prices, label="Price")
        axs[0].set_title("Price")
        # Plot returns in bottom subplot
        axs[1].plot(returns, label="Log Returns", color="orange")
        axs[1].set_title("Log Returns")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")


def test_cagr():
    assert abs(cagr(1000, 2000, 3) - 0.259921) < 1e-5
    assert cagr(1000, 0, 3) == "CAGR calculation error: All inputs must be positive numbers."

def test_roi():
    assert abs(roi(1500, 1000) - 0.5) < 1e-5
    assert roi(1500, 0) == "ROI calculation error: Cost of investment cannot be zero."

def test_sharpe_ratio():
    returns = [0.05, 0.10, 0.02, 0.08, 0.04]
    assert abs(sharpe_ratio(returns) - 0.707107) < 1e-5
    assert sharpe_ratio([0.01, 0.01, 0.01]) == "Sharpe Ratio error: Standard deviation of returns is zero."

def test_calculate_volatility():
    data = pd.Series([100, 105, 110, 115, 120])
    assert abs(calculate_volatility(data) - 0.046520) < 1e-5


if __name__ == "__main__":
    test_cagr()
    test_roi()
    test_sharpe_ratio()
    test_calculate_volatility()