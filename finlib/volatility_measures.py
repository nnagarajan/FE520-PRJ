import numpy as np
from statsmodels.tsa.stattools import acf


def roll_model_analysis(prices: np.ndarray) -> dict[str, float]:
    """Calculate volatility measures using Roll's model.

    Args:
        prices (np.ndarray): Array of asset prices

    Returns:
        dict[str, float]: Dictionary containing various volatility measures

    Raises:
        ValueError: If input array is empty or too short
        TypeError: If input cannot be converted to numpy array
    """
    try:
        if len(prices) < 2:
            raise ValueError("Price array must contain at least 2 values")

        # Calculate price differences
        dp = np.diff(prices)

        # Calculate autocovariance function
        covdp = acf(dp, nlags=10, fft=False, adjusted=True)

        # Extract first two autocovariance terms
        gamma0 = covdp[0]
        gamma1 = covdp[1]

        # Calculate Roll's measure of spread
        sig2u = gamma0 + 2 * gamma1

        # Calculate daily measures
        n_trades = len(dp)
        rvRoll = sig2u * n_trades
        sigRoll = np.sqrt(rvRoll)

        # Calculate average price
        av_price = np.mean(prices)

        # Calculate volatility measures
        sig_day = sigRoll
        sig_ann = np.sqrt(252) * sig_day  # Annualize using trading days
        sig_ann_ln = sig_ann / av_price  # Log-normal adjustment

        # Calculate total volatility measures
        sig_day_total = np.sqrt(gamma0 * n_trades)
        sig_ann_total = np.sqrt(252) * sig_day_total
        sig_ann_ln_total = sig_ann_total / av_price

        return {
            "Average Price": float(av_price),
            "Daily Volatility (Roll)": float(sig_day),
            "Annualized Volatility (Roll)": float(sig_ann),
            "Log-Normal Annualized Volatility (Roll)": float(sig_ann_ln),
            "Total Daily Volatility": float(sig_day_total),
            "Total Annualized Volatility": float(sig_ann_total),
            "Log-Normal Total Annualized Volatility": float(sig_ann_ln_total)
        }
    except TypeError as e:
        raise TypeError("Could not convert input to numpy array") from e


def test_roll_model_analysis():
    prices = np.array([100, 102, 101, 105, 103, 106])
    result = roll_model_analysis(prices)
    assert "Average Price" in result, "Missing 'Average Price' in result."
    assert "Daily Volatility (Roll)" in result, "Missing 'Daily Volatility (Roll)' in result."
    assert "Annualized Volatility (Roll)" in result, "Missing 'Annualized Volatility (Roll)' in result."
    assert "Log-Normal Annualized Volatility (Roll)" in result, "Missing 'Log-Normal Annualized Volatility (Roll)' in result."
    assert "Total Daily Volatility" in result, "Missing 'Total Daily Volatility' in result."
    assert "Total Annualized Volatility" in result, "Missing 'Total Annualized Volatility' in result."
    assert "Log-Normal Total Annualized Volatility" in result, "Missing 'Log-Normal Total Annualized Volatility' in result."
    print("Roll Model Analysis test passed.")

if __name__ == "__main__":
    test_roll_model_analysis()