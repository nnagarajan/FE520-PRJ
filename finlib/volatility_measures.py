import numpy as np
from statsmodels.tsa.stattools import acf


def roll_model_analysis(prices):
    dp = np.diff(prices)

    covdp = acf(dp, nlags=10, fft=False, adjusted=True)

    gamma0 = covdp[0]
    gamma1 = covdp[1]

    sig2u = gamma0 + 2 * gamma1

    n_trades = len(dp)
    rvRoll = sig2u * n_trades
    sigRoll = np.sqrt(rvRoll)

    av_price = np.mean(prices)

    sig_day = sigRoll
    sig_ann = np.sqrt(252) * sig_day
    sig_ann_ln = sig_ann / av_price

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