import numpy as np
import math as math
from scipy.optimize import minimize
from scipy.special import gammaln

def pin_ekop(buys, sells):
    """
    Calculates the probability of informed trading (PIN) using the Easley, Kiefer, O'Hara, and Paperman (EKOP) model.

    Args:
        buys (np.ndarray): Array of daily buy volumes.
        sells (np.ndarray): Array of daily sell volumes.

    Returns:
        float: The estimated PIN value.
    """

    def neg_log_likelihood(params, buys, sells):
        alpha, mu, epsilon_b, epsilon_s = params
        ll = 0
        for b, s in zip(buys, sells):
            if alpha == 0 or mu == 0 or epsilon_b == 0 or epsilon_s == 0:
                continue  # Avoid log(0) or divide by zero
            term1 = np.exp(-mu - epsilon_b) * (mu + epsilon_b) ** b / np.exp(gammaln(b + 1))
            term2 = np.exp(-epsilon_s) * epsilon_s ** s / np.exp(gammaln(s + 1))
            term3 = np.exp(-epsilon_b) * epsilon_b ** b / np.exp(gammaln(b + 1))
            term4 = np.exp(-mu - epsilon_s) * (mu + epsilon_s) ** s / np.exp(gammaln(s + 1))
            ll += np.log(
                alpha * (term1 * term2 + term3 * term4) + (1 - alpha) * (term3 * term2)
            )
        return -ll

    # Initial guesses for parameters
    params_init = np.array([0.2, 1, 1, 1])

    # Bounds for parameters
    bnds = ((0, 1), (0, None), (0, None), (0, None))

    # Optimization
    result = minimize(neg_log_likelihood, params_init, args=(buys, sells), bounds=bnds, method='L-BFGS-B')

    # Extract estimated parameters
    alpha_hat, mu_hat, epsilon_b_hat, epsilon_s_hat = result.x

    # Calculate PIN
    pin = alpha_hat * mu_hat / (alpha_hat * mu_hat + epsilon_b_hat + epsilon_s_hat)

    return pin


def test_pin_ekop():
    # Simple test case
    buys = np.array([5, 10, 15])
    sells = np.array([6, 9, 14])
    pin_value = pin_ekop(buys, sells)
    expected_pin_range = (0, 1)
    assert expected_pin_range[0] <= pin_value <= expected_pin_range[1], \
        f"Expected PIN to be in range {expected_pin_range}, but got {pin_value}"
    print(f"Test passed. PIN value: {pin_value}")


if __name__ == "__main__":
    test_pin_ekop()
