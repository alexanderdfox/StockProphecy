import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === Parameters ===
ticker = "APPL"
start_date = "1980-12-12"
end_date = "2025-07-31"  # Up to now, you can update this dynamically
strike_price = None  # Will set after fetching price
risk_free_rate = 0.03  # Annual risk-free rate approx 3%
future_years = 10
entropy_window = 10  # Rolling window size for entropy calc
entropy_percentile = 20  # Threshold percentile for demon action

# === Fetch historical data from MarketData.app API ===
url = f"https://api.marketdata.app/v1/stocks/candles/D/{ticker}?from={start_date}&to={end_date}"
resp = requests.get(url)
data = resp.json()

if data["s"] != "ok":
	raise RuntimeError(f"Error fetching data: {data}")

prices = np.array(data["c"])  # Closing prices

# Use last available price as strike if not set
if strike_price is None:
	strike_price = prices[-1]

# === Calculate log returns ===
log_returns = np.diff(np.log(prices))

# === Calculate annualized drift and volatility ===
trading_days_per_year = 252
mu = np.mean(log_returns) * trading_days_per_year
sigma = np.std(log_returns) * np.sqrt(trading_days_per_year)

# === Simulate future prices with GBM ===
dt = 1 / trading_days_per_year
N_future = int(future_years * trading_days_per_year)
np.random.seed(42)
Z = np.random.standard_normal(N_future)
future_prices = [prices[-1]]
for z in Z:
	S_next = future_prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
	future_prices.append(S_next)
future_prices = np.array(future_prices[1:])

# === Combine historical + future prices ===
full_prices = np.concatenate([prices, future_prices])

# === Black-Scholes formula for call option ===
def black_scholes_call(S, K, T, r, sigma):
	if T <= 0:
		return max(S - K, 0)
	d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
	d2 = d1 - sigma * np.sqrt(T)
	return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

bs_price_start = black_scholes_call(prices[-1], strike_price, future_years, risk_free_rate, sigma)
print(f"Black-Scholes price at start (t=0): ${bs_price_start:.2f}")

# === Compute entropy of price changes over rolling windows ===
def shannon_entropy(arr):
	hist, _ = np.histogram(arr, bins=20, density=True)
	hist = hist[hist > 0]
	return -np.sum(hist * np.log(hist))

rolling_entropy = []
for i in range(entropy_window, len(full_prices)):
	window = np.diff(full_prices[i - entropy_window : i + 1])
	H = shannon_entropy(window)
	rolling_entropy.append(H)
rolling_entropy = np.array(rolling_entropy)

# === Determine entropy threshold from historical data only ===
historical_entropy = rolling_entropy[: len(prices) - entropy_window]
threshold = np.percentile(historical_entropy, entropy_percentile)

print(f"Entropy threshold (at {entropy_percentile} percentile): {threshold:.4f}")

# === Maxwell's Demon selective hedging ===
holding = 0.0
demon_portfolio_value = 0.0

for i in range(len(prices), len(full_prices)):
	T_rem = (len(full_prices) - 1 - i) / trading_days_per_year
	if T_rem <= 0:
		break

	entropy_now = rolling_entropy[i - entropy_window]
	if entropy_now < threshold:
		delta = norm.cdf(
			(np.log(full_prices[i] / strike_price) + (risk_free_rate + 0.5 * sigma**2) * T_rem)
			/ (sigma * np.sqrt(T_rem))
		)
		holding = delta
		demon_portfolio_value = holding * full_prices[i] - delta * strike_price * np.exp(-risk_free_rate * T_rem)

# === Calculate final demon profit ===
final_payoff = max(full_prices[-1] - strike_price, 0) - demon_portfolio_value

print(f"Demon's estimated final profit: ${final_payoff:.2f}")

# === Plot price and demon entropy and hedging points ===
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(full_prices, label="AAPL Price (Hist + Sim)")
plt.axhline(strike_price, color="gray", linestyle="--", label="Strike Price")
plt.title("AAPL Stock Price: Historical + 10-Year Simulated Future")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(range(entropy_window, len(full_prices)), rolling_entropy, label="Rolling Entropy")
plt.axhline(threshold, color="red", linestyle="--", label="Entropy Threshold")
plt.title("Rolling Entropy of Price Changes")
plt.xlabel("Time (days)")
plt.ylabel("Entropy")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
