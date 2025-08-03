import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === Parameters ===
crypto_id = "bitcoin"  # CoinGecko id for Bitcoin
vs_currency = "usd"
days = "365"  # max allowed free API range (last 365 days)
future_years = 10
entropy_window = 10  # Rolling window size for entropy calc
entropy_percentile = 20  # Threshold percentile for demon action
risk_free_rate = 0.03  # Risk free rate (for analogy), annual
trading_days_per_year = 365  # Crypto trades every day

# === Fetch historical data from CoinGecko ===
url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency={vs_currency}&days={days}"
resp = requests.get(url)

print(f"API Response Status Code: {resp.status_code}")
print(f"Response Preview: {resp.text[:500]}")  # print first 500 chars for debugging

if resp.status_code != 200:
    raise RuntimeError(f"API request failed with status {resp.status_code}: {resp.text}")

data = resp.json()

if "prices" not in data:
    raise KeyError(f"'prices' key not found in response JSON. Response keys: {list(data.keys())}")

prices = np.array([p[1] for p in data["prices"]])

# === Calculate log returns ===
log_returns = np.diff(np.log(prices))

# === Calculate annualized drift and volatility ===
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

# === Shannon entropy calculation ===
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

# === Maxwell's Demon selective signal example ===
for i in range(len(prices), len(full_prices)):
    entropy_now = rolling_entropy[i - entropy_window]
    if entropy_now < threshold:
        print(f"Day {i}: Low entropy detected ({entropy_now:.4f}), consider action")

# === Plot price and entropy ===
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(full_prices, label=f"{crypto_id.capitalize()} Price (Hist + Sim)")
plt.title(f"{crypto_id.capitalize()} Price: Historical + {future_years}-Year Simulated Future")
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
