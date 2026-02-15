# Sift: Technical Signal Tracker

A terminal-based quantitative tool developed to identify market opportunities using **Pivot Point** structures and **RSI** momentum.
Disclaimer: This tool provides data-driven signals only, not investment advice.

### 🔗 [Click Here to Access][1]
*(No Python or installation required. Access directly from your browser!)*
## Built With
- Streamlit: Web Application Framework
- yFinance: Market Data API
- Pandas: Data Manipulation
- Plotly: High-performance Interactive Charting

## What This Project Does

- Reads price data for an asset (for example, a stock or crypto pair)
- Calculates RSI for a selected period (commonly `14`)
- Generates advice such as `Buy`, `Sell`, or `Hold`
- Prints or logs values so you can monitor momentum over time

## Local Development

If you want to run **Sift** on your own machine or inspect the source code:

1. Prerequisites
Ensure you have Python 3.9 or higher installed on your system.

```
bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/Sift-RSI.git](https://github.com/YOUR_USERNAME/Sift-RSI.git)
cd "Sift [RSI]"

# Install dependencies
python3 -m pip install -r requirements.txt

# Launch the app
python3 -m streamlit run rsi_monitor.py
```

## How RSI Is Calculated

RSI measures the speed and size of recent price moves.

### Formula

For period `N` (usually 14):

1. Compute price changes (`delta`) between consecutive closes
2. Split into:
	1. `Gain`: positive deltas (else 0)
	2. `Loss`: absolute value of negative deltas (else 0)
3. Compute averages:
	1. `Average Gain`
	2. `Average Loss`
4. Compute relative strength:

```text
RS = Average Gain / Average Loss
```

5. Compute RSI:

```text
RSI = 100 - (100 / (1 + RS))
```

### Wilder Smoothing (common approach)

After the first average, many RSI implementations use Wilder smoothing:

```text
AvgGain_t = (AvgGain_(t-1) * (N - 1) + Gain_t) / N
AvgLoss_t = (AvgLoss_(t-1) * (N - 1) + Loss_t) / N
```

This reduces noise compared to using a simple rolling average every time.

## What the Advice Means

Typical thresholds:

- `RSI >= 70`: **Overbought** zone
- `RSI <= 30`: **Oversold** zone
- `30 < RSI < 70`: neutral/mid zone

Advice mapping (common default):

- `Sell` or `Take Profit`: RSI is high (often `>= 70`), momentum may be overheated
- `Buy` or `Watch for Entry`: RSI is low (often `<= 30`), price may be stretched down
- `Hold`: RSI is in the middle range, no strong edge from RSI alone

Important: RSI is **not a guarantee**. Strong trends can keep RSI overbought/oversold for long periods.

## Term Glossary

- `RSI (Relative Strength Index)`: Momentum oscillator from `0` to `100`
- `Period (N)`: Number of candles used in the RSI calculation (default `14`)
- `Delta`: Current close minus previous close
- `Gain`: Positive delta value, otherwise `0`
- `Loss`: Absolute value of negative delta, otherwise `0`
- `Average Gain/Loss`: Smoothed recent gain/loss values
- `RS (Relative Strength)`: `Average Gain / Average Loss`
- `Overbought`: RSI high region (often above `70`)
- `Oversold`: RSI low region (often below `30`)
- `Signal`: The generated advice (`Buy`, `Sell`, `Hold`)


## Suggested Risk Notes

Use RSI with confirmation:

- Trend direction (higher timeframe)
- Support/resistance levels
- Volume
- Stop-loss and position sizing

## Author
Cayden Zhang 
Freshman @ Georgetown University


[1]:	https://sift-terminal-4p3sy2ex8swfxuhhqsbw7f.streamlit.app/
