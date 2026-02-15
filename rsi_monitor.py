#!/usr/bin/env python3
"""Streamlit-based RSI Dual-Horizon scanner with professional signal/action UI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


STOCKS_FILE = "stocks.txt"
PERIOD = "6mo"
INTERVAL = "1d"
RSI_WINDOW = 14
MA_WINDOW = 50
ATR_WINDOW = 14
SHORT_WINDOW = 5
LONG_WINDOW = 20

FIB_R1_RATIO = 0.382
FIB_R2_RATIO = 0.618
BREAKOUT_ATR_MULT = 0.15
SUPPORT_TEST_ATR_MULT = 0.10


def compute_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = ATR_WINDOW) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def read_default_tickers(file_path: Path) -> List[str]:
    if not file_path.exists():
        return []
    return [line.strip().upper() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_tickers(raw_text: str) -> List[str]:
    seen = set()
    tickers: List[str] = []
    for line in raw_text.splitlines():
        ticker = line.strip().upper()
        if not ticker:
            continue
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def normalize_ohlc(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(0):
            frame = frame[ticker]
        elif ticker in frame.columns.get_level_values(-1):
            frame = frame.xs(ticker, axis=1, level=-1)
        else:
            flat_cols = []
            for col in frame.columns:
                flat_cols.append(col[0] if isinstance(col, tuple) else str(col))
            frame = frame.copy()
            frame.columns = flat_cols

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in frame.columns for col in required):
        return pd.DataFrame()

    out = frame[required].copy()
    return out.apply(pd.to_numeric, errors="coerce").dropna()


@st.cache_data(show_spinner=False)
def fetch_one_ticker(ticker: str, period: str = PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    raw = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=False,
    )
    return normalize_ohlc(raw, ticker)


def calculate_signals(
    ohlc: pd.DataFrame,
    horizon: str = "Short",
    ticker: str | None = None,
) -> Dict[str, float | str] | None:
    if ohlc.empty or len(ohlc) < 2:
        return None

    frame = ohlc.copy()
    required_cols = {"Open", "High", "Low", "Close", "Volume"}

    # Normalize candidate column names to canonical OHLCV names.
    alias_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }

    def canonical_name(col: object) -> str:
        key = str(col).strip().replace("_", " ").lower()
        key = " ".join(key.split())
        return alias_map.get(key, str(col).strip())

    # Handle yfinance MultiIndex variations robustly by trying both common levels.
    if isinstance(frame.columns, pd.MultiIndex):
        candidate_frames: list[pd.DataFrame] = []
        for level in (0, -1):
            try:
                tmp = frame.copy()
                tmp.columns = [canonical_name(c) for c in frame.columns.get_level_values(level)]
                candidate_frames.append(tmp)
            except Exception:
                continue

        resolved = None
        for tmp in candidate_frames:
            if required_cols.issubset(set(tmp.columns)):
                resolved = tmp
                break
        if resolved is not None:
            frame = resolved
        else:
            # last resort: flatten tuples then canonicalize
            frame = frame.copy()
            frame.columns = [canonical_name(c[0] if isinstance(c, tuple) and c else c) for c in frame.columns]
    else:
        frame.columns = [canonical_name(c) for c in frame.columns]

    if frame.empty or not required_cols.issubset(set(frame.columns)):
        if ticker:
            got = ", ".join([str(c) for c in frame.columns]) if not frame.empty else "None"
            st.error(f"Missing required columns for {ticker}. Found: {got}")
        return None

    frame["RSI"] = compute_rsi(frame["Close"], RSI_WINDOW)
    frame["MA50"] = frame["Close"].rolling(window=MA_WINDOW, min_periods=MA_WINDOW).mean()
    frame["ATR"] = compute_atr(frame["High"], frame["Low"], frame["Close"], ATR_WINDOW)

    if "Volume" not in frame.columns or frame.empty:
        if ticker:
            got = ", ".join([str(c) for c in frame.columns]) if not frame.empty else "None"
            st.error(f"Missing required columns for {ticker}. Found: {got}")
        return None

    volume_avg20 = frame["Volume"].rolling(window=LONG_WINDOW, min_periods=LONG_WINDOW).mean()
    frame["Volume_Ratio"] = frame["Volume"] / volume_avg20

    latest = frame.iloc[-1]
    prev = frame.iloc[-2]

    rsi_now = latest["RSI"]
    atr_now = latest["ATR"]
    if pd.isna(rsi_now):
        return None

    price_now = float(latest["Close"])
    atr_value = 0.0 if pd.isna(atr_now) else float(atr_now)
    volume_ratio_raw = latest["Volume_Ratio"]
    volume_ratio = 1.0 if pd.isna(volume_ratio_raw) else float(volume_ratio_raw)

    prev_high = float(prev["High"])
    prev_low = float(prev["Low"])
    prev_close = float(prev["Close"])

    pp = (prev_high + prev_low + prev_close) / 3
    prev_range = prev_high - prev_low

    fib_r1 = pp + FIB_R1_RATIO * prev_range
    fib_r2 = pp + FIB_R2_RATIO * prev_range
    fib_s1 = pp - FIB_R1_RATIO * prev_range
    fib_s2 = pp - FIB_R2_RATIO * prev_range

    low_5 = frame["Low"].rolling(window=SHORT_WINDOW, min_periods=SHORT_WINDOW).min().shift(1).iloc[-1]
    high_5 = frame["High"].rolling(window=SHORT_WINDOW, min_periods=SHORT_WINDOW).max().shift(1).iloc[-1]
    low_20 = frame["Low"].rolling(window=LONG_WINDOW, min_periods=LONG_WINDOW).min().shift(1).iloc[-1]
    high_20 = frame["High"].rolling(window=LONG_WINDOW, min_periods=LONG_WINDOW).max().shift(1).iloc[-1]

    low_5_value = fib_s1 if pd.isna(low_5) else float(low_5)
    high_5_value = fib_r1 if pd.isna(high_5) else float(high_5)
    low_20_value = fib_s2 if pd.isna(low_20) else float(low_20)
    high_20_value = fib_r2 if pd.isna(high_20) else float(high_20)

    if horizon == "Long":
        resistance = max(fib_r2, high_20_value)
        support = min(fib_s2, low_20_value)
    else:
        resistance = min(fib_r1, high_5_value)
        support = max(fib_s1, low_5_value)

    breakout_buffer = atr_value * BREAKOUT_ATR_MULT
    breakout_line = resistance + breakout_buffer

    if price_now <= support:
        rr_ratio_display: float | str = "At Support"
        rr_sort = 9999.0
    elif price_now >= resistance:
        rr_ratio_display = "Target Hit"
        rr_sort = -1.0
    else:
        denominator = price_now - support
        rr_ratio_value = (resistance - price_now) / denominator
        rr_ratio_value = max(rr_ratio_value, 0.0)
        rr_ratio_display = round(rr_ratio_value, 2)
        rr_sort = rr_ratio_value

    is_high_value = (
        (
            rr_ratio_display == "At Support"
            or (isinstance(rr_ratio_display, (int, float)) and rr_ratio_display > 2.0)
        )
        and float(rsi_now) < 45
    )

    # Split Signal and Action (no emojis)
    if price_now > breakout_line and volume_ratio > 1.2:
        signal, action = "Strong Breakout", "Enter Long"
    elif price_now > breakout_line and volume_ratio <= 1.2:
        signal, action = "Weak Breakout", "Watch for Trap"
    elif is_high_value:
        signal, action = "High Value Opportunity", "Accumulate"
    elif float(rsi_now) < 30:
        signal, action = "Oversold", "Prepare to Buy"
    elif float(rsi_now) > 70:
        signal, action = "Overbought", "Take Profit"
    elif price_now < support + (SUPPORT_TEST_ATR_MULT * atr_value):
        signal, action = "Support Test", "Monitor Support"
    else:
        signal, action = "Neutral", "Wait"

    return {
        "Price": round(price_now, 2),
        "Support": round(support, 2),
        "Resistance": round(resistance, 2),
        "RSI": round(float(rsi_now), 2),
        "ATR": round(atr_value, 2),
        "Volume_Ratio": round(volume_ratio, 2),
        "RR_Ratio": rr_ratio_display,
        "RR_Sort": round(rr_sort, 4),
        "Signal": signal,
        "Action": action,
        "PP": round(pp, 2),
        "Fib_R1": round(fib_r1, 2),
        "Fib_R2": round(fib_r2, 2),
        "Fib_S1": round(fib_s1, 2),
        "Fib_S2": round(fib_s2, 2),
        "High_5": round(high_5_value, 2),
        "Low_5": round(low_5_value, 2),
        "High_20": round(high_20_value, 2),
        "Low_20": round(low_20_value, 2),
        "Breakout_Buffer": round(breakout_buffer, 2),
        "Breakout_Line": round(breakout_line, 2),
    }


def analyze_ticker(
    ticker: str,
    horizon: str = "Short",
) -> Tuple[Optional[Dict[str, float | str]], Optional[pd.DataFrame], Optional[str]]:
    try:
        ohlc = fetch_one_ticker(ticker)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{ticker}: data download failed ({exc})"

    if ohlc.empty:
        return None, None, f"{ticker}: missing valid OHLC data"

    signals = calculate_signals(ohlc, horizon=horizon, ticker=ticker)
    if signals is None:
        return None, None, f"{ticker}: insufficient data for indicators"

    ohlc = ohlc.copy()
    ohlc["RSI"] = compute_rsi(ohlc["Close"], RSI_WINDOW)
    ohlc["MA50"] = ohlc["Close"].rolling(window=MA_WINDOW, min_periods=MA_WINDOW).mean()

    result = {
        "Ticker": ticker,
        "Price": signals["Price"],
        "Support": signals["Support"],
        "Resistance": signals["Resistance"],
        "RSI": signals["RSI"],
        "RR_Ratio": signals["RR_Ratio"],
        "RR_Sort": signals["RR_Sort"],
        "Signal": signals["Signal"],
        "Action": signals["Action"],
        "ATR": signals["ATR"],
        "Volume_Ratio": signals["Volume_Ratio"],
        "PP": signals["PP"],
        "Fib_R1": signals["Fib_R1"],
        "Fib_R2": signals["Fib_R2"],
        "Fib_S1": signals["Fib_S1"],
        "Fib_S2": signals["Fib_S2"],
        "High_5": signals["High_5"],
        "Low_5": signals["Low_5"],
        "High_20": signals["High_20"],
        "Low_20": signals["Low_20"],
        "Breakout_Buffer": signals["Breakout_Buffer"],
        "Breakout_Line": signals["Breakout_Line"],
    }

    return result, ohlc, None


def style_results_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def color_signal_action(value: str) -> str:
        text = str(value).strip()

        green = "rgba(0, 200, 83, 0.2)"
        yellow = "rgba(255, 179, 0, 0.2)"
        blue = "rgba(3, 169, 244, 0.2)"
        red = "rgba(255, 82, 82, 0.2)"

        if text in {"Strong Breakout", "High Value Opportunity", "Enter Long", "Accumulate"}:
            return f"background-color: {green}; border: 1px solid {green}; font-weight: 700;"
        if text in {"Weak Breakout", "Watch for Trap"}:
            return f"background-color: {yellow}; border: 1px solid {yellow}; font-weight: 700;"
        if text in {"Oversold", "Prepare to Buy", "Support Test", "Monitor Support"}:
            return f"background-color: {blue}; border: 1px solid {blue}; font-weight: 700;"
        if text in {"Overbought", "Take Profit"}:
            return f"background-color: {red}; border: 1px solid {red}; font-weight: 700;"
        return ""

    def format_rr(value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.2f}"
        return str(value)

    return (
        df.style
        .format(
            {
                "Price": "{:.2f}",
                "Support": "{:.2f}",
                "Resistance": "{:.2f}",
                "RSI": "{:.2f}",
                "RR_Ratio": format_rr,
            }
        )
        .set_properties(subset=df.columns.tolist(), **{"text-align": "center"})
        .set_properties(subset=["Signal", "Action"], **{"font-weight": "700"})
        .applymap(color_signal_action, subset=["Signal", "Action"])
    )


def plot_ticker_detail(ticker: str, ohlc: pd.DataFrame, support: float, resistance: float) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{ticker} Price", "RSI(14)"),
    )

    fig.add_trace(
        go.Candlestick(
            x=ohlc.index,
            open=ohlc["Open"],
            high=ohlc["High"],
            low=ohlc["Low"],
            close=ohlc["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=ohlc.index, y=ohlc["MA50"], mode="lines", line=dict(color="#1f77b4", width=2), name="MA50"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ohlc.index,
            y=[support] * len(ohlc),
            mode="lines",
            line=dict(color="#2ca02c", width=1.8, dash="dash"),
            name="Support",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ohlc.index,
            y=[resistance] * len(ohlc),
            mode="lines",
            line=dict(color="#d62728", width=1.8, dash="dash"),
            name="Resistance",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=ohlc.index, y=ohlc["RSI"], mode="lines", line=dict(color="#ff7f0e", width=2), name="RSI(14)"),
        row=2,
        col=1,
    )

    fig.add_hline(y=70, line=dict(color="#d62728", dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#2ca02c", dash="dot"), row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=760,
        xaxis_rangeslider_visible=False,
        margin=dict(l=24, r=24, t=42, b=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    return fig


def main() -> None:
    st.set_page_config(page_title="Sift Terminal", layout="wide")

    st.title("Sift Terminal")
    st.markdown(
        "Dual-Horizon RSI scanner with Fibonacci pivots, breakout filters, and action-oriented signal mapping."
    )

    script_dir = Path(__file__).resolve().parent
    default_tickers = read_default_tickers(script_dir / STOCKS_FILE)
    default_text = "\n".join(default_tickers)

    st.sidebar.header("Scan Console")
    tickers_text = st.sidebar.text_area(
        "Tickers (one per line)",
        value=default_text,
        height=260,
        help="You can add/remove tickers manually. Empty lines are ignored.",
    )

    horizon_label = st.sidebar.radio(
        "Time Horizon",
        options=["Short-term (5D)", "Long-term (20D)"],
        index=0,
    )
    horizon = "Short" if horizon_label == "Short-term (5D)" else "Long"

    run_scan = st.sidebar.button("Run Scan", type="primary")

    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    if "history_map" not in st.session_state:
        st.session_state.history_map = {}
    if "warnings" not in st.session_state:
        st.session_state.warnings = []
    if "horizon" not in st.session_state:
        st.session_state.horizon = horizon

    if run_scan:
        tickers = parse_tickers(tickers_text)
        if not tickers:
            st.error("Please provide at least one valid ticker.")
            return

        rows: List[Dict[str, float | str]] = []
        history_map: Dict[str, pd.DataFrame] = {}
        warnings: List[str] = []

        with st.spinner("Running market scan..."):
            for ticker in tickers:
                row, history, warn = analyze_ticker(ticker, horizon=horizon)
                if warn:
                    warnings.append(warn)
                    continue
                if row is None or history is None:
                    continue
                rows.append(row)
                history_map[ticker] = history

        if not rows:
            st.error("No valid results. Please check symbols or try again later.")
            if warnings:
                for warn in warnings:
                    st.warning(warn)
            return

        results_df = pd.DataFrame(rows).sort_values(by=["RR_Sort", "Ticker"], ascending=[False, True]).reset_index(drop=True)

        st.session_state.results_df = results_df
        st.session_state.history_map = history_map
        st.session_state.warnings = warnings
        st.session_state.horizon = horizon

    results_df = st.session_state.results_df
    history_map = st.session_state.history_map
    warnings = st.session_state.warnings

    if results_df is None:
        st.info("Configure tickers, choose time horizon, and click 'Run Scan' in the sidebar.")
        return

    oversold_count = int((results_df["RSI"] < 35).sum())
    overbought_count = int((results_df["RSI"] > 70).sum())

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    c1.metric("Scanned Tickers", len(results_df))
    c2.metric("Oversold (<35)", oversold_count)
    c3.metric("Overbought (>70)", overbought_count)
    c4.metric("Horizon", st.session_state.horizon)

    st.subheader("Scan Results")
    display_df = results_df[["Ticker", "Price", "Support", "Resistance", "RSI", "RR_Ratio", "Signal", "Action"]].copy()
    with st.container():
        styled_df = style_results_table(display_df)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    if warnings:
        with st.expander("Skipped / Warning Details"):
            for warn in warnings:
                st.warning(warn)

    st.subheader("Ticker Detail")
    selected = st.selectbox("Select a ticker", options=results_df["Ticker"].tolist())

    selected_row = results_df[results_df["Ticker"] == selected].iloc[0]
    selected_history = history_map[selected]

    tab_chart, tab_advanced = st.tabs(["Chart", "Advanced"])

    with tab_chart:
        fig = plot_ticker_detail(
            ticker=selected,
            ohlc=selected_history,
            support=float(selected_row["Support"]),
            resistance=float(selected_row["Resistance"]),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_advanced:
        with st.expander("Show advanced metrics and formulas", expanded=False):
            metrics_col1, metrics_col2, metrics_col3 = st.columns([1, 1, 1])
            metrics_col1.metric("ATR", f"{float(selected_row['ATR']):.2f}")
            metrics_col2.metric("Volume Ratio", f"{float(selected_row['Volume_Ratio']):.2f}")
            rr_text = selected_row["RR_Ratio"] if isinstance(selected_row["RR_Ratio"], str) else f"{float(selected_row['RR_Ratio']):.2f}"
            metrics_col3.metric("RR Ratio", rr_text)

            st.markdown("**Calculation Notes**")
            st.markdown(
                "- `PP = (H_prev + L_prev + C_prev) / 3`\n"
                "- `Fib_R1 = PP + 0.382 * Range`, `Fib_R2 = PP + 0.618 * Range`\n"
                "- `Fib_S1 = PP - 0.382 * Range`, `Fib_S2 = PP - 0.618 * Range`\n"
                "- Short horizon: `Resistance = min(Fib_R1, High_5)`, `Support = max(Fib_S1, Low_5)`\n"
                "- Long horizon: `Resistance = max(Fib_R2, High_20)`, `Support = min(Fib_S2, Low_20)`\n"
                "- `Volume_Ratio = Volume / AvgVolume_20`\n"
                "- `Breakout Buffer = 0.15 * ATR`\n"
                "- `RR_Ratio = (Resistance - Price) / (Price - Support)` when Price is between Support and Resistance\n"
                "- `At Support`: Price <= Support, indicating downside risk is compressed near support\n"
                "- `Target Hit`: Price >= Resistance, indicating price has reached/exceeded the modeled target zone"
            )

            debug_df = pd.DataFrame(
                [
                    {
                        "PP": selected_row["PP"],
                        "Fib_R1": selected_row["Fib_R1"],
                        "Fib_R2": selected_row["Fib_R2"],
                        "Fib_S1": selected_row["Fib_S1"],
                        "Fib_S2": selected_row["Fib_S2"],
                        "High_5": selected_row["High_5"],
                        "Low_5": selected_row["Low_5"],
                        "High_20": selected_row["High_20"],
                        "Low_20": selected_row["Low_20"],
                        "Breakout_Buffer": selected_row["Breakout_Buffer"],
                        "Breakout_Line": selected_row["Breakout_Line"],
                    }
                ]
            )
            st.dataframe(debug_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
