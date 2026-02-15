#!/usr/bin/env python3
"""Streamlit-based RSI + Pivot Points scanner."""

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


def compute_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


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
                if isinstance(col, tuple):
                    flat_cols.append(col[0])
                else:
                    flat_cols.append(str(col))
            frame = frame.copy()
            frame.columns = flat_cols

    required = ["Open", "High", "Low", "Close"]
    if not all(col in frame.columns for col in required):
        return pd.DataFrame()

    out = frame[required].copy()
    out = out.apply(pd.to_numeric, errors="coerce").dropna()
    return out


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


def classify_signal(price_now: float, price_prev: float, support: float, resistance: float, rsi_now: float, rsi_prev: float) -> str:
    rsi_rebound = rsi_prev < 30 <= rsi_now
    support_reclaim = price_prev < support <= price_now
    up_breakout = price_now > resistance

    if rsi_now < 30 and price_now < support:
        return "Strong Buy / Oversold"
    if rsi_now < 35:
        return "Oversold Alert"
    if support_reclaim or rsi_rebound:
        return "Reversal Confirmed"
    if rsi_now > 75 and price_now > resistance:
        return "High Risk"
    if up_breakout:
        return "Breakout"
    if rsi_now > 70:
        return "Overbought Warning"
    if 40 <= rsi_now <= 60 and not up_breakout:
        return "Neutral / Waiting"
    return "Reversal Watch"


def analyze_ticker(ticker: str) -> Tuple[Optional[Dict[str, float | str]], Optional[pd.DataFrame], Optional[str]]:
    try:
        ohlc = fetch_one_ticker(ticker)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{ticker}: data download failed ({exc})"

    if ohlc.empty:
        return None, None, f"{ticker}: missing valid OHLC data"

    if len(ohlc) < 2:
        return None, None, f"{ticker}: missing yesterday OHLC data"

    ohlc = ohlc.copy()
    ohlc["RSI"] = compute_rsi(ohlc["Close"], RSI_WINDOW)
    ohlc["MA50"] = ohlc["Close"].rolling(window=MA_WINDOW, min_periods=MA_WINDOW).mean()

    rsi_now = ohlc["RSI"].iloc[-1]
    rsi_prev = ohlc["RSI"].iloc[-2]
    if pd.isna(rsi_now) or pd.isna(rsi_prev):
        return None, None, f"{ticker}: RSI history insufficient"

    latest = ohlc.iloc[-1]
    prev = ohlc.iloc[-2]

    price_now = float(latest["Close"])
    price_prev = float(prev["Close"])

    prev_high = float(prev["High"])
    prev_low = float(prev["Low"])
    prev_close = float(prev["Close"])

    pp = (prev_high + prev_low + prev_close) / 3
    support = (2 * pp) - prev_high
    resistance = (2 * pp) - prev_low

    signal = classify_signal(
        price_now=price_now,
        price_prev=price_prev,
        support=support,
        resistance=resistance,
        rsi_now=float(rsi_now),
        rsi_prev=float(rsi_prev),
    )

    result = {
        "Ticker": ticker,
        "Price": round(price_now, 2),
        "Support": round(support, 2),
        "Resistance": round(resistance, 2),
        "RSI": round(float(rsi_now), 2),
        "Signal": signal,
    }

    return result, ohlc, None


def style_signal_column(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def color_signal(value: str) -> str:
        upper = value.upper()
        if "BUY" in upper or "OVERSOLD" in upper or "REVERSAL" in upper:
            return "background-color: #1f8b4c; color: white; font-weight: 600;"
        if "RISK" in upper or "OVERBOUGHT" in upper:
            return "background-color: #b22222; color: white; font-weight: 600;"
        if "BREAKOUT" in upper:
            return "background-color: #9c7a00; color: white; font-weight: 600;"
        return ""

    return (
        df.style
        .format({"Price": "{:.2f}", "Support": "{:.2f}", "Resistance": "{:.2f}", "RSI": "{:.2f}"})
        .applymap(color_signal, subset=["Signal"])
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
        go.Scatter(
            x=ohlc.index,
            y=ohlc["MA50"],
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            name="MA50",
        ),
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
        go.Scatter(
            x=ohlc.index,
            y=ohlc["RSI"],
            mode="lines",
            line=dict(color="#ff7f0e", width=2),
            name="RSI(14)",
        ),
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
        "Professional RSI + Pivot scanner powered by Yahoo Finance. "
        "Adjust tickers in the sidebar, run a scan, and inspect each symbol with interactive charts."
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
    run_scan = st.sidebar.button("Run Scan", type="primary")

    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    if "history_map" not in st.session_state:
        st.session_state.history_map = {}
    if "warnings" not in st.session_state:
        st.session_state.warnings = []

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
                row, history, warn = analyze_ticker(ticker)
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

        results_df = pd.DataFrame(rows).sort_values(by=["Signal", "Ticker"]).reset_index(drop=True)
        st.session_state.results_df = results_df
        st.session_state.history_map = history_map
        st.session_state.warnings = warnings

    results_df = st.session_state.results_df
    history_map = st.session_state.history_map
    warnings = st.session_state.warnings

    if results_df is None:
        st.info("Configure tickers and click 'Run Scan' in the sidebar.")
        return

    oversold_count = int((results_df["RSI"] < 35).sum())
    overbought_count = int((results_df["RSI"] > 70).sum())

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Scanned Tickers", len(results_df))
    c2.metric("Oversold (<35)", oversold_count)
    c3.metric("Overbought (>70)", overbought_count)

    st.subheader("Scan Results")
    st.dataframe(style_signal_column(results_df), use_container_width=True, hide_index=True)

    if warnings:
        with st.expander("Skipped / Warning Details"):
            for warn in warnings:
                st.warning(warn)

    st.subheader("Ticker Detail")
    selected = st.selectbox("Select a ticker", options=results_df["Ticker"].tolist())

    selected_row = results_df[results_df["Ticker"] == selected].iloc[0]
    selected_history = history_map[selected]

    fig = plot_ticker_detail(
        ticker=selected,
        ohlc=selected_history,
        support=float(selected_row["Support"]),
        resistance=float(selected_row["Resistance"]),
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
