import os
import pandas as pd
import yfinance as yf
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
# ------------- Helper to format percentages -------------


def format_pct(x):
    """Format a decimal return like 0.0123 as '1.23%' and handle NaN."""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "N/A"
        return f"{x * 100:.2f}%"
    except Exception:
        return "N/A"

# -------------------------
# NEWS API SETTINGS + HELPER
# -------------------------

NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"


def get_news_for_ticker(ticker: str, from_date: datetime.date, to_date: datetime.date, max_articles: int = 2):
    """
    Fetch a few recent news headlines for a given ticker using NewsAPI.

    Returns list of dictionaries with fields: title, source, url, published_at.
    Returns [] if key is missing or API error occurs.
    """
    if not NEWS_API_KEY:
        print("NEWS_API_KEY not set, skipping news fetch.")
        return []

    query = f"{ticker} stock OR shares OR earnings OR company"
    params = {
        "q": query,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
    }

    try:
        resp = requests.get(NEWS_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"News error for {ticker}: {e}")
        return []

    data = resp.json() or {}
    articles = data.get("articles", [])[:max_articles]

    results = []
    for a in articles:
        results.append(
            {
                "title": a.get("title"),
                "source": (a.get("source") or {}).get("name"),
                "url": a.get("url"),
                "published_at": a.get("publishedAt"),
            }
        )

    return results


# ============== CONFIG (tickers, universe file) ===================

MACRO_TICKERS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Russell 2000": "IWM",
    "VIX": "^VIX",
    "US 10Y Yield": "^TNX",
    "DXY Dollar Index": "DX-Y.NYB",
}

SECTOR_ETFS = {
    "Energy": "XLE",
    "Technology": "XLK",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Health Care": "XLV",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

COMMODITY_TICKERS = {
    "WTI Crude": "CL=F",
    "Brent Crude": "BZ=F",
    "Natural Gas": "NG=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
}

# Your watchlist CSV (same as before)
UNIVERSE_FILE = "sectors_tickers.csv"

# News config – keys come from environment variables (GitHub secrets)
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

# Email config – again from environment variables
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
TO_EMAIL = os.environ.get("TO_EMAIL", EMAIL_USER)
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))


# ============== DATA HELPERS ======================================

def get_price_data(tickers, period="60d"):
    data = yf.download(tickers, period=period)["Close"]
    return data.dropna(how="all")


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty or price_df.shape[0] < 21:
        return pd.DataFrame(columns=["return_1d", "return_5d", "return_20d"])

    returns_1d = price_df.pct_change(1).iloc[-1]
    returns_5d = price_df.pct_change(5).iloc[-1]
    returns_20d = price_df.pct_change(20).iloc[-1]

    out = pd.DataFrame(
        {
            "return_1d": returns_1d,
            "return_5d": returns_5d,
            "return_20d": returns_20d,
        }
    )
    out.index.name = "ticker"
    return out


def safe_returns(tickers, period="60d"):
    try:
        prices = get_price_data(tickers, period=period)
    except Exception as e:
        print(f"Price download error for {tickers}: {e}")
        return pd.DataFrame()
    return compute_returns(prices)


def load_universe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def generate_watchlist_signals(merged_universe: pd.DataFrame):
    """
    Build simple long/short idea lists from the merged_universe table.

    merged_universe is expected to have at least:
    - sector
    - ticker
    - return_1d
    - return_5d
    - return_20d
    """

    longs = []
    shorts = []

    required_cols = {"sector", "ticker", "return_1d", "return_5d", "return_20d"}
    if not required_cols.issubset(merged_universe.columns):
        print("generate_watchlist_signals: missing columns, skipping signal generation.")
        return pd.DataFrame(), pd.DataFrame()

    for _, row in merged_universe.iterrows():
        r1 = row["return_1d"]
        r5 = row["return_5d"]
        r20 = row["return_20d"]
        sector = row["sector"]
        ticker = row["ticker"]

        # Skip rows with missing data
        if pd.isna(r1) or pd.isna(r5) or pd.isna(r20):
            continue

        # ---- Simple example rules (we can tweak later) ----
        # Long idea: strong short-term momentum, not deeply negative on 20d
        if r1 > 0.01 and r5 > 0.03 and r20 > -0.02:
            longs.append(
                {
                    "sector": sector,
                    "ticker": ticker,
                    "return_1d": r1,
                    "return_5d": r5,
                    "return_20d": r20,
                    "signal": "Momentum long",
                }
            )

        # Short idea: consistently weak across horizons
        if r1 < -0.01 and r5 < -0.03 and r20 < -0.05:
            shorts.append(
                {
                    "sector": sector,
                    "ticker": ticker,
                    "return_1d": r1,
                    "return_5d": r5,
                    "return_20d": r20,
                    "signal": "Momentum short",
                }
            )

    # Turn lists into DataFrames
    long_df = pd.DataFrame(longs)
    short_df = pd.DataFrame(shorts)

    # Handle empty cases safely
    if long_df.empty:
        print("No long signals generated for today.")
    else:
        if "return_20d" in long_df.columns:
            long_df = long_df.sort_values("return_20d", ascending=False)

    if short_df.empty:
        print("No short signals generated for today.")
    else:
        if "return_20d" in short_df.columns:
            short_df = short_df.sort_values("return_20d", ascending=True)

    return long_df, short_df


def compute_breadth(uni_returns: pd.DataFrame):
    if uni_returns.empty:
        return None
    pos = (uni_returns["return_20d"] > 0).sum()
    neg = (uni_returns["return_20d"] <= 0).sum()
    total = pos + neg
    if total == 0:
        return None
    return pos / total


def compute_risk_score(macro_df: pd.DataFrame) -> str:
    """
    Very simple risk-on / risk-off text based on:
    - SPY 20D
    - VIX 20D
    - DXY 20D
    """
    if macro_df.empty:
        return "Macro: data unavailable."

    def get_ret(name):
        row = macro_df[macro_df["name"] == name]
        if row.empty:
            return None
        return float(row["return_20d"].iloc[0])

    spy = get_ret("S&P 500")
    vix = get_ret("VIX")
    dxy = get_ret("DXY Dollar Index")

    if spy is None or vix is None or dxy is None:
        return "Macro: partial data, interpret with caution."

    text = []

    if spy > 0.03 and vix < 0 and dxy <= 0:
        text.append("Overall **risk-on**: equities trending higher, volatility easing, dollar stable/weaker.")
    elif spy < -0.03 and vix > 0 and dxy >= 0:
        text.append("Overall **risk-off**: equities weak, volatility rising, dollar firming.")
    else:
        text.append("Macro mixed: no clear regime, be selective and focus on stock/sector specifics.")

    text.append(f"S&P 500 20D: {format_pct(spy)}, VIX 20D: {format_pct(vix)}, DXY 20D: {format_pct(dxy)}")
    return " ".join(text)


# ============== DASHBOARD BUILDER =================================

def build_dashboard_html():
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)

    # --- Macro ---
    macro_returns = safe_returns(list(MACRO_TICKERS.values()), period="60d")
    if not macro_returns.empty:
        rev_macro = {v: k for k, v in MACRO_TICKERS.items()}
        macro_returns = macro_returns.rename_axis("ticker")
        macro_returns["name"] = macro_returns.index.map(rev_macro)
        macro_returns = macro_returns.reset_index()

    # --- Commodities ---
    commodity_returns = safe_returns(list(COMMODITY_TICKERS.values()), period="60d")
    if not commodity_returns.empty:
        rev_cmd = {v: k for k, v in COMMODITY_TICKERS.items()}
        commodity_returns = commodity_returns.rename_axis("ticker")
        commodity_returns["name"] = commodity_returns.index.map(rev_cmd)
        commodity_returns = commodity_returns.reset_index()

    # --- Sector ETFs ---
    sector_returns = safe_returns(list(SECTOR_ETFS.values()), period="60d")
    if not sector_returns.empty:
        sector_returns["sector_name"] = list(SECTOR_ETFS.keys())
        sector_returns = sector_returns.set_index("sector_name")

    # --- Universe ---
    universe = load_universe(UNIVERSE_FILE)
    tickers = universe["ticker"].unique().tolist()

    uni_returns = safe_returns(tickers, period="60d")
    if not uni_returns.empty:
        uni_returns = uni_returns.reset_index()
        merged_uni = universe.merge(uni_returns, on="ticker", how="left")
    else:
        merged_uni = universe.copy()
        merged_uni["return_1d"] = None
        merged_uni["return_5d"] = None
        merged_uni["return_20d"] = None

    breadth = compute_breadth(uni_returns) if not uni_returns.empty else None

    # sector perf inside your universe (5D)
    if not merged_uni["return_5d"].isna().all():
        sector_perf_universe = (
            merged_uni.groupby("sector")["return_5d"]
            .mean()
            .sort_values(ascending=False)
        )
    else:
        sector_perf_universe = pd.Series(dtype=float)

    # --- Signals ---
    long_df, short_df = generate_watchlist_signals(merged_uni)

    # --- News ---
    news_lookup = {}
    for t in tickers:
        news_lookup[t] = get_news_for_ticker(t, from_date=yesterday, to_date=today, max_articles=2)

    # --- Build HTML ---
    html = []
    html.append(f"<h2>Daily Trading Dashboard — {today}</h2>")

    # Macro risk score
    risk_summary = compute_risk_score(macro_returns if not macro_returns.empty else pd.DataFrame())
    html.append(f"<p>{risk_summary}</p>")

    if breadth is not None:
        html.append(f"<p>Universe breadth (%% of tickers up over 20D): <b>{breadth*100:.1f}%</b></p>")

    # Macro table
    html.append("<h3>1. Macro Overview (Indices / Rates / FX)</h3>")
    if macro_returns.empty:
        html.append("<p><i>Macro data unavailable.</i></p>")
    else:
        html.append("<table border='1' cellpadding='4'><tr><th>Name</th><th>Ticker</th><th>1D</th><th>5D</th><th>20D</th></tr>")
        for _, row in macro_returns.iterrows():
            html.append(
                f"<tr><td>{row['name']}</td><td>{row['ticker']}</td>"
                f"<td>{format_pct(row['return_1d'])}</td>"
                f"<td>{format_pct(row['return_5d'])}</td>"
                f"<td>{format_pct(row['return_20d'])}</td></tr>"
            )
        html.append("</table>")

    # Commodities
    html.append("<h3>2. Commodities Overview</h3>")
    if commodity_returns.empty:
        html.append("<p><i>Commodity data unavailable.</i></p>")
    else:
        html.append("<table border='1' cellpadding='4'><tr><th>Name</th><th>Ticker</th><th>1D</th><th>5D</th><th>20D</th></tr>")
        for _, row in commodity_returns.iterrows():
            html.append(
                f"<tr><td>{row['name']}</td><td>{row['ticker']}</td>"
                f"<td>{format_pct(row['return_1d'])}</td>"
                f"<td>{format_pct(row['return_5d'])}</td>"
                f"<td>{format_pct(row['return_20d'])}</td></tr>"
            )
        html.append("</table>")

    # Sector ETFs + heat ranking
    html.append("<h3>3. Sector ETF Performance</h3>")
    if sector_returns.empty:
        html.append("<p><i>Sector ETF data unavailable.</i></p>")
    else:
        html.append("<table border='1' cellpadding='4'><tr><th>Sector</th><th>1D</th><th>5D</th><th>20D</th></tr>")
        for sector_name, row in sector_returns.iterrows():
            html.append(
                f"<tr><td>{sector_name}</td>"
                f"<td>{format_pct(row['return_1d'])}</td>"
                f"<td>{format_pct(row['return_5d'])}</td>"
                f"<td>{format_pct(row['return_20d'])}</td></tr>"
            )
        html.append("</table>")

        # sector ranking by 5D
        sector_rank = sector_returns["return_5d"].sort_values(ascending=False)
        top3 = sector_rank.head(3)
        bottom3 = sector_rank.tail(3)
        html.append("<p><b>Strongest sectors (5D):</b> " +
                    ", ".join(f"{s} ({format_pct(v)})" for s, v in top3.items()) + "</p>")
        html.append("<p><b>Weakest sectors (5D):</b> " +
                    ", ".join(f"{s} ({format_pct(v)})" for s, v in bottom3.items()) + "</p>")

    # Watchlist metrics
    html.append("<h3>4. Watchlist Metrics</h3>")
    html.append("<table border='1' cellpadding='4'><tr><th>Sector</th><th>Ticker</th><th>1D</th><th>5D</th><th>20D</th></tr>")
    for _, row in merged_uni.sort_values(["sector", "ticker"]).iterrows():
        html.append(
            f"<tr><td>{row['sector']}</td><td>{row['ticker']}</td>"
            f"<td>{format_pct(row['return_1d'])}</td>"
            f"<td>{format_pct(row['return_5d'])}</td>"
            f"<td>{format_pct(row['return_20d'])}</td></tr>"
        )
    html.append("</table>")

    # Signals
    html.append("<h3>5. Signals & Ideas</h3>")

    if long_df.empty:
        html.append("<h4>Long-side ideas</h4><p><i>No strong long signals.</i></p>")
    else:
        html.append("<h4>Long-side ideas (strong stocks in strong sectors)</h4><ul>")
        for _, row in long_df.head(15).iterrows():
            html.append(
                f"<li><b>{row['ticker']}</b> in <b>{row['sector']}</b> – "
                f"20D: {format_pct(row['return_20d'])}, "
                f"sector 20D: {format_pct(row['sector_20d'])}, "
                f"5D: {format_pct(row['return_5d'])}.</li>"
            )
        html.append("</ul>")

    if short_df.empty:
        html.append("<h4>Short-side ideas</h4><p><i>No strong short signals.</i></p>")
    else:
        html.append("<h4>Short-side ideas (weak stocks in weak sectors)</h4><ul>")
        for _, row in short_df.head(15).iterrows():
            html.append(
                f"<li><b>{row['ticker']}</b> in <b>{row['sector']}</b> – "
                f"20D: {format_pct(row['return_20d'])}, "
                f"sector 20D: {format_pct(row['sector_20d'])}, "
                f"5D: {format_pct(row['return_5d'])}.</li>"
            )
        html.append("</ul>")

    # News
    html.append("<h3>6. Ticker News (last 24h)</h3>")
    if not NEWS_API_KEY:
        html.append("<p><i>News disabled (no NEWS_API_KEY set).</i></p>")
    else:
        for t in tickers:
            html.append(f"<h4>{t}</h4>")
            headlines = news_lookup.get(t, [])
            if not headlines:
                html.append("<p><i>No major headlines in last 24h.</i></p>")
           else:
    html.append("<ul>")
    for h in headlines:
        title = h.get("title")
        source = h.get("source")
        url = h.get("url")

        if not title:
            continue

        label = title
        if source:
            label += f" <i>({source})</i>"

        if url:
            html.append(f"<li><a href='{url}'>{label}</a></li>")
        else:
            html.append(f"<li>{label}</li>")
    html.append("</ul>")


    return "\n".join(html)


# ============== EMAIL SENDER ======================================

def send_email(subject: str, html_body: str):
    if not EMAIL_USER or not EMAIL_PASS or not TO_EMAIL:
        print("Email credentials not configured. Skipping email send.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = TO_EMAIL

    part_html = MIMEText(html_body, "html")
    msg.attach(part_html)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        print(f"Email sent to {TO_EMAIL}")
    except smtplib.SMTPAuthenticationError as e:
        print(f"Email authentication failed: {e}. Check EMAIL_USER / EMAIL_PASS secrets.")
    except Exception as e:
        print(f"Error sending email: {e}")



# ============== MAIN ==============================================

def main():
    today = datetime.utcnow().date()
    html = build_dashboard_html()

    # Save a copy to disk (useful for debugging / archive)
    out_path = Path(f"daily_dashboard_{today}.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to {out_path.resolve()}")

    send_email(subject=f"Daily Trading Dashboard — {today}", html_body=html)


if __name__ == "__main__":
    main()
