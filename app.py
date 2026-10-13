#!/usr/bin/env python3
"""
Market Wizard Engine
--------------------
Local, self-contained macro-force aggregator that:

  1. Pulls liquidity/rates (FRED 10Y yield, DXY)
  2. Pulls macro health (FRED unemployment)
  3. Scores live RSS headlines with local FinBERT
  4. Measures micro/corporate health (SPY 30-day return)
  5. Measures market structure / vol (VIX)

Then computes a unified Bleed Severity Score (0–100) and maps it onto
the technical trading-system execution directives.
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# API / model configuration
# ---------------------------------------------------------------------------
FRED_API_KEY = "502c9b36b39d4ba8285a442df6231eab"

FRED_SERIES_DGS10 = "DGS10"
FRED_SERIES_UNRATE = "UNRATE"
DXY_TICKER = "DX-Y.NYB"
SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"

RSS_FEEDS = (
    "https://finance.yahoo.com/news/rssindex",
    "https://www.federalreserve.gov/feeds/press_all.xml",
)
FINBERT_MODEL = "ProsusAI/finbert"
SYSTEMIC_KEYWORDS = (
    "rate",
    "cpi",
    "inflation",
    "fed",
    "war",
    "tariff",
    "emergency",
    "crisis",
    "liquidity",
)
HEADLINE_LIMIT_PER_FEED = 15

# Score weights (sum of max contributions = 100)
YIELD_SPIKE_POINTS = 25
YIELD_SPIKE_THRESHOLD_PCT = 2.0
SYSTEMIC_NEGATIVE_POINTS = 25
SYSTEMIC_NEGATIVE_MIN_COUNT = 2
SPY_DRAWDOWN_POINTS = 20
SPY_DRAWDOWN_THRESHOLD_PCT = -2.0
VIX_SEVERE_POINTS = 30
VIX_ELEVATED_POINTS = 15
VIX_SEVERE_LEVEL = 25.0
VIX_ELEVATED_LEVEL = 20.0

# Safe defaults when a live fetch fails so scoring still runs
DEFAULT_YIELD_5D_PCT = 0.0
DEFAULT_DXY_LAST = None
DEFAULT_UNRATE = None
DEFAULT_SPY_30D_PCT = 0.0
DEFAULT_VIX = 18.0  # below elevated threshold; adds 0 points

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("market_wizard")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class HeadlineAnalysis:
    title: str
    source: str
    label: str  # positive | negative | neutral
    score: float
    has_systemic_keyword: bool
    matched_keywords: list[str] = field(default_factory=list)

    @property
    def is_systemic_negative(self) -> bool:
        return self.label == "negative" and self.has_systemic_keyword


@dataclass
class ForceSnapshot:
    yield_last: Optional[float] = None
    yield_5d_pct: Optional[float] = DEFAULT_YIELD_5D_PCT
    dxy_last: Optional[float] = DEFAULT_DXY_LAST
    dxy_change_pct: Optional[float] = None
    unrate: Optional[float] = DEFAULT_UNRATE
    spy_last: Optional[float] = None
    spy_30d_pct: Optional[float] = DEFAULT_SPY_30D_PCT
    vix_last: Optional[float] = DEFAULT_VIX
    headlines: list[HeadlineAnalysis] = field(default_factory=list)
    fetch_notes: list[str] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    yield_points: int = 0
    news_points: int = 0
    spy_points: int = 0
    vix_points: int = 0
    systemic_negative_count: int = 0

    @property
    def total(self) -> int:
        return min(100, self.yield_points + self.news_points + self.spy_points + self.vix_points)


@dataclass
class Directive:
    regime: str
    emoji: str
    action: str
    rules: str
    color_name: str


# ---------------------------------------------------------------------------
# Fetch helpers (each isolated so one failure never kills the pipeline)
# ---------------------------------------------------------------------------
def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_change(new: float, old: float) -> Optional[float]:
    if old == 0:
        return None
    return ((new - old) / old) * 100.0


def fetch_dgs10_five_day_change() -> tuple[Optional[float], Optional[float], str]:
    """Force 1a: 10Y Treasury yield and 5-session percent change via FRED."""
    try:
        from fredapi import Fred

        fred = Fred(api_key=FRED_API_KEY)
        series = fred.get_series(FRED_SERIES_DGS10)
        if series is None or series.empty:
            raise ValueError("Empty DGS10 series")
        clean = series.dropna()
        if len(clean) < 6:
            raise ValueError(f"Insufficient DGS10 observations ({len(clean)})")
        latest = float(clean.iloc[-1])
        prior = float(clean.iloc[-6])  # 5 trading sessions back
        change = _pct_change(latest, prior)
        note = (
            f"DGS10 last={latest:.3f}% | 5-session change={change:+.3f}% "
            f"(prior={prior:.3f}%)"
        )
        logger.info(note)
        return latest, change, note
    except Exception as exc:
        note = f"FRED DGS10 fallback ({DEFAULT_YIELD_5D_PCT:+.2f}% 5d): {exc}"
        logger.warning(note)
        return None, DEFAULT_YIELD_5D_PCT, note


def fetch_dxy() -> tuple[Optional[float], Optional[float], str]:
    """Force 1b: US Dollar Index last close and 5-session change via yfinance."""
    try:
        import yfinance as yf

        hist = yf.Ticker(DXY_TICKER).history(period="1mo", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            raise ValueError("Empty DXY history")
        close = hist["Close"].dropna()
        if close.empty:
            raise ValueError("No DXY closes")
        last = float(close.iloc[-1])
        change = None
        if len(close) >= 6:
            change = _pct_change(last, float(close.iloc[-6]))
        note = (
            f"DXY ({DXY_TICKER}) last={last:.3f}"
            + (f" | 5-session change={change:+.3f}%" if change is not None else "")
        )
        logger.info(note)
        return last, change, note
    except Exception as exc:
        note = f"yfinance DXY fallback: {exc}"
        logger.warning(note)
        return DEFAULT_DXY_LAST, None, note


def fetch_unemployment() -> tuple[Optional[float], str]:
    """Force 2: US unemployment rate via FRED UNRATE."""
    try:
        from fredapi import Fred

        fred = Fred(api_key=FRED_API_KEY)
        series = fred.get_series(FRED_SERIES_UNRATE)
        if series is None or series.empty:
            raise ValueError("Empty UNRATE series")
        last = float(series.dropna().iloc[-1])
        note = f"UNRATE last={last:.2f}%"
        logger.info(note)
        return last, note
    except Exception as exc:
        note = f"FRED UNRATE fallback: {exc}"
        logger.warning(note)
        return DEFAULT_UNRATE, note


def _parse_feed(url: str, limit: int) -> list[tuple[str, str]]:
    import feedparser

    parsed = feedparser.parse(url)
    if getattr(parsed, "bozo", False) and not parsed.entries:
        raise ValueError(getattr(parsed, "bozo_exception", "bozo feed with no entries"))
    rows: list[tuple[str, str]] = []
    for entry in parsed.entries[:limit]:
        title = (entry.get("title") or "").strip()
        if title:
            rows.append((title, url))
    if not rows:
        raise ValueError("No titled entries in feed")
    return rows


def fetch_headlines() -> tuple[list[tuple[str, str]], list[str]]:
    """Force 3: live RSS titles. Continues if one feed dies."""
    collected: list[tuple[str, str]] = []
    notes: list[str] = []
    for url in RSS_FEEDS:
        try:
            rows = _parse_feed(url, HEADLINE_LIMIT_PER_FEED)
            collected.extend(rows)
            notes.append(f"RSS ok ({len(rows)} titles): {url}")
            logger.info(notes[-1])
        except Exception as exc:
            notes.append(f"RSS fallback ({url}): {exc}")
            logger.warning(notes[-1])
    if not collected:
        notes.append("No headlines retrieved; FinBERT will score an empty set.")
    return collected, notes


def _contains_systemic_keywords(text: str) -> tuple[bool, list[str]]:
    lowered = text.lower()
    matched = [kw for kw in SYSTEMIC_KEYWORDS if kw in lowered]
    return bool(matched), matched


def load_finbert_pipeline():
    """Load local FinBERT. Returns None on failure so callers can fall back."""
    try:
        from transformers import pipeline

        logger.info("Loading local FinBERT model (%s)…", FINBERT_MODEL)
        nlp = pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT pipeline ready.")
        return nlp
    except Exception as extra:
        logger.warning("FinBERT unavailable (%s). Headlines will be treated as NEUTRAL.", extra)
        return None


def _normalize_finbert_label(raw: str) -> str:
    label = (raw or "").strip().lower()
    if label in {"positive", "pos", "label_2"}:
        return "positive"
    if label in {"negative", "neg", "label_0"}:
        return "negative"
    return "neutral"


def analyze_headlines(
    raw_headlines: list[tuple[str, str]],
    nlp,
) -> list[HeadlineAnalysis]:
    results: list[HeadlineAnalysis] = []
    if not raw_headlines:
        return results

    titles = [t for t, _ in raw_headlines]
    inferences: list[dict[str, Any]] = []
    if nlp is not None:
        try:
            inferences = nlp(titles)
        except Exception as extra:
            logger.warning("FinBERT batch inference failed (%s). Using NEUTRAL.", extra)
            inferences = []

    for idx, (title, source) in enumerate(raw_headlines):
        has_kw, matched = _contains_systemic_keywords(title)
        label = "neutral"
        conf = 0.0
        if idx < len(inferences) and isinstance(inferences[idx], dict):
            label = _normalize_finbert_label(str(inferences[idx].get("label", "neutral")))
            conf = float(inferences[idx].get("score") or 0.0)
        results.append(
            HeadlineAnalysis(
                title=title,
                source=source,
                label=label,
                score=conf,
                has_systemic_keyword=has_kw,
                matched_keywords=matched,
            )
        )
    return results


def fetch_spy_30d() -> tuple[Optional[float], Optional[float], str]:
    """Force 4: SPY last close and ~30-calendar-day percent return."""
    try:
        import yfinance as yf

        hist = yf.Ticker(SPY_TICKER).history(period="3mo", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            raise ValueError("Empty SPY history")
        close = hist["Close"].dropna()
        if close.empty:
            raise ValueError("No SPY closes")
        last = float(close.iloc[-1])
        idx = close.index
        # Prefer a true 30-calendar-day lookback; fall back to 21 trading sessions.
        target = idx[-1] - pd.Timedelta(days=30)
        prior_idx = idx[idx <= target]
        if len(prior_idx) > 0:
            prior = float(close.loc[prior_idx[-1]])
        elif len(close) > 21:
            prior = float(close.iloc[-22])
        else:
            raise ValueError("Not enough SPY history for 30-day return")
        change = _pct_change(last, prior)
        note = f"SPY last={last:.2f} | 30-day return={change:+.3f}%"
        logger.info(note)
        return last, change, note
    except Exception as extra:
        note = f"yfinance SPY fallback ({DEFAULT_SPY_30D_PCT:+.2f}% 30d): {extra}"
        logger.warning(note)
        return None, DEFAULT_SPY_30D_PCT, note


def fetch_vix() -> tuple[Optional[float], str]:
    """Force 5: CBOE Volatility Index last close."""
    try:
        import yfinance as yf

        hist = yf.Ticker(VIX_TICKER).history(period="5d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            raise ValueError("Empty VIX history")
        last = float(hist["Close"].dropna().iloc[-1])
        note = f"VIX last={last:.2f}"
        logger.info(note)
        return last, note
    except Exception as extra:
        note = f"yfinance VIX fallback ({DEFAULT_VIX:.1f}): {extra}"
        logger.warning(note)
        return DEFAULT_VIX, note


# ---------------------------------------------------------------------------
# Scoring + directives
# ---------------------------------------------------------------------------
def compute_bleed_score(snap: ForceSnapshot) -> ScoreBreakdown:
    br = ScoreBreakdown()

    y5 = snap.yield_5d_pct if snap.yield_5d_pct is not None else DEFAULT_YIELD_5D_PCT
    if y5 > YIELD_SPIKE_THRESHOLD_PCT:
        br.yield_points = YIELD_SPIKE_POINTS

    br.systemic_negative_count = sum(1 for h in snap.headlines if h.is_systemic_negative)
    if br.systemic_negative_count >= SYSTEMIC_NEGATIVE_MIN_COUNT:
        br.news_points = SYSTEMIC_NEGATIVE_POINTS

    spy = snap.spy_30d_pct if snap.spy_30d_pct is not None else DEFAULT_SPY_30D_PCT
    if spy < SPY_DRAWDOWN_THRESHOLD_PCT:
        br.spy_points = SPY_DRAWDOWN_POINTS

    vix = snap.vix_last if snap.vix_last is not None else DEFAULT_VIX
    if vix >= VIX_SEVERE_LEVEL:
        br.vix_points = VIX_SEVERE_POINTS
    elif vix >= VIX_ELEVATED_LEVEL:
        br.vix_points = VIX_ELEVATED_POINTS

    return br


def map_directive(bleed_score: int) -> Directive:
    if bleed_score < 35:
        return Directive(
            regime="BULL REGIME",
            emoji="🟢",
            color_name="GREEN",
            action="LOOK FOR 20D EMA ENTRIES & DCA",
            rules=(
                "Take setups above 200D/50D MA stack with RSI < 70. "
                "Confirm monthly BX trend. Safe to deploy cash buffer."
            ),
        )
    if bleed_score <= 65:
        return Directive(
            regime="CHOPPY REGIME",
            emoji="🟡",
            color_name="YELLOW",
            action="HOLD & LOCK BREAKEVEN STOPS",
            rules=(
                "Move stops to breakeven on ATH positions. "
                "NO new 3X ETF leverage. Wait out volatility."
            ),
        )
    return Directive(
        regime="SEVERE BLEED REGIME",
        emoji="🔴",
        color_name="RED",
        action="TRIM / SELL IMMEDIATELY & PROTECT CAPITAL",
        rules=(
            "Enforce strict 3.5%–5% max loss stop-loss on 3X ETFs. "
            "Trim PT1/PT2 profits on winning trades. DO NOT deploy cash buffer."
        ),
    )


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def _fmt(value: Optional[float], digits: int = 2, suffix: str = "", signed: bool = False) -> str:
    if value is None:
        return "N/A"
    spec = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return spec.format(value) + suffix


def _bar(score: int, width: int = 40) -> str:
    filled = int(round((max(0, min(100, score)) / 100.0) * width))
    return "█" * filled + "░" * (width - filled)


def _sentiment_tally(headlines: Iterable[HeadlineAnalysis]) -> dict[str, int]:
    tally = {"positive": 0, "negative": 0, "neutral": 0}
    for h in headlines:
        tally[h.label] = tally.get(h.label, 0) + 1
    return tally


def render_report(snap: ForceSnapshot, breakdown: ScoreBreakdown) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    score = breakdown.total
    directive = map_directive(score)
    tally = _sentiment_tally(snap.headlines)

    lines: list[str] = []
    sep = "=" * 78
    thin = "-" * 78

    lines += [
        sep,
        "  MARKET WIZARD ENGINE  |  Local Macro-Force Aggregator",
        f"  Generated: {now}",
        sep,
        "",
        "  FORCE 1  ·  LIQUIDITY & RATES",
        thin,
        f"  10Y Treasury (DGS10) last ........ {_fmt(snap.yield_last, 3, '%')}",
        f"  10Y 5-session % change ........... {_fmt(snap.yield_5d_pct, 3, '%', signed=True)}"
        f"   {'⚠ SPIKE' if breakdown.yield_points else '· stable'}",
        f"  US Dollar Index (DX-Y.NYB) ...... {_fmt(snap.dxy_last, 3)}",
        f"  DXY 5-session % change ........... {_fmt(snap.dxy_change_pct, 3, '%', signed=True)}",
        "",
        "  FORCE 2  ·  MACRO ECONOMIC HEALTH",
        thin,
        f"  US Unemployment Rate (UNRATE) ... {_fmt(snap.unrate, 2, '%')}",
        "",
        "  FORCE 3  ·  GEOPOLITICS & BREAKING HEADLINES",
        thin,
        f"  Headlines scored (FinBERT) ...... {len(snap.headlines)}",
        f"  Sentiment mix ................... "
        f"+{tally['positive']}  −{tally['negative']}  ○{tally['neutral']}",
        f"  Systemic-negative hits .......... {breakdown.systemic_negative_count}  "
        f"(threshold ≥ {SYSTEMIC_NEGATIVE_MIN_COUNT})",
        f"  Keywords scanned ................ {', '.join(SYSTEMIC_KEYWORDS)}",
    ]

    flagged = [h for h in snap.headlines if h.is_systemic_negative]
    if flagged:
        lines.append("  Flagged systemic negatives:")
        for h in flagged[:8]:
            kw = ",".join(h.matched_keywords)
            lines.append(f"    • [{h.label.upper()} {h.score:.2f} | {kw}] {h.title[:88]}")
    elif snap.headlines:
        lines.append("  No systemic-negative headline cluster detected.")
    else:
        lines.append("  No headlines available (feeds unavailable).")

    # compact sample of remaining headlines
    sample = snap.headlines[:6]
    if sample:
        lines.append("  Sample scored headlines:")
        for h in sample:
            marker = {"positive": "↑", "negative": "↓", "neutral": "·"}[h.label]
            lines.append(f"    {marker} [{h.label:8s} {h.score:.2f}] {h.title[:72]}")

    lines += [
        "",
        "  FORCE 4  ·  MICRO & CORPORATE HEALTH",
        thin,
        f"  SPY last ........................ {_fmt(snap.spy_last, 2)}",
        f"  SPY 30-day return ............... {_fmt(snap.spy_30d_pct, 3, '%', signed=True)}"
        f"   {'⚠ DRAWDOWN' if breakdown.spy_points else '· constructive'}",
        "",
        "  FORCE 5  ·  MARKET STRUCTURE & VOLATILITY",
        thin,
        f"  CBOE VIX (^VIX) ................. {_fmt(snap.vix_last, 2)}"
        f"   {'⚠ SEVERE' if breakdown.vix_points == VIX_SEVERE_POINTS else ('⚠ ELEVATED' if breakdown.vix_points else '· contained')}",
        "",
        "  BLEED SEVERITY SCORE",
        thin,
        f"  10Y yield spike (> {YIELD_SPIKE_THRESHOLD_PCT:.1f}%) ........ {breakdown.yield_points:+3d}  / {YIELD_SPIKE_POINTS}",
        f"  Systemic-negative news (≥{SYSTEMIC_NEGATIVE_MIN_COUNT}) ...... {breakdown.news_points:+3d}  / {SYSTEMIC_NEGATIVE_POINTS}",
        f"  SPY 30d < {SPY_DRAWDOWN_THRESHOLD_PCT:.1f}% ................ {breakdown.spy_points:+3d}  / {SPY_DRAWDOWN_POINTS}",
        f"  VIX (≥{VIX_ELEVATED_LEVEL:.0f} / ≥{VIX_SEVERE_LEVEL:.0f}) ................. {breakdown.vix_points:+3d}  / {VIX_SEVERE_POINTS}",
        thin,
        f"  TOTAL  {score:3d} / 100    [{_bar(score)}]",
        "",
        "  SYSTEM EXECUTION DIRECTIVE",
        sep,
        f"  {directive.emoji}  {directive.color_name} — {directive.regime}  (score {score})",
        f"  ACTION : {directive.action}",
        f"  RULES  : {directive.rules}",
        sep,
    ]

    if snap.fetch_notes:
        lines.append("  FETCH NOTES")
        lines.append(thin)
        for note in snap.fetch_notes:
            lines.append(f"  · {note}")
        lines.append(sep)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def collect_snapshot() -> ForceSnapshot:
    snap = ForceSnapshot()

    y_last, y_chg, y_note = fetch_dgs10_five_day_change()
    snap.yield_last = y_last
    snap.yield_5d_pct = y_chg
    snap.fetch_notes.append(y_note)

    dxy_last, dxy_chg, dxy_note = fetch_dxy()
    snap.dxy_last = dxy_last
    snap.dxy_change_pct = dxy_chg
    snap.fetch_notes.append(dxy_note)

    unrate, un_note = fetch_unemployment()
    snap.unrate = unrate
    snap.fetch_notes.append(un_note)

    raw_headlines, rss_notes = fetch_headlines()
    snap.fetch_notes.extend(rss_notes)
    nlp = load_finbert_pipeline() if raw_headlines else None
    if raw_headlines and nlp is None:
        snap.fetch_notes.append("FinBERT fallback: all headlines scored NEUTRAL.")
    snap.headlines = analyze_headlines(raw_headlines, nlp)

    spy_last, spy_chg, spy_note = fetch_spy_30d()
    snap.spy_last = spy_last
    snap.spy_30d_pct = spy_chg
    snap.fetch_notes.append(spy_note)

    vix, vix_note = fetch_vix()
    snap.vix_last = vix
    snap.fetch_notes.append(vix_note)

    return snap


def run() -> int:
    logger.info("Market Wizard Engine starting.")
    snap = collect_snapshot()
    breakdown = compute_bleed_score(snap)
    report = render_report(snap, breakdown)
    sys.stdout.write(report)
    sys.stdout.flush()
    logger.info("Bleed score=%d → %s", breakdown.total, map_directive(breakdown.total).regime)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
