# ==============================
# FinDocGPT — Financial Copilot
# One-Click Quick Demo populates all tabs (doc + Q&A + forecast + anomaly + export)
# ==============================

import os
import json
import yaml
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# ---- Local modules (your existing core files)
from core.loader import load_text_from_file
from core.qna import split_sentences, answer_question
from core.sentiment import doc_sentiment, rolling_sentiment
from core.forecast import fetch_prices, forecast_prices, growth_pct, clean_price_csv
from core.anomaly import anomaly_from_prices
from core.strategy import investment_strategy


# -------------------------
# NLTK bootstrap (lazy)
# -------------------------
def _ensure_nltk():
    import nltk
    local_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
    os.makedirs(local_dir, exist_ok=True)
    if local_dir not in nltk.data.path:
        nltk.data.path.append(local_dir)

    def need(path):
        try:
            nltk.data.find(path)
            return False
        except LookupError:
            return True

    try:
        if need("tokenizers/punkt"):
            nltk.download("punkt", download_dir=local_dir, quiet=True)
        if need("tokenizers/punkt_tab"):
            nltk.download("punkt_tab", download_dir=local_dir, quiet=True)
        if need("sentiment/vader_lexicon"):
            nltk.download("vader_lexicon", download_dir=local_dir, quiet=True)
    except Exception:
        # Non-blocking if download fails
        pass


# -------------------------
# Page config + minimal CSS
# -------------------------
st.set_page_config(page_title="FinDocGPT", page_icon="💹", layout="wide")

st.markdown("""
<style>
  .hero {
    background: linear-gradient(90deg,#7c3aed 0%,#06b6d4 100%);
    color:#fff; border-radius:16px; padding:16px 18px; margin-bottom:.75rem;
  }
  .brand { display:flex; align-items:center; gap:.75rem; font-weight:800; font-size:1.25rem; }
  .brand svg { width:26px; height:26px; }

  .pill-wrap { display:flex; gap:.5rem; margin:.25rem 0 .5rem 0; }
  .pill button { border-radius:999px !important; }
</style>
""", unsafe_allow_html=True)


# -------------------------
# Brand header
# -------------------------
st.markdown("""
<div class="hero">
  <div class="brand">
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#7c3aed"/>
          <stop offset="100%" stop-color="#06b6d4"/>
        </linearGradient>
      </defs>
      <circle cx="50" cy="50" r="46" fill="url(#g)"/>
      <path d="M25 60 L45 40 L58 53 L80 30" stroke="white" stroke-width="7" fill="none" stroke-linecap="round"/>
      <circle cx="25" cy="60" r="5" fill="white"/>
      <circle cx="45" cy="40" r="5" fill="white"/>
      <circle cx="58" cy="53" r="5" fill="white"/>
      <circle cx="80" cy="30" r="5" fill="white"/>
    </svg>
    <span>FinDocGPT — Financial Intelligence Copilot</span>
  </div>
  <div>Documents · Q&A · Sentiment · Forecast · Strategy · Export</div>
</div>
""", unsafe_allow_html=True)


# -------------------------
# Load config.yaml
# -------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


# -------------------------
# Session state helpers
# -------------------------
def set_state(**kvs):
    for k,v in kvs.items():
        st.session_state[k] = v

def get_state(k, d=None):
    return st.session_state.get(k, d)


# Defaults
st.session_state.setdefault("mode", "📄 Overview & Q&A")

st.session_state.setdefault("doc_text", "")
st.session_state.setdefault("doc_source", "")
st.session_state.setdefault("nltk_ready", False)

st.session_state.setdefault("latest_sentiment", None)
st.session_state.setdefault("latest_anomaly_label", "None")
st.session_state.setdefault("latest_proj_change", None)

# demo artifacts so Forecast & Q&A render instantly after demo
st.session_state.setdefault("demo_hist", None)      # DataFrame or None
st.session_state.setdefault("demo_fore", None)      # DataFrame or None
st.session_state.setdefault("demo_question", None)  # str or None
st.session_state.setdefault("demo_answers", None)   # list of (score, sent) or None
st.session_state.setdefault("latest_export", None)


# -------------------------
# One-Click Quick Demo (fills ALL tabs)
# -------------------------
def run_quick_demo():
    """Load sample doc + precompute Q&A + sample forecast + anomaly + export; fill all tabs."""
    # 1) Sample doc
    sample_doc = (
        "Q1 2025 Earnings Report – TechNova Inc.\n\n"
        "Revenue: TechNova reported total revenue of $4.8B, a 12% increase YoY.\n"
        "Net Profit: Net profit rose to $920M vs $850M last year.\n"
        "Expenses: Operating expenses were $3.1B (+8%), driven by R&D.\n"
        "Risks: Supply chain disruptions and semiconductor delays.\n"
        "Outlook: Q2 revenue expected at $4.9–5.0B; inflation pressure persists.\n"
        "Investor Sentiment: Analysts remain positive on the AI division.\n"
    )
    st.session_state.doc_text = sample_doc
    st.session_state.doc_source = "sample_document.txt"

    if not st.session_state.nltk_ready:
        _ensure_nltk()
        st.session_state.nltk_ready = True

    # 2) Sentiment (for KPIs, Sentiment tab)
    s = doc_sentiment(st.session_state.doc_text)
    st.session_state.latest_sentiment = float(s.get("compound", 0.0))

    # 3) Pre-fill Q&A (so Overview shows answers without extra clicks)
    try:
        default_q = (CFG["qna"]["quick_questions"][0]
                     if CFG.get("qna") and CFG["qna"].get("quick_questions")
                     else "What is the quarterly revenue?")
    except Exception:
        default_q = "What is the quarterly revenue?"
    sents = split_sentences(st.session_state.doc_text)
    top_k = int(CFG.get("qna", {}).get("top_k", 3))
    demo_ans = answer_question(default_q, sents, top_k)
    st.session_state.demo_question = default_q
    st.session_state.demo_answers = demo_ans

    # 4) Sample prices + forecast (for Forecast tab)
    start_date = datetime.utcnow().date() - timedelta(days=150)
    dates = pd.date_range(start_date, periods=150, freq="D")
    base = 70.0; growth = 0.001  # mild upward drift
    prices = [base]
    for i in range(1, len(dates)):
        prices.append(prices[-1] * (1.0 + growth))
    hist = pd.DataFrame({"Date": dates, "Close": prices})
    fore = forecast_prices(hist, horizon_days=90)

    st.session_state.demo_hist = hist
    st.session_state.demo_fore = fore
    proj_change = growth_pct(hist, fore)
    st.session_state.latest_proj_change = proj_change

    # 5) Anomaly + Strategy + Export payload
    label, _stats = anomaly_from_prices(hist)
    st.session_state.latest_anomaly_label = label

    decision, reason = investment_strategy(
        forecast_change=proj_change,
        sentiment_score=float(st.session_state.latest_sentiment or 0.0),
        anomaly_level=label,
    )

    set_state(latest_export={
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "source": "Quick Demo",
        "ticker": None,
        "period": None,
        "horizon_days": 90,
        "model": "Drift (avg return)",
        "last_close": float(hist["Close"].iloc[-1]),
        "forecast_last": float(fore["Predicted"].iloc[-1]),
        "projected_change_pct": proj_change,
        "anomaly": {"label": label},
        "sentiment_compound": float(st.session_state.latest_sentiment or 0.0),
        "decision": decision,
        "reason": reason,
    })


# -------------------------
# Top navigation
# -------------------------
modes = ["📄 Overview & Q&A", "📈 Sentiment", "📊 Forecast & Strategy", "📥 Export"]
st.markdown('<div class="pill-wrap">', unsafe_allow_html=True)
nav = st.columns([1,1,1,1,6])
for i, m in enumerate(modes):
    if nav[i].button(m, key=f"mode_btn_{i}", use_container_width=True):
        st.session_state.mode = m
st.markdown('</div>', unsafe_allow_html=True)


# -------------------------
# KPI row — native st.metric (always readable)
# -------------------------
avg_sent_view = get_state("latest_sentiment")
anomaly_val    = get_state("latest_anomaly_label","None")
proj_change    = get_state("latest_proj_change")

c1, c2, c3 = st.columns(3)
with c1:
    cval = f"{avg_sent_view:+.3f}" if avg_sent_view is not None else "—"
    st.metric("Sentiment (compound)", cval)
with c2:
    st.metric("Anomaly", anomaly_val)
with c3:
    cval = f"{proj_change:.2f}%" if proj_change is not None else "—"
    st.metric("Forecast Δ", cval)

st.write("")  # spacing


# =========================================================
# MODE: Overview & Q&A
# =========================================================
if st.session_state.mode.startswith("📄"):
    st.subheader("Overview & Q&A")

    colL, colR = st.columns([1.2, 1.8])

    # ---- Left: Document
    with colL:
        st.markdown("**Load a document**")
        uploaded = st.file_uploader("PDF / DOCX / TXT", type=["pdf","docx","txt"], key="doc_upload")

        # Single merged button: full app demo (doc + Q&A + forecast + anomaly + export)
        if st.button("⚡ Run Quick Demo (doc + Q&A + forecast)", type="primary", use_container_width=True):
            run_quick_demo()
            st.success("Quick demo loaded. All tabs are now populated.")

        if uploaded:
            tmp = f".tmp_{uploaded.name}"
            with open(tmp, "wb") as f:
                f.write(uploaded.getbuffer())
            try:
                text, source_name = load_text_from_file(tmp)
                st.session_state.doc_text = text
                st.session_state.doc_source = source_name

                # Clear demo Q&A if user provides own doc
                st.session_state.demo_question = None
                st.session_state.demo_answers = None

                # Update sentiment for uploaded doc
                if not st.session_state.nltk_ready:
                    _ensure_nltk(); st.session_state.nltk_ready = True
                s = doc_sentiment(st.session_state.doc_text)
                st.session_state.latest_sentiment = float(s.get("compound", 0.0))
            except Exception as e:
                st.error(f"Could not read file: {e}")

        if st.session_state.doc_text:
            st.markdown("**Source**")
            st.write(st.session_state.doc_source or "(uploaded)")
            st.text_area("Preview", st.session_state.doc_text[:5000], height=260)
        else:
            st.info("Upload a document or click ‘Run Quick Demo (doc + Q&A + forecast)’.")
            set_state(latest_sentiment=None)

    # ---- Right: Q&A
    with colR:
        st.markdown("**Ask questions about the document**")
        q = st.text_input("Your question", value=st.session_state.demo_question or "", placeholder="e.g., What is the quarterly revenue?")
        top_k = st.number_input("Top K", 1, 5, int(CFG.get("qna", {}).get("top_k", 3)), key="qa_topk")
        go = st.button("Get Answer", type="primary", key="qa_go")

        if st.session_state.doc_text:
            if not st.session_state.nltk_ready:
                _ensure_nltk(); st.session_state.nltk_ready = True

            sents = split_sentences(st.session_state.doc_text)

            # show demo answers if present (no extra click)
            if st.session_state.demo_answers and not go and not q:
                with st.expander(f"Demo Q&A — {st.session_state.demo_question}", expanded=True):
                    for score, sent in st.session_state.demo_answers:
                        st.write(f"**Score:** {score:.3f}")
                        st.write(f"• {sent}")
                        st.write("---")

            if go and (q or st.session_state.demo_question):
                query = q if q else st.session_state.demo_question
                answers = answer_question(query, sents, int(top_k))
                with st.expander(f"Answers — {query}", expanded=True):
                    for score, sent in answers:
                        st.write(f"**Score:** {score:.3f}")
                        st.write(f"• {sent}")
                        st.write("---")

            st.caption("Quick questions")
            quicks = CFG.get("qna", {}).get("quick_questions", [])
            if quicks:
                qq_cols = st.columns(len(quicks))
                for i, qq in enumerate(quicks):
                    if qq_cols[i].button(qq, key=f"qq_{i}"):
                        answers = answer_question(qq, sents, int(CFG.get("qna", {}).get("top_k", 3)))
                        with st.expander(f"Q: {qq}", expanded=True):
                            for score, sent in answers:
                                st.write(f"- ({score:.3f}) {sent}")
            else:
                st.caption("Configure quick_questions in config.yaml to show one-click prompts.")
        else:
            st.info("Load or run the demo to use Q&A.")


# =========================================================
# MODE: Sentiment
# =========================================================
elif st.session_state.mode.startswith("📈"):
    st.subheader("Sentiment Timeline")

    if st.session_state.doc_text:
        if not st.session_state.nltk_ready:
            _ensure_nltk(); st.session_state.nltk_ready = True

        win = int(CFG.get("sentiment", {}).get("window_sentences", 3))
        rows = rolling_sentiment(st.session_state.doc_text, window_sentences=win)
        if rows:
            df = pd.DataFrame(rows)
            st.line_chart(df.set_index("index")["compound"])
        else:
            st.info("Not enough sentences to compute rolling sentiment.")
    else:
        st.info("Load a document in Overview & Q&A or run Quick Demo first.")


# =========================================================
# MODE: Forecast & Strategy
# =========================================================
elif st.session_state.mode.startswith("📊"):
    st.subheader("Forecast & Strategy")

    # If the demo was run, render it first (no extra clicks)
    if isinstance(st.session_state.demo_hist, pd.DataFrame) and isinstance(st.session_state.demo_fore, pd.DataFrame):
        st.markdown("**Demo forecast** (from Quick Demo)")
        hist = st.session_state.demo_hist.copy()
        fore = st.session_state.demo_fore.copy()

        proj_change = growth_pct(hist, fore)
        set_state(latest_proj_change=proj_change)
        label, an_stats = anomaly_from_prices(hist)
        set_state(latest_anomaly_label=label)

        c1, c2, c3 = st.columns(3)
        last_actual = float(hist["Close"].iloc[-1])
        last_pred   = float(fore["Predicted"].iloc[-1])
        c1.metric("Projected change", f"{proj_change:.4f}%")
        c2.metric("Last actual close", f"{last_actual:,.4f}")
        c3.metric(f"Price in {len(fore)} days", f"{last_pred:,.4f}")
        st.caption(f"Anomaly: {label} (n={len(hist)})")

        hist_chart = hist.rename(columns={"Close":"Price"})[["Date","Price"]].copy(); hist_chart["Type"]="History"
        fore_chart = fore.rename(columns={"Predicted":"Price"})[["Date","Price"]].copy(); fore_chart["Type"]="Forecast"
        combo = pd.concat([hist_chart, fore_chart], ignore_index=True)
        combo = combo.pivot_table(index="Date", columns="Type", values="Price")
        st.line_chart(combo)

        sentiment_val = get_state("latest_sentiment") or 0.0
        decision, reason = investment_strategy(
            forecast_change=proj_change,
            sentiment_score=float(sentiment_val),
            anomaly_level=label,
        )
        st.markdown(f"### Recommendation: **{decision}**")
        st.write(f"**Why:** {reason}")
        st.caption(f"Inputs → Forecast Δ: {proj_change:.2f}%, Sentiment: {float(sentiment_val):.2f}, Anomaly: {label}")

        # Update export with demo decision
        latest = get_state("latest_export") or {}
        latest.update({"decision": decision, "reason": reason})
        set_state(latest_export=latest)

        st.divider()

    # Controls to override with real data
    st.markdown("#### Run your own forecast")
    c1, c2, c3 = st.columns([1.2,1.2,1.4])
    with c1:
        source = st.radio("Data source", ["Ticker (Yahoo)", "CSV file"], index=0, horizontal=True)
    with c2:
        model = st.radio("Model", ["Drift (avg return)", "Persistence (last)"], index=0, horizontal=True)
    with c3:
        horizon = st.selectbox("Horizon (days)", [7,14,30,60,90,120,150,180], index=4)

    hist = None

    if source.startswith("Ticker"):
        tcol1, tcol2 = st.columns([1,1])
        with tcol1:
            ticker = st.text_input("Ticker", value="AAPL")
        with tcol2:
            period = st.selectbox("History period", ["3mo","6mo","1y","2y","5y","10y","max"], index=2)
        if st.button("Run Forecast (Ticker)"):
            with st.spinner("Fetching prices…"):
                hist = fetch_prices(ticker, period=period)

    else:  # CSV
        csv_file = st.file_uploader("Upload prices CSV (Date + Close/Adj Close/Price)", type=["csv"], key="csv_upload_main")
        if st.button("Run Forecast (CSV)"):
            if csv_file:
                with st.spinner("Cleaning CSV…"):
                    hist = clean_price_csv(csv_file)
            else:
                st.warning("Upload a CSV first.")

    if isinstance(hist, pd.DataFrame) and not hist.empty:
        try:
            if model.startswith("Drift"):
                fore = forecast_prices(hist, horizon_days=int(horizon))
            else:
                last_actual_val = float(hist["Close"].astype(float).iloc[-1])
                future_dates = pd.date_range(pd.to_datetime(hist["Date"].iloc[-1]) + pd.Timedelta(days=1),
                                             periods=int(horizon), freq="D")
                fore = pd.DataFrame({"Date": future_dates, "Predicted": [last_actual_val]*int(horizon)})

            proj_change = growth_pct(hist, fore)
            set_state(latest_proj_change=proj_change)

            anomaly_label, an_stats = anomaly_from_prices(hist)
            set_state(latest_anomaly_label=anomaly_label)

            c1, c2, c3 = st.columns(3)
            last_actual = float(hist["Close"].iloc[-1])
            last_pred   = float(fore["Predicted"].iloc[-1])
            c1.metric("Projected change", f"{proj_change:.4f}%")
            c2.metric("Last actual close", f"{last_actual:,.4f}")
            c3.metric(f"Price in {int(horizon)} days", f"{last_pred:,.4f}")
            st.caption(f"Anomaly: {anomaly_label} (n={len(hist)})")

            hist_chart = hist.rename(columns={"Close":"Price"})[["Date","Price"]]; hist_chart["Type"]="History"
            fore_chart = fore.rename(columns={"Predicted":"Price"})[["Date","Price"]]; fore_chart["Type"]="Forecast"
            combo = pd.concat([hist_chart, fore_chart], ignore_index=True)
            combo = combo.pivot_table(index="Date", columns="Type", values="Price")
            st.line_chart(combo)

            sentiment_val = get_state("latest_sentiment") or 0.0
            decision, reason = investment_strategy(proj_change, float(sentiment_val), anomaly_label)
            st.markdown(f"### Recommendation: **{decision}**")
            st.write(f"**Why:** {reason}")

            set_state(latest_export={
                "timestamp": datetime.utcnow().isoformat()+"Z",
                "source": source,
                "ticker": locals().get("ticker"),
                "period": locals().get("period"),
                "horizon_days": int(horizon),
                "model": model,
                "last_close": last_actual,
                "forecast_last": last_pred,
                "projected_change_pct": proj_change,
                "anomaly": {"label": anomaly_label, **an_stats},
                "sentiment_compound": float(sentiment_val),
                "decision": decision,
                "reason": reason,
            })

        except Exception as e:
            st.error(f"Forecast failed: {e}")
            try:
                st.write("hist head:", hist.head()); st.write("hist dtypes:", hist.dtypes)
            except Exception:
                pass


# =========================================================
# MODE: Export
# =========================================================
elif st.session_state.mode.startswith("📥"):
    st.subheader("Export Results")
    latest = get_state("latest_export")
    if latest:
        st.json(latest)
        st.download_button(
            "Download JSON",
            data=json.dumps(latest, indent=2),
            file_name=f"findocgpt_export_{latest['timestamp'].replace(':','-')}.json",
            mime="application/json",
            use_container_width=True,
        )
        st.success("Attach this JSON to your submission.")
    else:
        st.info("Run Quick Demo or your own forecast to enable export.")


# =========================================================
# About / Help / Checklist
# =========================================================
with st.expander("ℹ️ About • Help • Submission checklist", expanded=False):
    st.markdown("""
### FinDocGPT — Financial Intelligence Copilot *(Solo)*
**Included**
- **📄 Q&A:** Ask questions over uploaded PDFs/DOCX/TXT.
- **📈 Sentiment:** VADER compound + rolling timeline.
- **📊 Forecast & Strategy:** Drift/Persistence baselines, anomaly flags, plain-English **BUY/HOLD/SELL**.
- **📥 Export:** JSON artifact for submission.

**One-click demo**
- Use **“Run Quick Demo (doc + Q&A + forecast)”** in **Overview** to auto-populate the entire app, including demo Q&A answers and a demo forecast chart.

**Judging alignment**
- *Technical depth:* modular core, robust CSV cleaning, local NLTK bootstrap.
- *Communication:* KPIs, demo answers, charts, JSON export.
- *Innovation & creativity:* transparent baseline + anomaly + sentiment fusion for a decision.

**Stack**
- Streamlit · NLTK (VADER) · yfinance/CSV · Plotly · pandas/scikit-learn

**Submission checklist**
- [x] Stage 1: Q&A + Sentiment  
- [x] Stage 2: Forecast + Strategy  
- [x] Export JSON attached  
""")
