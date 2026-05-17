import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from model import detect_emotion
from insights import generate_business_insights

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentioAI – Emotion Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}
.main { background-color: #060b18; }
.block-container { padding: 2rem 2.5rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1b2a 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0f2240 0%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(56,189,248,0.15);
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.82rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-left: 3px solid #38bdf8;
    padding-left: 0.75rem;
    margin: 2rem 0 1rem 0;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #0c1f3d 0%, #111827 100%);
    border: 1px solid #818cf8;
    border-left: 4px solid #818cf8;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    font-size: 0.97rem;
    line-height: 1.75;
    color: #cbd5e1;
}

/* Fake review badge */
.fake-badge {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid #ef4444;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.clean-badge {
    display: inline-block;
    background: rgba(34,197,94,0.12);
    color: #4ade80;
    border: 1px solid #22c55e;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* Result row in table */
.review-row {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
}

/* Single analyze result */
.result-emotion {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.6rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Plotly chart backgrounds */
.js-plotly-plot .plotly { border-radius: 12px; }

/* Tab styling */
[data-baseweb="tab-list"] { gap: 6px; }
[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 6px 18px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ───────────────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","is","it","in","on","at","to","for","of","and","or","but",
    "this","that","was","with","my","i","me","we","they","he","she","you","are",
    "be","been","have","has","had","do","did","not","so","as","up","by","from",
    "its","our","your","their","all","no","just","if","would","could","should",
    "very","more","than","out","also","what","there","which","when","how","get",
    "got","will","can","about","after","before","been","some","any","each","over",
}

EMOJI_MAP = {
    # original 6
    "joy":          "😊", "sadness":      "😢", "anger":        "😠",
    "fear":         "😨", "love":         "❤️", "surprise":     "😲",
    "disgust":      "🤢", "neutral":      "😐",
    # GoEmotions 28
    "admiration":   "🤩", "amusement":    "😄", "annoyance":    "😤",
    "approval":     "👍", "caring":       "🤗", "confusion":    "😕",
    "curiosity":    "🧐", "desire":       "😍", "disappointment":"😞",
    "disapproval":  "👎", "embarrassment":"😳", "excitement":   "🎉",
    "gratitude":    "🙏", "grief":        "😭",
    "nervousness":  "😬", "optimism":     "🌟", "pride":        "😎",
    "realization":  "💡", "relief":       "😌", "remorse":      "😔",
    "repulsion":    "🤮",
}

EMOTION_COLORS = {
    # original 6
    "joy":           "#facc15", "sadness":       "#60a5fa", "anger":         "#f87171",
    "fear":          "#c084fc", "love":          "#f472b6", "surprise":      "#34d399",
    "disgust":       "#a3e635", "neutral":       "#94a3b8",
    # GoEmotions 28
    "admiration":    "#f59e0b", "amusement":     "#fbbf24", "annoyance":     "#fb923c",
    "approval":      "#4ade80", "caring":        "#f472b6", "confusion":     "#a78bfa",
    "curiosity":     "#38bdf8", "desire":        "#e879f9", "disappointment":"#64748b",
    "disapproval":   "#ef4444", "embarrassment": "#f97316", "excitement":    "#facc15",
    "gratitude":     "#86efac", "grief":         "#475569", "nervousness":   "#c084fc",
    "optimism":      "#fde68a", "pride":         "#818cf8", "realization":   "#67e8f9",
    "relief":        "#6ee7b7", "remorse":       "#94a3b8", "repulsion":     "#a3e635",
}

def extract_keywords(texts: list[str], top_n: int = 15) -> dict:
    """Extract top non-stopword keywords from a list of texts."""
    words = []
    for t in texts:
        for w in re.findall(r'\b[a-zA-Z]{3,}\b', t.lower()):
            if w not in STOPWORDS:
                words.append(w)
    return dict(Counter(words).most_common(top_n))

def is_fake_review(text: str, all_texts: list[str]) -> tuple[bool, str]:
    """
    Simple heuristic fake / spam detection.
    Returns (is_fake: bool, reason: str)
    """
    text_lower = text.lower().strip()
    reasons = []

    # 1. Too short (< 4 words)
    if len(text.split()) < 4:
        reasons.append("too short")

    # 2. ALL CAPS
    alpha = [c for c in text if c.isalpha()]
    if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.75:
        reasons.append("excessive caps")

    # 3. Excessive punctuation / exclamation
    if text.count('!') >= 3:
        reasons.append("excessive exclamation marks")

    # 4. Repetitive words (same word > 30% of words)
    word_list = re.findall(r'\b\w+\b', text_lower)
    if word_list:
        freq = Counter(word_list)
        most_common_ratio = freq.most_common(1)[0][1] / len(word_list)
        if most_common_ratio > 0.3 and len(word_list) > 3:
            reasons.append("repetitive words")

    # 5. Exact duplicate of another review
    duplicates = [t for t in all_texts if t.strip().lower() == text_lower and all_texts.count(t) > 1]
    if duplicates:
        reasons.append("duplicate review")

    # 6. Generic spam phrases
    spam_phrases = ["click here", "buy now", "free offer", "100% guaranteed",
                    "visit our", "check out our", "limited time", "discount code"]
    for phrase in spam_phrases:
        if phrase in text_lower:
            reasons.append(f"spam phrase: '{phrase}'")
            break

    is_fake = len(reasons) > 0
    return is_fake, ", ".join(reasons) if reasons else "clean"

def plot_emotion_donut(emotion_counts: dict):
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    colors = [EMOTION_COLORS.get(e, "#94a3b8") for e in labels]

    fig = go.Figure(go.Pie(
        labels=[f"{EMOJI_MAP.get(l,'🙂')} {l.capitalize()}" for l in labels],
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#060b18", width=2)),
        textfont=dict(family="DM Sans", color="#e2e8f0"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8", family="DM Sans"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
    )
    return fig

def plot_bar_emotions(emotion_counts: dict):
    labels = [f"{EMOJI_MAP.get(e,'🙂')} {e.capitalize()}" for e in emotion_counts]
    values = list(emotion_counts.values())
    colors = [EMOTION_COLORS.get(e, "#94a3b8") for e in emotion_counts]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=values, textposition="outside",
        textfont=dict(color="#e2e8f0", family="DM Sans"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#64748b", tickfont=dict(family="DM Sans")),
        yaxis=dict(showgrid=True, gridcolor="#1e3a5f", color="#64748b"),
        margin=dict(t=20, b=20, l=10, r=10),
        height=300,
    )
    return fig

def plot_keyword_bar(keyword_freq: dict):
    if not keyword_freq:
        return None
    words = list(keyword_freq.keys())
    counts = list(keyword_freq.values())

    fig = go.Figure(go.Bar(
        y=words[::-1], x=counts[::-1],
        orientation='h',
        marker=dict(
            color=counts[::-1],
            colorscale=[[0, "#1e3a5f"], [1, "#38bdf8"]],
            showscale=False,
        ),
        text=counts[::-1], textposition="outside",
        textfont=dict(color="#e2e8f0", family="DM Sans"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#1e3a5f", color="#64748b"),
        yaxis=dict(showgrid=False, color="#94a3b8", tickfont=dict(family="DM Sans", size=12)),
        margin=dict(t=10, b=10, l=10, r=30),
        height=max(250, len(words) * 32),
    )
    return fig

def plot_sentiment_trend(sentiments: list[str]):
    """Bar chart of positive/negative/neutral trend by review index."""
    mapping = {"joy": "Positive", "love": "Positive", "surprise": "Positive",
               "anger": "Negative", "sadness": "Negative", "fear": "Negative", "disgust": "Negative",
               "neutral": "Neutral",
               "annoyance": "Negative", "disappointment": "Negative", "disapproval": "Negative", "embarrassment": "Negative",
               "grief": "Negative",
               "nervousness": "Negative", "remorse": "Negative"}
    buckets = [mapping.get(s, "Neutral") for s in sentiments]
    cumulative = {"Positive": [], "Negative": [], "Neutral": []}
    pos = neg = neu = 0
    for b in buckets:
        if b == "Positive": pos += 1
        elif b == "Negative": neg += 1
        else: neu += 1
        cumulative["Positive"].append(pos)
        cumulative["Negative"].append(neg)
        cumulative["Neutral"].append(neu)

    xs = list(range(1, len(sentiments) + 1))
    fig = go.Figure()
    colors_trend = {"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#94a3b8"}
    for label, vals in cumulative.items():
        fig.add_trace(go.Scatter(
            x=xs, y=vals, mode="lines+markers",
            name=label,
            line=dict(color=colors_trend[label], width=2.5),
            marker=dict(size=5),
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8", family="DM Sans"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor="#1e3a5f", color="#64748b", title="Review #"),
        yaxis=dict(showgrid=True, gridcolor="#1e3a5f", color="#64748b", title="Cumulative Count"),
        margin=dict(t=10, b=20, l=10, r=10),
        height=280,
    )
    return fig


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: Syne, sans-serif; font-size:1.3rem; font-weight:800;
    background: linear-gradient(135deg,#38bdf8,#818cf8); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.2rem;'>SentioAI</div>
    <div style='font-size:0.75rem; color:#475569; margin-bottom:1.5rem;'>
    Emotion Intelligence Platform</div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Select Mode",
        ["🔍 Single Text Analyzer", "📊 Bulk Review Dashboard"],
        help="Single mode: analyze one text. Bulk mode: paste multiple reviews for full dashboard."
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#334155;'>Powered by DistilRoBERTa + Claude AI</div>", unsafe_allow_html=True)


# ─── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 SentioAI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Emotion Intelligence Platform — from text to business insights</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: SINGLE TEXT ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🔍 Single Text Analyzer":
    st.markdown('<div class="section-header">Analyze a Single Text</div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([1.2, 1], gap="large")

    with col_input:
        text = st.text_area("Enter your text", height=160,
                            placeholder="e.g. The delivery was terrible but the product itself exceeded my expectations.")
        if st.button("Analyze Emotion", use_container_width=True):
            if not text.strip():
                st.warning("⚠️ Please enter some text first.")
            else:
                with st.spinner("Analyzing..."):
                    emotion, score, all_results = detect_emotion(text)

                emoji = EMOJI_MAP.get(emotion.lower(), "🙂")
                st.session_state["single_result"] = (emotion, score, all_results, emoji, text)

    with col_result:
        if "single_result" in st.session_state:
            emotion, score, all_results, emoji, analyzed_text = st.session_state["single_result"]
            st.markdown(f"""
            <div class="metric-card" style="text-align:left; margin-top:0.5rem;">
                <div style="font-size:2.5rem; margin-bottom:0.2rem;">{emoji}</div>
                <div class="result-emotion" style="color:{EMOTION_COLORS.get(emotion.lower(),'#38bdf8')};">
                    {emotion.upper()}
                </div>
                <div style="color:#64748b; font-size:0.85rem; margin-top:0.3rem;">
                    Confidence: <span style="color:#38bdf8; font-weight:600;">{round(score*100,2)}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # All emotion scores bar
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            if isinstance(all_results[0], dict):
                sorted_r = sorted(all_results, key=lambda x: x['score'], reverse=True)
                for r in sorted_r:
                    em = r['label'].lower()
                    pct = round(r['score'] * 100, 1)
                    bar_color = EMOTION_COLORS.get(em, "#94a3b8")
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; margin-bottom:6px; gap:10px;">
                        <span style="width:70px; font-size:0.78rem; color:#94a3b8;">{EMOJI_MAP.get(em,'🙂')} {em}</span>
                        <div style="flex:1; background:#1e3a5f; border-radius:6px; height:8px;">
                            <div style="width:{pct}%; background:{bar_color}; height:8px; border-radius:6px;"></div>
                        </div>
                        <span style="width:45px; text-align:right; font-size:0.78rem; color:#64748b;">{pct}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Fake check for single text
            is_fake, reason = is_fake_review(analyzed_text, [analyzed_text])
            if is_fake:
                st.markdown(f'<div class="fake-badge">⚠️ SUSPICIOUS — {reason}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="clean-badge">✅ Looks Genuine</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: BULK REVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-header">Paste Reviews for Dashboard Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0f2240; border:1px solid #1e3a5f; border-radius:10px;
    padding:0.75rem 1.1rem; font-size:0.82rem; color:#64748b; margin-bottom:1rem;'>
    💡 <b style='color:#94a3b8;'>How to use:</b> Paste one review per line. Works with customer reviews, social media comments, survey responses, etc.
    </div>
    """, unsafe_allow_html=True)

    bulk_text = st.text_area(
        "Paste reviews (one per line)",
        height=180,
        placeholder="Great product! Really happy with it.\nDelivery took too long, very disappointed.\nThe battery drains fast. Not worth it.\nAbsolutely love the design and quality!"
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_btn = st.button("🚀 Run Full Analysis", use_container_width=True)

    if analyze_btn:
        lines = [l.strip() for l in bulk_text.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            st.warning("⚠️ Please enter at least 2 reviews (one per line).")
        else:
            with st.spinner(f"Analyzing {len(lines)} reviews..."):
                results = []
                for line in lines:
                    emotion, score, all_r = detect_emotion(line)
                    is_fake, reason = is_fake_review(line, lines)
                    results.append({
                        "text": line,
                        "emotion": emotion.lower(),
                        "score": score,
                        "is_fake": is_fake,
                        "fake_reason": reason,
                    })
            st.session_state["bulk_results"] = results
            st.session_state["bulk_lines"] = lines

    if "bulk_results" in st.session_state:
        results = st.session_state["bulk_results"]
        lines = st.session_state["bulk_lines"]
        total = len(results)
        fake_count = sum(1 for r in results if r["is_fake"])
        genuine_count = total - fake_count

        emotion_counts = Counter(r["emotion"] for r in results if not r["is_fake"])
        all_emotions = [r["emotion"] for r in results]
        keyword_freq = extract_keywords(lines)

        # ── TOP METRICS ─────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Dashboard Overview</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)

        pos_emotions = {
            "joy", "love", "surprise", "admiration", "amusement", "approval",
            "caring", "excitement", "gratitude", "optimism", "pride", "relief"
        }
        neg_emotions = {
            "anger", "sadness", "fear", "disgust", "annoyance", "disappointment",
            "disapproval", "embarrassment", "grief", "nervousness", "remorse", "repulsion"
        }
        pos_count = sum(1 for r in results if r["emotion"] in pos_emotions and not r["is_fake"])
        neg_count = sum(1 for r in results if r["emotion"] in neg_emotions and not r["is_fake"])
        top_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "—"
        top_emoji = EMOJI_MAP.get(top_emotion, "🙂")

        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total}</div><div class="metric-label">Total Reviews</div></div>', unsafe_allow_html=True)
        with m2:
            pct_pos = round(pos_count / genuine_count * 100) if genuine_count else 0
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#4ade80;">{pct_pos}%</div><div class="metric-label">Positive</div></div>', unsafe_allow_html=True)
        with m3:
            pct_neg = round(neg_count / genuine_count * 100) if genuine_count else 0
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f87171;">{pct_neg}%</div><div class="metric-label">Negative</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f87171;">{fake_count}</div><div class="metric-label">Flagged / Fake</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{top_emoji}</div><div class="metric-label">Top Emotion: {top_emotion.capitalize()}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── TAB LAYOUT ──────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment Charts", "🔑 Keywords", "🕵️ Fake Detection", "🤖 AI Insights"])

        # TAB 1 — CHARTS
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Emotion Distribution**")
                if emotion_counts:
                    st.plotly_chart(plot_emotion_donut(dict(emotion_counts)), use_container_width=True, config={"displayModeBar": False})
            with c2:
                st.markdown("**Emotion Breakdown**")
                if emotion_counts:
                    st.plotly_chart(plot_bar_emotions(dict(emotion_counts)), use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Sentiment Trend Across Reviews**")
            st.plotly_chart(plot_sentiment_trend(all_emotions), use_container_width=True, config={"displayModeBar": False})

        # TAB 2 — KEYWORDS
        with tab2:
            st.markdown("**Top Keywords Mentioned**")
            if keyword_freq:
                fig_kw = plot_keyword_bar(keyword_freq)
                if fig_kw:
                    st.plotly_chart(fig_kw, use_container_width=True, config={"displayModeBar": False})

            # Keyword frequency table
            st.markdown("**Frequency Table**")
            kw_df = pd.DataFrame(list(keyword_freq.items()), columns=["Keyword", "Mentions"])
            kw_df = kw_df.sort_values("Mentions", ascending=False).reset_index(drop=True)
            st.dataframe(kw_df, use_container_width=True, hide_index=True)

        # TAB 3 — FAKE DETECTION
        with tab3:
            st.markdown(f"""
            <div style="display:flex; gap:1rem; margin-bottom:1rem;">
                <div class="metric-card" style="flex:1;">
                    <div class="metric-value" style="color:#4ade80;">{genuine_count}</div>
                    <div class="metric-label">✅ Genuine Reviews</div>
                </div>
                <div class="metric-card" style="flex:1;">
                    <div class="metric-value" style="color:#f87171;">{fake_count}</div>
                    <div class="metric-label">⚠️ Flagged Reviews</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Review-by-Review Breakdown**")
            for i, r in enumerate(results):
                badge = f'<span class="fake-badge">⚠️ FLAGGED — {r["fake_reason"]}</span>' if r["is_fake"] else '<span class="clean-badge">✅ Genuine</span>'
                em_color = EMOTION_COLORS.get(r["emotion"], "#94a3b8")
                em_emoji = EMOJI_MAP.get(r["emotion"], "🙂")
                st.markdown(f"""
                <div class="review-row">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">
                        <div style="flex:1; color:#cbd5e1;">"{r['text']}"</div>
                        <div style="text-align:right; white-space:nowrap;">
                            <span style="color:{em_color}; font-weight:600; font-size:0.85rem;">{em_emoji} {r['emotion'].capitalize()} {round(r['score']*100,1)}%</span><br>
                            {badge}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # TAB 4 — AI INSIGHTS
        with tab4:
            st.markdown("**AI-Generated Business Intelligence Report**")
            st.markdown("""
            <div style='font-size:0.83rem; color:#475569; margin-bottom:1rem;'>
            Powered by Claude — transforms raw emotion data into actionable business insights.
            </div>
            """, unsafe_allow_html=True)

            if st.button("✨ Generate AI Insights", use_container_width=False):
                with st.spinner("Claude is analyzing your data..."):
                    try:
                        insight_text = generate_business_insights(
                            emotion_counts=dict(emotion_counts),
                            keyword_freq=keyword_freq,
                            fake_count=fake_count,
                            total=total
                        )
                        st.session_state["insight_text"] = insight_text
                    except Exception as e:
                        st.error(f"Could not generate insights: {e}")

            if "insight_text" in st.session_state:
                st.markdown(f'<div class="insight-box">🤖 {st.session_state["insight_text"]}</div>', unsafe_allow_html=True)

                # Download report
                report = f"""SENTIOAI BUSINESS INTELLIGENCE REPORT
======================================
Total Reviews Analyzed: {total}
Genuine Reviews: {genuine_count}
Flagged/Fake Reviews: {fake_count}

Positive Sentiment: {pct_pos}%
Negative Sentiment: {pct_neg}%

Emotion Breakdown:
{chr(10).join(f'  {k.capitalize()}: {v}' for k,v in emotion_counts.items())}

Top Keywords:
{chr(10).join(f'  {k}: {v}' for k,v in list(keyword_freq.items())[:10])}

AI INSIGHTS:
{st.session_state['insight_text']}
"""
                st.download_button(
                    "⬇️ Download Report (.txt)",
                    data=report,
                    file_name="sentioai_report.txt",
                    mime="text/plain"
                )
