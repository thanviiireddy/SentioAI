import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import re
from model import detect_emotion
from llm import analyze_emotion_with_llm
from insights import generate_business_insights
from config import GEMINI_API_KEY

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentioAI – Emotion-Aware Text Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500;700&display=swap');

/* Base styling */
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

/* Header typography */
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

/* Cards */
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
    margin: 1.8rem 0 1rem 0;
}

/* LLM Analysis Box */
.llm-box {
    background: linear-gradient(135deg, #0c1f3d 0%, #111827 100%);
    border: 1px solid #38bdf8;
    border-left: 4px solid #38bdf8;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #cbd5e1;
    margin-top: 1rem;
}

/* Badges */
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

/* Result typography */
.result-emotion {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Tabs */
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
    "got","will","can","about","after","before","some","any","each","over",
}

EMOJI_MAP = {
    "joy": "😊", "sadness": "😢", "anger": "😠",
    "fear": "😨", "love": "❤️", "surprise": "😲",
    "disgust": "🤢", "neutral": "😐",
}

EMOTION_COLORS = {
    "joy": "#facc15", "sadness": "#60a5fa", "anger": "#f87171",
    "fear": "#c084fc", "love": "#f472b6", "surprise": "#34d399",
    "disgust": "#a3e635", "neutral": "#94a3b8",
}

def extract_keywords(texts: list[str], top_n: int = 15) -> dict:
    words = []
    for t in texts:
        for w in re.findall(r'\b[a-zA-Z]{3,}\b', t.lower()):
            if w not in STOPWORDS:
                words.append(w)
    return dict(Counter(words).most_common(top_n))

def is_fake_review(text: str, all_texts: list[str]) -> tuple[bool, str]:
    text_lower = text.lower().strip()
    reasons = []
    if len(text.split()) < 4:
        reasons.append("too short")
    alpha = [c for c in text if c.isalpha()]
    if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.75:
        reasons.append("excessive caps")
    if text.count('!') >= 3:
        reasons.append("excessive exclamation marks")
    word_list = re.findall(r'\b\w+\b', text_lower)
    if word_list:
        freq = Counter(word_list)
        if freq.most_common(1)[0][1] / len(word_list) > 0.3 and len(word_list) > 3:
            reasons.append("repetitive words")
    duplicates = [t for t in all_texts if t.strip().lower() == text_lower and all_texts.count(t) > 1]
    if duplicates:
        reasons.append("duplicate review")
    spam_phrases = ["click here", "buy now", "free offer", "100% guaranteed", "visit our", "limited time"]
    for phrase in spam_phrases:
        if phrase in text_lower:
            reasons.append(f"spam phrase: '{phrase}'")
            break
    return len(reasons) > 0, ", ".join(reasons) if reasons else "clean"

def plot_emotion_donut(emotion_counts: dict):
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    colors = [EMOTION_COLORS.get(e, "#94a3b8") for e in labels]
    fig = go.Figure(go.Pie(
        labels=[f"{EMOJI_MAP.get(l,'🙂')} {l.capitalize()}" for l in labels],
        values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="#060b18", width=2)),
        textfont=dict(family="DM Sans", color="#e2e8f0"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8", family="DM Sans"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10), height=320,
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
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#64748b", tickfont=dict(family="DM Sans")),
        yaxis=dict(showgrid=True, gridcolor="#1e3a5f", color="#64748b"),
        margin=dict(t=20, b=20, l=10, r=10), height=300,
    )
    return fig

def plot_keyword_bar(keyword_freq: dict):
    if not keyword_freq: return None
    words = list(keyword_freq.keys())
    counts = list(keyword_freq.values())
    fig = go.Figure(go.Bar(
        y=words[::-1], x=counts[::-1], orientation='h',
        marker=dict(color=counts[::-1], colorscale=[[0, "#1e3a5f"], [1, "#38bdf8"]], showscale=False),
        text=counts[::-1], textposition="outside",
        textfont=dict(color="#e2e8f0", family="DM Sans"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#1e3a5f", color="#64748b"),
        yaxis=dict(showgrid=False, color="#94a3b8", tickfont=dict(family="DM Sans", size=12)),
        margin=dict(t=10, b=10, l=10, r=30), height=max(250, len(words) * 32),
    )
    return fig


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: Syne, sans-serif; font-size:1.4rem; font-weight:800;
    background: linear-gradient(135deg,#38bdf8,#818cf8); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; margin-bottom:0.2rem;'>SentioAI</div>
    <div style='font-size:0.78rem; color:#64748b; margin-bottom:1.5rem;'>
    Emotion-Aware Text Analyzer</div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Select Mode",
        ["🔍 Single Text Analyzer", "📊 Bulk Review Dashboard"],
        help="Single mode: classifier + Gemini LLM context. Bulk mode: batch analysis."
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#64748b;'>
    <b>Architecture:</b><br>
    • <b>Classifier:</b> DistilRoBERTa<br>
    • <b>LLM API:</b> Google Gemini 2.5 Flash
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 SentioAI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Emotion-Aware Text Analyzer — Combining DistilRoBERTa & Google Gemini LLM</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: SINGLE TEXT ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🔍 Single Text Analyzer":
    st.markdown('<div class="section-header">Analyze Text & Contextual Interpretation</div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        text = st.text_area(
            "Enter text to analyze",
            height=160,
            placeholder="e.g. I studied really hard for my exam but still failed. I feel completely useless."
        )
        analyze_click = st.button("Analyze Emotion", use_container_width=True)

        if analyze_click:
            if not text.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            else:
                # 1. DistilRoBERTa Emotion Classifier execution
                with st.spinner("Classifying primary emotion with DistilRoBERTa..."):
                    emotion, score, all_results = detect_emotion(text)
                
                # 2. Gemini LLM Contextual Analysis execution
                llm_response = None
                llm_error = None
                with st.spinner("Generating contextual analysis via Google Gemini API..."):
                    try:
                        llm_response = analyze_emotion_with_llm(text, emotion, score)
                    except Exception as e:
                        # Log error internally and prepare friendly message
                        err_msg = str(e)
                        if "API_KEY" in err_msg or "GEMINI_API_KEY" in err_msg:
                            llm_error = "Gemini API Key missing or unconfigured in .env file."
                        elif "quota" in err_msg.lower() or "resourceexhausted" in err_msg.lower():
                            llm_error = "Gemini API quota exceeded or rate limited."
                        else:
                            llm_error = f"API connection issue: {err_msg[:120]}"

                emoji = EMOJI_MAP.get(emotion.lower(), "🙂")
                st.session_state["single_result"] = {
                    "text": text,
                    "emotion": emotion,
                    "score": score,
                    "all_results": all_results,
                    "emoji": emoji,
                    "llm_response": llm_response,
                    "llm_error": llm_error
                }

    with col_result:
        if "single_result" in st.session_state:
            res = st.session_state["single_result"]
            emotion = res["emotion"]
            score = res["score"]
            all_results = res["all_results"]
            emoji = res["emoji"]
            llm_response = res["llm_response"]
            llm_error = res["llm_error"]

            st.markdown("### --------------------------------")
            st.markdown("### NLP EMOTION RESULT")
            st.markdown("### --------------------------------")

            # Main Result Card
            st.markdown(f"""
            <div class="metric-card" style="text-align:left; margin-top:0.5rem;">
                <div style="font-size:2.2rem; margin-bottom:0.2rem;">{emoji} Detected Emotion</div>
                <div class="result-emotion" style="color:{EMOTION_COLORS.get(emotion.lower(),'#38bdf8')};">
                    {emotion.upper()}
                </div>
                <div style="color:#94a3b8; font-size:0.95rem; margin-top:0.4rem;">
                    Confidence Score: <span style="color:#38bdf8; font-weight:700; font-size:1.1rem;">{round(score*100,2)}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # All Emotion Probabilities
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            if isinstance(all_results, list) and len(all_results) > 0 and isinstance(all_results[0], dict):
                sorted_r = sorted(all_results, key=lambda x: x['score'], reverse=True)
                for r in sorted_r[:5]:
                    em = r['label'].lower()
                    pct = round(r['score'] * 100, 1)
                    bar_color = EMOTION_COLORS.get(em, "#94a3b8")
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; margin-bottom:6px; gap:10px;">
                        <span style="width:85px; font-size:0.82rem; color:#94a3b8;">{EMOJI_MAP.get(em,'🙂')} {em.capitalize()}</span>
                        <div style="flex:1; background:#1e3a5f; border-radius:6px; height:8px;">
                            <div style="width:{pct}%; background:{bar_color}; height:8px; border-radius:6px;"></div>
                        </div>
                        <span style="width:50px; text-align:right; font-size:0.82rem; color:#64748b; font-weight:600;">{pct}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Gemini LLM Analysis Output Section ──
            st.markdown("### --------------------------------")
            st.markdown("### AI CONTEXTUAL ANALYSIS")
            st.markdown("### --------------------------------")

            if llm_response:
                st.markdown(f"""
                <div class="llm-box">
                    <div style="font-weight:700; color:#38bdf8; margin-bottom:0.6rem; font-size:1.05rem;">
                        ✨ Google Gemini LLM Interpretation
                    </div>
                    {llm_response}
                </div>
                """, unsafe_allow_html=True)
            elif llm_error:
                st.warning(f"⚠️ AI contextual analysis is currently unavailable.\n\n*{llm_error}*")
                st.info("💡 Note: DistilRoBERTa primary emotion detection above is fully operational. To enable Gemini LLM analysis, ensure a valid `GEMINI_API_KEY` is added to your `.env` file.")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: BULK REVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-header">Batch Review Dashboard</div>', unsafe_allow_html=True)

    bulk_text = st.text_area(
        "Paste multiple texts or reviews (one per line)",
        height=180,
        placeholder="I studied really hard for my exam but still failed. I feel completely useless.\nI am exhausted after working all night, but I'm really happy that I finally completed my project.\nGreat quality! Exceeded my expectations.\nItem arrived damaged and late. Very disappointed."
    )

    if st.button("🚀 Run Dashboard Analysis", use_container_width=True):
        lines = [l.strip() for l in bulk_text.strip().split('\n') if l.strip()]
        if len(lines) < 1:
            st.warning("⚠️ Please enter at least 1 text line.")
        else:
            with st.spinner(f"Classifying {len(lines)} texts..."):
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

        emotion_counts = Counter(r["emotion"] for r in results)
        keyword_freq = extract_keywords(lines)

        st.markdown('<div class="section-header">Summary Overview</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        top_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "—"
        top_emoji = EMOJI_MAP.get(top_emotion, "🙂")

        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total}</div><div class="metric-label">Total Texts</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#4ade80;">{genuine_count}</div><div class="metric-label">Clean Texts</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f87171;">{fake_count}</div><div class="metric-label">Flagged / Spam</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{top_emoji}</div><div class="metric-label">Top: {top_emotion.capitalize()}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Emotion Distribution", "🔑 Keywords", "🤖 LLM Batch Insights"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Emotion Proportion**")
                if emotion_counts:
                    st.plotly_chart(plot_emotion_donut(dict(emotion_counts)), use_container_width=True, config={"displayModeBar": False})
            with c2:
                st.markdown("**Emotion Counts**")
                if emotion_counts:
                    st.plotly_chart(plot_bar_emotions(dict(emotion_counts)), use_container_width=True, config={"displayModeBar": False})

        with tab2:
            st.markdown("**Frequent Keywords**")
            if keyword_freq:
                fig_kw = plot_keyword_bar(keyword_freq)
                if fig_kw:
                    st.plotly_chart(fig_kw, use_container_width=True, config={"displayModeBar": False})

        with tab3:
            st.markdown("**AI Batch Intelligence Report**")
            if st.button("✨ Generate Gemini Intelligence Summary"):
                with st.spinner("Gemini LLM is synthesizing batch insights..."):
                    try:
                        insight = generate_business_insights(dict(emotion_counts), keyword_freq, fake_count, total)
                        st.session_state["insight_text"] = insight
                    except Exception as e:
                        st.error(f"Could not generate insights: {e}")

            if "insight_text" in st.session_state:
                st.markdown(f'<div class="llm-box">🤖 <b>Gemini Summary:</b><br>{st.session_state["insight_text"]}</div>', unsafe_allow_html=True)
