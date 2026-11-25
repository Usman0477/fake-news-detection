import streamlit as st
from predictor import load_model, predict_news

st.set_page_config(
    page_title="TruthLens | Fake News Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 80% 0%, rgba(0, 112, 255, .12), transparent 30%),
      radial-gradient(circle at 20% 80%, rgba(85, 35, 230, .12), transparent 32%),
      #050b1b;
    color: #f4f6ff;
}
.block-container { max-width: 1440px; padding: 1.5rem 1.8rem 4.5rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #121c47 0%, #09142e 72%, #101047 100%);
    border-right: 1px solid rgba(79, 146, 255, .45);
    min-width: 248px;
    max-width: 248px;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.15rem; }
.sidebar-brand { text-align: center; padding: .3rem 0 1.15rem; border-bottom: 1px solid rgba(145, 169, 255, .17); }
.sidebar-shield { font-size: 3.8rem; line-height: 1.2; filter: drop-shadow(0 0 14px rgba(0, 206, 255, .5)); }
.sidebar-brand h2 { margin: .2rem 0; color: white; font-family: 'Space Grotesk', sans-serif; }
.sidebar-brand h2 span { color: #21c9ff; }
.sidebar-brand p { color: #bdc9e7; font-size: .88rem; margin: 0; }
.side-heading { color: #eef2ff; font-weight: 800; margin: 1.1rem 0 .6rem; }
.steps { display: grid; gap: .75rem; }
.step { display: flex; align-items: flex-start; gap: .65rem; color: #d7e0f6; font-size: .9rem; }
.step b {
    min-width: 1.65rem; height: 1.65rem; border-radius: 50%; display: grid; place-items: center;
    color: white; background: linear-gradient(135deg, #8b45ff, #168dff);
    box-shadow: 0 0 12px rgba(120, 75, 255, .55);
}
.sidebar-about { color: #bdc9e7; font-size: .84rem; line-height: 1.55; }
.hero {
    min-height: 178px; position: relative; overflow: hidden; display: flex; align-items: center;
    gap: 1.25rem; padding: 1.6rem 2rem; border-radius: 24px;
    background:
      radial-gradient(circle at 90% 50%, rgba(0, 190, 255, .28), transparent 27%),
      linear-gradient(112deg, #5524c9 0%, #102765 48%, #063d71 100%);
    border: 1px solid rgba(93, 171, 255, .7);
    box-shadow: 0 18px 55px rgba(0, 28, 90, .35), inset 0 0 32px rgba(50, 124, 255, .12);
    margin-bottom: 1.45rem;
}
.hero-copy { position: relative; z-index: 1; }
.hero-icon { font-size: 4.5rem; line-height: 1; filter: drop-shadow(0 0 15px rgba(0, 201, 255, .45)); }
.hero-globe { position: absolute; right: 1.8rem; font-size: 8rem; opacity: .7; filter: drop-shadow(0 0 25px rgba(0, 183, 255, .5)); }
.eyebrow { color: #a4efff; font-size: .76rem; letter-spacing: .17em; font-weight: 800; }
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.4rem, 5vw, 3.7rem); margin: .2rem 0; color: white; }
.hero h1 span { color: #18c8ff; }
.hero p { color: #e0eaff; font-size: 1rem; max-width: 720px; margin: 0; }
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(18, 27, 55, .96), rgba(7, 17, 38, .96));
    border: 1px solid rgba(104, 119, 222, .34);
    border-radius: 20px;
    padding: 1rem 1.1rem;
    box-shadow: 0 14px 35px rgba(0, 0, 0, .18);
}
.metric {
    background: linear-gradient(145deg, #17264a, #10243b);
    border: 1px solid rgba(255,255,255,.09); border-radius: 16px;
    padding: 1rem; text-align: center;
}
.result-item { display: flex; gap: .7rem; margin: 1rem 0; color: #e0e8fa; line-height: 1.5; }
.result-icon {
    flex: 0 0 1.7rem; height: 1.7rem; border-radius: 50%; display: grid; place-items: center;
    color: white; font-weight: 800; background: linear-gradient(135deg, #00bd8b, #00a7bd);
}
.result-icon.real { background: linear-gradient(135deg, #248aff, #13bdd5); }
.result-fake {
    background: linear-gradient(110deg, rgba(255,83,110,.18), rgba(255,83,110,.06));
    border: 1px solid rgba(255,105,125,.45); border-radius: 18px; padding: 1.3rem;
}
.result-real {
    background: linear-gradient(110deg, rgba(0,220,170,.18), rgba(0,220,170,.05));
    border: 1px solid rgba(0,220,170,.4); border-radius: 18px; padding: 1.3rem;
}
.small-note { color: #aebbd1; font-size: .85rem; line-height: 1.5; }
div.stButton > button {
    background: linear-gradient(100deg, #9834f5, #04bde7); color: white;
    border: 0; border-radius: 999px; padding: .68rem 1.25rem;
    font-weight: 800; width: 100%; min-height: 2.8rem;
}
div.stButton > button:hover { color: white; border: 1px solid #66e4ff; filter: brightness(1.08); }
div.stButton > button[kind="secondary"] { background: transparent; border: 0; color: #d5def5; width: auto; }
[data-testid="stTextArea"] textarea {
    border-radius: 14px !important; background: #111b37; border: 1px solid #384878;
    color: #f4f6ff; min-height: 210px;
}
[data-testid="stTextArea"] textarea:focus { border-color: #1bcaff; box-shadow: 0 0 0 1px #1bcaff; }
h3 { color: #f6f7ff; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-icon">🔎</div>
  <div class="hero-copy">
    <div class="eyebrow">AI • NLP • MACHINE LEARNING</div>
    <h1>Truth<span>Lens</span></h1>
    <p>Analyze a news headline or article with a machine-learning model trained on the ISOT dataset.</p>
  </div>
  <div class="hero-globe">🌐</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-shield">🛡️</div>
      <h2>Truth<span>Lens</span></h2>
      <p>Your AI-powered news fact analyzer</p>
    </div>
    <div class="side-heading">💡 How to use</div>
    <div class="steps">
      <div class="step"><b>1</b><span>Paste a news headline or article.</span></div>
      <div class="step"><b>2</b><span>Click Analyze News.</span></div>
      <div class="step"><b>3</b><span>Review the prediction and model score.</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="side-heading">ⓘ About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-about">Model: TF-IDF (word + character features) with Logistic Regression.<br><br>'
        'Training data: ISOT Fake News dataset.<br><br>Built with ❤️ using AI & Machine Learning.</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.caption("⚠️ This is a text classifier, not a professional fact-checking service.")

@st.cache_resource
def get_model():
    return load_model()


def clear_news_text():
    st.session_state["news_text"] = ""


try:
    model = get_model()
    model_ready = True
except FileNotFoundError:
    model_ready = False
    st.error("Model file not found. In VS Code terminal, run `python train_model.py` first.")

left, right = st.columns([1.65, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown("### 📰 Enter news to analyze")
        st.caption("Paste the headline and/or full article text. Longer context can help the classifier.")
        news = st.text_area(
            "News text",
            height=210,
            placeholder="e.g. Paste a news headline or article here...",
            label_visibility="collapsed",
            key="news_text"
        )
        c1, c2 = st.columns([1.65, 1])
        with c1:
            st.button("🗑 Clear", type="secondary", on_click=clear_news_text)
        with c2:
            analyze = st.button("▶ Analyze News", type="primary", disabled=not model_ready)

with right:
    with st.container(border=True):
        st.markdown("### 📊 What the result means")
        st.markdown("""
        <div class="result-item"><span class="result-icon">✓</span><span><b>FAKE:</b> The text resembles fake-labeled examples in the training data.</span></div>
        <div class="result-item"><span class="result-icon real">✓</span><span><b>REAL:</b> The text resembles real-labeled examples in the training data.</span></div>
        <p class="small-note">A high score is not proof that a claim is true or false. Always verify important claims with independent, trusted sources.</p>
        """, unsafe_allow_html=True)

if analyze:
    if not news.strip():
        st.warning("Please paste some news text first.")
    else:
        try:
            with st.spinner("Analyzing text..."):
                result = predict_news(model, news)

            st.markdown("## 🎯 Analysis result")
            css_class = "result-fake" if result["label"] == "FAKE" else "result-real"
            icon = "🚨" if result["label"] == "FAKE" else "✅"
            st.markdown(
                f'<div class="{css_class}"><h2>{icon} {result["label"]} NEWS</h2>'
                f'<p>Model score: <b>{result["confidence"]:.1%}</b></p></div>',
                unsafe_allow_html=True
            )
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Fake score", f'{result["fake_probability"]:.1%}')
                st.progress(result["fake_probability"])
            with m2:
                st.metric("Real score", f'{result["real_probability"]:.1%}')
                st.progress(result["real_probability"])

            st.info(
                "Interpretation: this prediction reflects patterns learned from the ISOT dataset. "
                "It does not independently verify the events, people, dates, or claims in the article."
            )
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")
st.markdown(
    '<p class="small-note" style="text-align:center">TRUTHLENS · Academic project prototype · '
    'Use independent fact-checking for important decisions.</p>',
    unsafe_allow_html=True
)
