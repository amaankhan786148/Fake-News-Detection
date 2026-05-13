# ============================================================
# app.py
# Fake News Detection — Streamlit Web Application
# B.Tech Major Project | Final Year
# ============================================================
# This is the main user-facing application.
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import pickle
import os
import re
import string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# ============================================================
# PAGE CONFIGURATION (must be the FIRST streamlit command)
# ============================================================
st.set_page_config(
    page_title="Fake News Detector | B.Tech Project",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL DARK THEME CSS
# Injected into the page using st.markdown + unsafe_allow_html
# ============================================================
DARK_CSS = """
<style>
/* ── Root / Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0E1117 !important;
    color: #E8EAF0 !important;
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%) !important;
    border-right: 1px solid #2D2D44;
}

/* ── Header strip ── */
.main-header {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 14px;
    border: 1px solid #2D3561;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,212,255,0.08);
}
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D4FF, #7B2FBE, #FF6B6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-subtitle {
    color: #8B8FA8;
    font-size: 1.0rem;
    margin-top: 0.4rem;
    letter-spacing: 0.5px;
}

/* ── Cards ── */
.card {
    background: #1A1A2E;
    border: 1px solid #2D2D44;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #00D4FF;
    letter-spacing: 0.5px;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
}

/* ── Result boxes ── */
.result-fake {
    background: linear-gradient(135deg, #2D1B1B, #3D1515);
    border: 2px solid #FF4444;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,68,68,0.25);
    margin: 1rem 0;
}
.result-real {
    background: linear-gradient(135deg, #1B2D1B, #153D15);
    border: 2px solid #44FF88;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(68,255,136,0.25);
    margin: 1rem 0;
}
.result-label {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: 2px;
}
.result-confidence {
    font-size: 1.15rem;
    opacity: 0.85;
    margin-top: 0.3rem;
}

/* ── Metric tiles ── */
.metric-tile {
    background: #1A1A2E;
    border: 1px solid #2D3561;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #00D4FF; }
.metric-label { font-size: 0.8rem; color: #8B8FA8; text-transform: uppercase; letter-spacing: 1px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0F3460, #533483);
    color: white !important;
    border: 1px solid #00D4FF44;
    border-radius: 10px;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: all 0.25s ease;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00D4FF22, #533483);
    border-color: #00D4FF;
    box-shadow: 0 0 18px rgba(0,212,255,0.3);
    transform: translateY(-1px);
}

/* ── Text area ── */
.stTextArea textarea {
    background-color: #1A1A2E !important;
    color: #E8EAF0 !important;
    border: 1px solid #2D3561 !important;
    border-radius: 10px !important;
    font-size: 0.97rem;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.2) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1A1A2E;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #8B8FA8 !important;
    border-radius: 8px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #0F3460 !important;
    color: #00D4FF !important;
}

/* ── Sidebar elements ── */
.sidebar-section {
    background: #0E1117;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    border: 1px solid #2D2D44;
}
.sidebar-heading {
    color: #00D4FF;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #00D4FF, #7B2FBE) !important;
    border-radius: 99px;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #1A1A2E !important;
    color: #E8EAF0 !important;
    border-radius: 8px;
}

/* ── DataFrames ── */
.dataframe { background: #1A1A2E !important; color: #E8EAF0 !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #444;
    font-size: 0.8rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #2D2D44;
    margin-top: 2rem;
}
</style>
"""


# ============================================================
# HELPER: Text Cleaning (same as in train_model.py)
# ============================================================
def clean_text(text: str) -> str:
    """
    Applies the same preprocessing pipeline used during training.
    IMPORTANT: The exact same cleaning must be applied to new inputs,
    otherwise the model will behave unpredictably.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# HELPER: Load Saved Models
# ============================================================
@st.cache_resource(show_spinner=False)
# ============================================================
# AUTO-TRAIN: If model files don't exist (e.g. on cloud),
# automatically run train_model.py before loading the app.
# This is needed for Streamlit Cloud deployment.
# ============================================================
import subprocess, sys

if not os.path.exists("models/pac_model.pkl"):
    with st.spinner("⚙️ First-time setup: Training model... please wait ~30 seconds"):
        subprocess.run([sys.executable, "train_model.py"], check=True)
    st.rerun()












def load_models():
    """
    Loads all saved pickle files.
    @st.cache_resource ensures models are loaded only ONCE
    and reused across every user interaction — much faster.
    Returns a dict with all loaded objects, or None on failure.
    """
    required = {
        "tfidf"   : "models/tfidf_vectorizer.pkl",
        "pac"     : "models/pac_model.pkl",
        "lr"      : "models/lr_model.pkl",
        "metadata": "models/metadata.pkl",
    }
    loaded = {}
    for key, path in required.items():
        if not os.path.exists(path):
            return None, key   # Return the missing file name for error display
        with open(path, 'rb') as f:
            loaded[key] = pickle.load(f)
    return loaded, None


# ============================================================
# HELPER: Prediction
# ============================================================
def predict_news(text: str, models: dict) -> dict:
    """
    Takes raw user input text and returns prediction results.
    Steps:
      1. Clean the text
      2. Transform using TF-IDF vectorizer
      3. Predict label using PAC (main model)
      4. Get confidence via Logistic Regression's predict_proba()
         (PAC has no native probability, LR does)
      5. Return structured result dict
    """
    cleaned = clean_text(text)

    if len(cleaned.split()) < 5:
        return {"error": "Please enter at least 5 meaningful words for accurate detection."}

    # Transform text to TF-IDF vector
    vectorized = models["tfidf"].transform([cleaned])

    # PAC prediction: 0 = FAKE, 1 = REAL
    pac_pred = models["pac"].predict(vectorized)[0]

    # Logistic Regression gives probability estimates
    lr_proba   = models["lr"].predict_proba(vectorized)[0]
    fake_prob  = round(lr_proba[0] * 100, 1)   # Probability of FAKE
    real_prob  = round(lr_proba[1] * 100, 1)   # Probability of REAL
    lr_pred    = models["lr"].predict(vectorized)[0]

    return {
        "label"      : "FAKE" if pac_pred == 0 else "REAL",
        "fake_prob"  : fake_prob,
        "real_prob"  : real_prob,
        "confidence" : fake_prob if pac_pred == 0 else real_prob,
        "pac_pred"   : pac_pred,
        "lr_pred"    : lr_pred,
        "lr_label"   : "FAKE" if lr_pred == 0 else "REAL",
        "word_count" : len(text.split()),
        "char_count" : len(text),
    }


# ============================================================
# HELPER: Matplotlib Plot — Confidence Bar Chart
# ============================================================
def plot_confidence_bar(fake_prob: float, real_prob: float) -> plt.Figure:
    """Creates a styled horizontal confidence bar chart."""
    fig, ax = plt.subplots(figsize=(6, 2))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#1A1A2E')

    bars = ax.barh(
        ['FAKE', 'REAL'],
        [fake_prob, real_prob],
        color=['#FF4444', '#44FF88'],
        height=0.5,
        edgecolor='none'
    )

    # Labels inside bars
    for bar, val in zip(bars, [fake_prob, real_prob]):
        ax.text(
            min(val + 1.5, 95),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va='center', color='white', fontsize=13, fontweight='bold'
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", color='#8B8FA8', fontsize=10)
    ax.tick_params(colors='white', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.set_title("Prediction Confidence (LR Probabilities)", color='white', fontsize=11, pad=10)

    plt.tight_layout()
    return fig


# ============================================================
# HELPER: Confusion Matrix Plot
# ============================================================
def plot_confusion_matrix(cm_data: list, title: str) -> plt.Figure:
    """Creates a styled seaborn heatmap for the confusion matrix."""
    import numpy as np
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#1A1A2E')

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['FAKE', 'REAL'],
        yticklabels=['FAKE', 'REAL'],
        linewidths=1,
        linecolor='#2D2D44',
        ax=ax,
        annot_kws={"size": 16, "color": "white", "weight": "bold"},
        cbar=False
    )

    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', color='#8B8FA8', fontsize=11)
    ax.set_ylabel('Actual Label', color='#8B8FA8', fontsize=11)
    ax.tick_params(colors='white', labelsize=11)
    plt.tight_layout()
    return fig


# ============================================================
# HELPER: Accuracy Comparison Plot
# ============================================================
def plot_accuracy_comparison(pac_acc: float, lr_acc: float) -> plt.Figure:
    """Grouped bar chart comparing PAC vs LR accuracy."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#1A1A2E')

    x      = ['Passive Aggressive\nClassifier', 'Logistic\nRegression']
    values = [pac_acc, lr_acc]
    colors = ['#00D4FF', '#7B2FBE']

    bars = ax.bar(x, values, color=colors, width=0.45, edgecolor='none')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Accuracy (%)', color='#8B8FA8', fontsize=11)
    ax.set_title('Model Accuracy Comparison', color='white', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='white', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')

    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}%",
            ha='center', color='white', fontsize=14, fontweight='bold'
        )

    plt.tight_layout()
    return fig


# ============================================================
# HELPER: Classification Report Table
# ============================================================
def render_classification_report(report: dict, model_name: str):
    """Renders a classification report dict as a styled Streamlit table."""
    import pandas as pd
    rows = []
    for cls in ['FAKE', 'REAL']:
        key = cls.lower() if cls.lower() in report else cls
        if key in report:
            r = report[key]
            rows.append({
                "Class"    : cls,
                "Precision": f"{r['precision']:.3f}",
                "Recall"   : f"{r['recall']:.3f}",
                "F1-Score" : f"{r['f1-score']:.3f}",
                "Support"  : int(r['support'])
            })
    if 'accuracy' in report:
        rows.append({
            "Class"    : "Accuracy",
            "Precision": "—",
            "Recall"   : "—",
            "F1-Score" : f"{report['accuracy']:.3f}",
            "Support"  : ""
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar(metadata: dict):
    """Renders the sidebar with project info and model stats."""
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:1rem 0;'>
            <div style='font-size:3rem;'>🔍</div>
            <div style='font-size:1.1rem; font-weight:800; color:#00D4FF;
                        letter-spacing:1px; margin-top:0.3rem;'>
                FAKE NEWS DETECTOR
            </div>
            <div style='font-size:0.75rem; color:#8B8FA8; margin-top:0.2rem;'>
                B.Tech Final Year Project
            </div>
        </div>
        <hr style='border-color:#2D2D44; margin:0.8rem 0;'>
        """, unsafe_allow_html=True)

        # ── Model Performance ──
        st.markdown('<div class="sidebar-heading">📊 Model Performance</div>', unsafe_allow_html=True)

        pac = metadata.get("pac_accuracy", 0)
        lr  = metadata.get("lr_accuracy",  0)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{pac:.0f}%</div>
                <div class="metric-label">PAC</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value">{lr:.0f}%</div>
                <div class="metric-label">LR</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Dataset Info ──
        st.markdown('<div class="sidebar-heading">📁 Dataset Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:0.88rem; color:#C0C4D0; line-height:2;'>
            🗂️ Total Samples: <b style='color:white'>{metadata.get('total_samples',0)}</b><br>
            🏋️ Training Set: <b style='color:white'>{metadata.get('training_samples',0)}</b><br>
            🧪 Testing Set:  <b style='color:white'>{metadata.get('testing_samples',0)}</b><br>
            📖 Vocab Size:   <b style='color:white'>{metadata.get('vocabulary_size',0):,}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tech Stack ──
        st.markdown('<div class="sidebar-heading">⚙️ Tech Stack</div>', unsafe_allow_html=True)
        tech = [
            ("🐍", "Python 3.10+"),
            ("📊", "Scikit-learn"),
            ("🌐", "Streamlit"),
            ("🔢", "TF-IDF Vectorizer"),
            ("⚡", "Passive Aggressive"),
            ("📈", "Logistic Regression"),
            ("🎨", "Matplotlib / Seaborn"),
        ]
        for icon, name in tech:
            st.markdown(
                f"<div style='font-size:0.85rem; color:#C0C4D0; padding:2px 0'>{icon} {name}</div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; font-size:0.75rem; color:#555; padding:0.5rem;
                    border-top:1px solid #2D2D44;'>
            Made with ❤️ for B.Tech Viva<br>
            Fake News Detection Project
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # Inject CSS
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # ── Load Models ──
    models, missing = load_models()

    if models is None:
        st.error("⚠️ Models not found!")
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🚀 First-Time Setup</div>
            <p>The trained model files were not found. Please run the training script first.</p>
            <br>
            <b>Open a terminal in the project folder and run:</b>
        </div>
        """, unsafe_allow_html=True)
        st.code("python train_model.py", language="bash")
        st.info("💡 This will train the models and save them to the `models/` folder.")
        return

    meta = models["metadata"]

    # ── Sidebar ──
    render_sidebar(meta)

    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🔍 Fake News Detection System</div>
        <div class="main-subtitle">
            Using Machine Learning · TF-IDF Vectorizer · Passive Aggressive Classifier
        </div>
        <div style='display:flex; gap:2rem; margin-top:1.2rem; flex-wrap:wrap;'>
            <span style='font-size:0.8rem; color:#8B8FA8;'>
                🎓 B.Tech Final Year Major Project
            </span>
            <span style='font-size:0.8rem; color:#8B8FA8;'>
                📚 Department of Computer Science &amp; Engineering
            </span>
            <span style='font-size:0.8rem; color:#8B8FA8;'>
                🤖 Natural Language Processing · ML Classification
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Detection",
        "📊 Model Analysis",
        "📈 Visualizations",
        "📚 Project Info"
    ])


    # ══════════════════════════════════════════════════════
    # TAB 1: DETECTION
    # ══════════════════════════════════════════════════════
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)

        left_col, right_col = st.columns([1.3, 1], gap="large")

        with left_col:
            st.markdown("""
            <div class="card-title" style="font-size:1.1rem; margin-bottom:0.8rem;">
                📝 Enter News Article
            </div>
            """, unsafe_allow_html=True)

            # Example selector
            example_option = st.selectbox(
                "Load an example (optional):",
                ["— Write your own text —",
                 "Example: Real News",
                 "Example: Fake News"],
                label_visibility="collapsed"
            )

            example_texts = {
                "Example: Real News": (
                    "The Indian Space Research Organisation (ISRO) successfully launched "
                    "the PSLV-C57 rocket carrying the Aditya-L1 solar observatory on September 2. "
                    "The spacecraft is now on its way to the Sun-Earth Lagrange Point 1, "
                    "approximately 1.5 million kilometres from Earth. Scientists at ISRO's "
                    "Telemetry, Tracking and Command Network confirmed that all systems are "
                    "functioning normally. This mission will study solar winds and their "
                    "impact on Earth's magnetosphere."
                ),
                "Example: Fake News": (
                    "SHOCKING REVELATION: Government scientists have CONFIRMED that the Earth "
                    "is actually hollow and ancient civilizations live inside! A whistleblower "
                    "from NASA has leaked classified documents proving that world leaders have "
                    "known about this for decades. The mainstream media is HIDING this truth. "
                    "Share this before it gets deleted! The global elite are using 5G towers "
                    "to control your mind and prevent you from learning the REAL facts!"
                )
            }

            default_text = example_texts.get(example_option, "")

            news_text = st.text_area(
                label="News Article",
                value=default_text,
                placeholder=(
                    "Paste or type a news article here...\n\n"
                    "Tips:\n"
                    "• Enter at least 2-3 sentences for best results\n"
                    "• Works for both English news articles and headlines\n"
                    "• The model analyses vocabulary patterns and writing style"
                ),
                height=260,
                label_visibility="collapsed"
            )

            # Word count display
            word_count = len(news_text.split()) if news_text.strip() else 0
            char_count = len(news_text)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Words",      word_count)
            col_b.metric("Characters", char_count)
            col_c.metric("Lines",      news_text.count('\n') + 1 if news_text.strip() else 0)

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔍 ANALYZE NEWS ARTICLE", use_container_width=True)

        with right_col:
            st.markdown("""
            <div class="card-title" style="font-size:1.1rem; margin-bottom:0.8rem;">
                📋 Detection Result
            </div>
            """, unsafe_allow_html=True)

            if predict_btn:
                if not news_text.strip():
                    st.warning("⚠️ Please enter some news text to analyze.")
                else:
                    with st.spinner("Analyzing article..."):
                        result = predict_news(news_text, models)

                    if "error" in result:
                        st.warning(f"⚠️ {result['error']}")
                    else:
                        label = result["label"]
                        conf  = result["confidence"]

                        # ── Verdict Box ──
                        if label == "FAKE":
                            st.markdown(f"""
                            <div class="result-fake">
                                <div style='font-size:3rem; margin-bottom:0.3rem;'>❌</div>
                                <div class="result-label" style='color:#FF4444;'>FAKE NEWS</div>
                                <div class="result-confidence" style='color:#FF8888;'>
                                    Confidence: {conf:.1f}%
                                </div>
                                <div style='font-size:0.82rem; color:#AA6666; margin-top:0.5rem;'>
                                    This article shows characteristics of misinformation.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-real">
                                <div style='font-size:3rem; margin-bottom:0.3rem;'>✅</div>
                                <div class="result-label" style='color:#44FF88;'>REAL NEWS</div>
                                <div class="result-confidence" style='color:#88FFB0;'>
                                    Confidence: {conf:.1f}%
                                </div>
                                <div style='font-size:0.82rem; color:#66AA77; margin-top:0.5rem;'>
                                    This article appears to be credible information.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # ── Confidence Bar Chart ──
                        fig_conf = plot_confidence_bar(result["fake_prob"], result["real_prob"])
                        st.pyplot(fig_conf, use_container_width=True)
                        plt.close(fig_conf)

                        # ── Progress bars ──
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Fake Probability:**")
                        st.progress(int(result["fake_prob"]))
                        st.markdown("**Real Probability:**")
                        st.progress(int(result["real_prob"]))

                        # ── Cross-model comparison ──
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander("🔬 Cross-Model Comparison", expanded=True):
                            m1, m2 = st.columns(2)
                            m1.metric(
                                "Passive Aggressive Classifier",
                                "FAKE" if result["pac_pred"] == 0 else "REAL",
                                "Main Model"
                            )
                            m2.metric(
                                "Logistic Regression",
                                result["lr_label"],
                                "Comparison Model"
                            )
                            agreement = result["pac_pred"] == result["lr_pred"]
                            if agreement:
                                st.success("✅ Both models agree on this prediction.")
                            else:
                                st.warning("⚠️ Models disagree — result may be uncertain.")

            else:
                # Placeholder when no prediction yet
                st.markdown("""
                <div style='display:flex; flex-direction:column; align-items:center;
                            justify-content:center; height:350px; opacity:0.4;'>
                    <div style='font-size:4rem;'>🤖</div>
                    <div style='font-size:1.1rem; margin-top:1rem;'>
                        Results will appear here
                    </div>
                    <div style='font-size:0.85rem; margin-top:0.4rem; color:#666;'>
                        Enter text and click Analyze
                    </div>
                </div>
                """, unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════
    # TAB 2: MODEL ANALYSIS
    # ══════════════════════════════════════════════════════
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Accuracy Cards ──
        st.markdown("#### 🏆 Model Accuracy")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Passive Aggressive Classifier",
                value=f"{meta['pac_accuracy']:.1f}%",
                delta="Main Model"
            )
        with c2:
            st.metric(
                label="Logistic Regression",
                value=f"{meta['lr_accuracy']:.1f}%",
                delta="Comparison Model"
            )
        with c3:
            diff = meta['pac_accuracy'] - meta['lr_accuracy']
            st.metric(
                label="Accuracy Difference",
                value=f"{abs(diff):.1f}%",
                delta=f"PAC {'better' if diff >= 0 else 'worse'}"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confusion Matrices ──
        st.markdown("#### 🔲 Confusion Matrices")
        cm1, cm2 = st.columns(2)
        with cm1:
            fig_cm_pac = plot_confusion_matrix(meta["pac_conf_matrix"], "PAC Model")
            st.pyplot(fig_cm_pac, use_container_width=True)
            plt.close(fig_cm_pac)
        with cm2:
            fig_cm_lr = plot_confusion_matrix(meta["lr_conf_matrix"], "Logistic Regression")
            st.pyplot(fig_cm_lr, use_container_width=True)
            plt.close(fig_cm_lr)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Classification Reports ──
        st.markdown("#### 📋 Classification Reports")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Passive Aggressive Classifier**")
            render_classification_report(meta["pac_report"], "PAC")
        with r2:
            st.markdown("**Logistic Regression**")
            render_classification_report(meta["lr_report"], "LR")

        # ── Understanding the Metrics ──
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📖 Understanding These Metrics"):
            st.markdown("""
| Metric | Description | Good Value |
|--------|-------------|------------|
| **Precision** | Of all articles predicted as FAKE, how many were actually FAKE? | Close to 1.0 |
| **Recall** | Of all actual FAKE articles, how many did the model correctly identify? | Close to 1.0 |
| **F1-Score** | Harmonic mean of Precision and Recall — balances both | Close to 1.0 |
| **Support** | Number of test samples in each class | — |
| **Accuracy** | Overall percentage of correct predictions | > 90% is excellent |

**Confusion Matrix explained:**
- **True Positive (TP)**: Correctly predicted REAL as REAL
- **True Negative (TN)**: Correctly predicted FAKE as FAKE  
- **False Positive (FP)**: Predicted REAL but was actually FAKE (Type I Error)
- **False Negative (FN)**: Predicted FAKE but was actually REAL (Type II Error)
            """)


    # ══════════════════════════════════════════════════════
    # TAB 3: VISUALIZATIONS
    # ══════════════════════════════════════════════════════
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Accuracy Comparison Chart ──
        st.markdown("#### 📊 Model Accuracy Comparison")
        v1, v2 = st.columns([1, 1.2])
        with v1:
            fig_acc = plot_accuracy_comparison(meta["pac_accuracy"], meta["lr_accuracy"])
            st.pyplot(fig_acc, use_container_width=True)
            plt.close(fig_acc)
        with v2:
            st.markdown("""
            <div class="card" style="height:100%;">
                <div class="card-title">📝 Interpretation</div>
                <ul style='color:#C0C4D0; font-size:0.92rem; line-height:2;'>
                    <li>Both models were trained on the same TF-IDF features</li>
                    <li>Passive Aggressive Classifier is faster and handles streaming data well</li>
                    <li>Logistic Regression provides calibrated probability outputs</li>
                    <li>Higher accuracy = better generalisation to unseen news articles</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Combined Metrics Chart ──
        st.markdown("#### 📈 Detailed Metrics Comparison")

        try:
            fig_metrics, ax = plt.subplots(figsize=(10, 5))
            fig_metrics.patch.set_facecolor('#1A1A2E')
            ax.set_facecolor('#1A1A2E')

            classes   = ['FAKE', 'REAL']
            metrics   = ['precision', 'recall', 'f1-score']
            x         = np.arange(len(classes))
            width     = 0.14
            offsets   = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
            colors    = ['#00D4FF', '#0090BB', '#005580', '#FF6B6B', '#BB4040', '#882020']
            labels    = ['PAC Precision','PAC Recall','PAC F1',
                         'LR Precision', 'LR Recall', 'LR F1']
            models_k  = ['pac_report', 'pac_report', 'pac_report',
                         'lr_report',  'lr_report',  'lr_report']
            metrics_k = ['precision', 'recall', 'f1-score',
                         'precision', 'recall', 'f1-score']

            for i, (model_k, metric_k, color, label, offset) in enumerate(
                    zip(models_k, metrics_k, colors, labels, offsets)):
                vals = [meta[model_k].get(cls, {}).get(metric_k, 0) * 100
                        for cls in classes]
                ax.bar(x + offset * width, vals, width * 0.9,
                       label=label, color=color, edgecolor='none')

            ax.set_xticks(x)
            ax.set_xticklabels(classes, color='white', fontsize=13)
            ax.set_ylim(0, 120)
            ax.set_ylabel('Score (%)', color='#8B8FA8', fontsize=11)
            ax.set_title('Precision / Recall / F1-Score — PAC vs LR',
                         color='white', fontsize=13, fontweight='bold', pad=12)
            ax.tick_params(colors='white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.legend(
                framealpha=0.2,
                facecolor='#1A1A2E',
                labelcolor='white',
                fontsize=9,
                ncol=3
            )
            plt.tight_layout()
            st.pyplot(fig_metrics, use_container_width=True)
            plt.close(fig_metrics)
        except Exception as e:
            st.warning(f"Could not render metrics chart: {e}")

        # ── Training Plot (if it exists) ──
        st.markdown("<br>", unsafe_allow_html=True)
        training_plot = os.path.join("models", "training_results.png")
        if os.path.exists(training_plot):
            st.markdown("#### 🖼️ Training Results Overview")
            st.image(training_plot, use_container_width=True)


    # ══════════════════════════════════════════════════════
    # TAB 4: PROJECT INFO
    # ══════════════════════════════════════════════════════
    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Why TF-IDF ──
        with st.expander("📖 Why TF-IDF Vectorizer?", expanded=True):
            st.markdown("""
**TF-IDF = Term Frequency × Inverse Document Frequency**

Machine learning models cannot work directly with raw text — they need numbers.  
TF-IDF converts each article into a numerical vector where each element represents  
the importance of a word to that specific document.

| Component | Formula | Meaning |
|-----------|---------|---------|
| **TF** | count(word) / total_words_in_doc | How often a word appears in ONE article |
| **IDF** | log(N / df(word)) | How rare a word is ACROSS ALL articles |
| **TF-IDF** | TF × IDF | High if word is frequent here but rare elsewhere |

**Why it beats simple word counts (Bag-of-Words):**
- Words like *"the"*, *"is"*, *"and"* appear in every article — TF-IDF downweights them automatically
- Distinctive words like *"microchip"*, *"conspiracy"*, *"BREAKING"* get higher scores in fake news
- Bigrams (two-word phrases) like *"mind control"* or *"proven false"* are captured together

**Advantages:** Fast, low memory, handles large vocabularies, no neural network needed.  
**Limitations:** Ignores word order and semantic meaning (e.g. "not good" vs "good").
            """)

        # ── Why Passive Aggressive Classifier ──
        with st.expander("⚡ Why Passive Aggressive Classifier?", expanded=True):
            st.markdown("""
**PAC is an online learning algorithm designed for large-scale text classification.**

The name explains the algorithm:
- 🟢 **Passive**: If the current prediction is correct (within the tolerance margin), **do nothing** — keep the current model weights.
- 🔴 **Aggressive**: If the prediction is wrong, **aggressively update** the model weights to correct the mistake as much as possible in one step.

**The update rule:**  
`w ← w + τ · y · x`  where `τ = loss / ||x||²`

**Why PAC for Fake News Detection:**
| Property | Benefit |
|----------|---------|
| **Online learning** | Can update with new news articles in real-time |
| **Speed** | Much faster than SVMs or neural networks |
| **Sparse data** | Works natively with TF-IDF sparse matrices |
| **No probability** | Compensated by using LR confidence scores alongside |
| **High accuracy** | Consistently outperforms naive Bayes on news classification |

**Advantages:** Fast convergence, handles high-dimensional text well, low memory usage.  
**Limitations:** No probability outputs, sensitive to noisy labels, not easily interpretable.
            """)

        # ── Why Logistic Regression ──
        with st.expander("📊 Why Logistic Regression (Comparison)?", expanded=True):
            st.markdown("""
**Logistic Regression is the standard baseline for binary text classification.**

- Uses the **sigmoid function** σ(z) = 1/(1+e⁻ᶻ) to map a linear score to a probability (0–1)
- Output: probability that the article is REAL (class=1)
- Decision boundary: if P(REAL) > 0.5 → predict REAL, else → predict FAKE
- We use LR's `predict_proba()` to show **confidence percentages** in the UI

**Why include it?**  
- Provides a **baseline accuracy** to compare PAC against
- Gives interpretable **probability estimates** (PAC does not)
- Well-understood mathematically — great for viva explanations!

**Advantages:** Interpretable, provides probabilities, regularized (L2), fast.  
**Limitations:** Assumes linear decision boundary, less competitive on very large datasets.
            """)

        # ── Project Architecture ──
        with st.expander("🏗️ Project Architecture", expanded=False):
            st.markdown("""
```
fake_news_detection/
├── app.py                  ← Main Streamlit web application (this file)
├── train_model.py          ← Model training and evaluation script
├── requirements.txt        ← Python package dependencies
├── README.md               ← Project documentation
│
├── dataset/
│   └── news.csv            ← Labelled dataset (REAL/FAKE news articles)
│
└── models/                 ← Generated after running train_model.py
    ├── tfidf_vectorizer.pkl ← Fitted TF-IDF vectorizer
    ├── pac_model.pkl        ← Trained Passive Aggressive Classifier
    ├── lr_model.pkl         ← Trained Logistic Regression model
    ├── metadata.pkl         ← Accuracy scores, confusion matrices, reports
    └── training_results.png ← Training visualization plot
```

**Data Flow:**
```
Raw Text Input
     ↓
Text Cleaning (lowercase, remove URLs/punctuation/digits)
     ↓
TF-IDF Vectorization (text → sparse numerical matrix)
     ↓
PAC Prediction  →  FAKE / REAL label
     ↓
LR Prediction   →  Confidence Probability (%)
     ↓
Streamlit UI Display
```
            """)

        # ── Viva Q&A ──
        with st.expander("🎓 Viva Voice Q&A Preparation", expanded=False):
            st.markdown("""
**Q1: What is the problem statement of your project?**  
A: The widespread spread of misinformation on social media and online platforms has become a major societal challenge. Our project builds an automated system to classify news articles as REAL or FAKE using Machine Learning, specifically TF-IDF vectorization combined with a Passive Aggressive Classifier.

**Q2: What datasets did you use?**  
A: We used a labelled CSV dataset containing news articles tagged as either REAL or FAKE. In production, this would be extended with large public datasets like LIAR or FakeNewsNet containing thousands of real-world articles.

**Q3: Why not use a Neural Network or BERT?**  
A: For this scope, traditional ML provides excellent accuracy (>90%) with far lower computational cost. BERT/transformers require GPUs and long training times, making them impractical for a project viva demo. TF-IDF + PAC achieves comparable results for English news classification tasks.

**Q4: What are the limitations of your system?**  
A: (1) The model is trained on specific vocabulary patterns — adversarially crafted fake news that mimics credible writing style may fool it. (2) Context and source credibility are not considered. (3) Performance degrades on non-English text. (4) The model can become outdated as language evolves.

**Q5: How is confidence percentage calculated?**  
A: We use Logistic Regression's `predict_proba()` method which applies the softmax function to the linear prediction scores, returning a probability distribution over both classes. For example, [0.82, 0.18] means 82% confidence that the article is FAKE.

**Q6: What improvements would you suggest for future work?**  
A: (1) Use BERT embeddings for semantic understanding. (2) Incorporate source credibility scoring. (3) Add web scraping to verify claims against trusted sources. (4) Deploy as a browser extension for real-time detection.
            """)

    # ── Footer ──
    st.markdown("""
    <div class="footer">
        🔍 Fake News Detection System · B.Tech Final Year Major Project ·
        Built with Python, Scikit-learn & Streamlit
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
