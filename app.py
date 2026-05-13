# ============================================================
# app.py — Fake News Detection (Cloud-Ready Single File)
# B.Tech Final Year Major Project
# ============================================================
import streamlit as st

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)
import streamlit as st
import pickle, os, re, string, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector | B.Tech Project",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#0E1117!important;color:#E8EAF0!important;font-family:'Segoe UI',sans-serif}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1A1A2E,#16213E)!important;border-right:1px solid #2D2D44}
.main-header{background:linear-gradient(135deg,#1A1A2E,#16213E,#0F3460);padding:2rem 2.5rem;border-radius:14px;border:1px solid #2D3561;margin-bottom:1.5rem;box-shadow:0 8px 32px rgba(0,212,255,.08)}
.main-title{font-size:2.6rem;font-weight:800;background:linear-gradient(90deg,#00D4FF,#7B2FBE,#FF6B6B);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0}
.main-subtitle{color:#8B8FA8;font-size:1rem;margin-top:.4rem}
.result-fake{background:linear-gradient(135deg,#2D1B1B,#3D1515);border:2px solid #FF4444;border-radius:14px;padding:1.8rem;text-align:center;box-shadow:0 0 30px rgba(255,68,68,.25);margin:1rem 0}
.result-real{background:linear-gradient(135deg,#1B2D1B,#153D15);border:2px solid #44FF88;border-radius:14px;padding:1.8rem;text-align:center;box-shadow:0 0 30px rgba(68,255,136,.25);margin:1rem 0}
.metric-tile{background:#1A1A2E;border:1px solid #2D3561;border-radius:10px;padding:1rem;text-align:center}
.metric-value{font-size:2rem;font-weight:800;color:#00D4FF}
.metric-label{font-size:.8rem;color:#8B8FA8;text-transform:uppercase;letter-spacing:1px}
.stButton>button{background:linear-gradient(135deg,#0F3460,#533483);color:white!important;border:1px solid #00D4FF44;border-radius:10px;font-size:1.05rem;font-weight:700;padding:.65rem 2rem;width:100%}
.stTextArea textarea{background-color:#1A1A2E!important;color:#E8EAF0!important;border:1px solid #2D3561!important;border-radius:10px!important}
.stTabs [data-baseweb="tab-list"]{background:#1A1A2E;border-radius:10px;gap:4px;padding:4px}
.stTabs [data-baseweb="tab"]{color:#8B8FA8!important;border-radius:8px;font-weight:600}
.stTabs [aria-selected="true"]{background:#0F3460!important;color:#00D4FF!important}
.footer{text-align:center;color:#444;font-size:.8rem;padding:1.5rem 0 .5rem;border-top:1px solid #2D2D44;margin-top:2rem}
</style>
""", unsafe_allow_html=True)


# ============================================================
# STEP 1 — Text Cleaning
# ============================================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================
# STEP 2 — Train & Save Models (runs once on first launch)
# ============================================================
def train_and_save():
    """Trains TF-IDF + PAC + LR and saves to models/ folder."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Try PassiveAggressiveClassifier; fall back to SGDClassifier equivalent
    try:
        from sklearn.linear_model import PassiveAggressiveClassifier
        pac_model = PassiveAggressiveClassifier(C=0.5, max_iter=1000, random_state=42, tol=1e-4)
    except Exception:
        from sklearn.linear_model import SGDClassifier
        pac_model = SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1',
                                  eta0=0.5, max_iter=1000, random_state=42, tol=1e-4)

    # Load dataset
    df = pd.read_csv(os.path.join("dataset", "news.csv"))
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    df['label_encoded'] = df['label'].map({'REAL': 1, 'FAKE': 0})

    X = df['clean_text']
    y = df['label_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # TF-IDF
    tfidf = TfidfVectorizer(max_df=0.7, min_df=1, max_features=5000,
                            ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
    X_train_v = tfidf.fit_transform(X_train)
    X_test_v  = tfidf.transform(X_test)

    # PAC
    pac_model.fit(X_train_v, y_train)
    pac_pred   = pac_model.predict(X_test_v)
    pac_acc    = accuracy_score(y_test, pac_pred)
    pac_cm     = confusion_matrix(y_test, pac_pred)
    pac_report = classification_report(y_test, pac_pred,
                                       target_names=['FAKE','REAL'], output_dict=True)

    # Logistic Regression
    lr_model  = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
    lr_model.fit(X_train_v, y_train)
    lr_pred   = lr_model.predict(X_test_v)
    lr_acc    = accuracy_score(y_test, lr_pred)
    lr_cm     = confusion_matrix(y_test, lr_pred)
    lr_report = classification_report(y_test, lr_pred,
                                      target_names=['FAKE','REAL'], output_dict=True)

    # Save
    os.makedirs("models", exist_ok=True)
    for obj, fname in [(tfidf,     "tfidf_vectorizer.pkl"),
                       (pac_model, "pac_model.pkl"),
                       (lr_model,  "lr_model.pkl")]:
        with open(f"models/{fname}", 'wb') as f:
            pickle.dump(obj, f)

    metadata = {
        "pac_accuracy": round(pac_acc * 100, 2),
        "lr_accuracy" : round(lr_acc  * 100, 2),
        "pac_conf_matrix": pac_cm.tolist(),
        "lr_conf_matrix" : lr_cm.tolist(),
        "pac_report": pac_report,
        "lr_report" : lr_report,
        "vocabulary_size"  : len(tfidf.vocabulary_),
        "training_samples" : len(X_train),
        "testing_samples"  : len(X_test),
        "total_samples"    : len(df),
    }
    with open("models/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)

    return metadata


# ============================================================
# STEP 3 — Load Models (cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_models():
    files = ["models/tfidf_vectorizer.pkl","models/pac_model.pkl",
             "models/lr_model.pkl","models/metadata.pkl"]
    if not all(os.path.exists(p) for p in files):
        return None
    result = {}
    for path in files:
        key = os.path.basename(path).replace(".pkl","").replace("tfidf_vectorizer","tfidf").replace("pac_model","pac").replace("lr_model","lr")
        with open(path,'rb') as f:
            result[key] = pickle.load(f)
    return result


# ============================================================
# STEP 4 — Prediction
# ============================================================
def predict_news(text, models):
    cleaned = clean_text(text)
    if len(cleaned.split()) < 5:
        return {"error": "Please enter at least 5 meaningful words."}
    vec      = models["tfidf"].transform([cleaned])
    pac_pred = models["pac"].predict(vec)[0]
    lr_proba = models["lr"].predict_proba(vec)[0]
    lr_pred  = models["lr"].predict(vec)[0]
    fake_p   = round(lr_proba[0]*100, 1)
    real_p   = round(lr_proba[1]*100, 1)
    return {
        "label"     : "FAKE" if pac_pred == 0 else "REAL",
        "fake_prob" : fake_p,
        "real_prob" : real_p,
        "confidence": fake_p if pac_pred==0 else real_p,
        "pac_pred"  : pac_pred,
        "lr_pred"   : lr_pred,
        "lr_label"  : "FAKE" if lr_pred==0 else "REAL",
    }


# ============================================================
# STEP 5 — Plot Helpers
# ============================================================
def plot_confidence(fake_p, real_p):
    fig, ax = plt.subplots(figsize=(6,2))
    fig.patch.set_facecolor('#1A1A2E'); ax.set_facecolor('#1A1A2E')
    bars = ax.barh(['FAKE','REAL'],[fake_p,real_p],color=['#FF4444','#44FF88'],height=0.5)
    for bar,val in zip(bars,[fake_p,real_p]):
        ax.text(min(val+1.5,92), bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va='center', color='white', fontsize=13, fontweight='bold')
    ax.set_xlim(0,110); ax.tick_params(colors='white',labelsize=12)
    ax.set_xlabel("Confidence (%)",color='#8B8FA8',fontsize=10)
    ax.set_title("Prediction Confidence",color='white',fontsize=11,pad=8)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['bottom','left']: ax.spines[s].set_color('#444')
    plt.tight_layout(); return fig

def plot_confusion(cm_data, title):
    fig, ax = plt.subplots(figsize=(5,4))
    fig.patch.set_facecolor('#1A1A2E'); ax.set_facecolor('#1A1A2E')
    sns.heatmap(np.array(cm_data), annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE','REAL'], yticklabels=['FAKE','REAL'],
                linewidths=1, linecolor='#2D2D44', ax=ax, cbar=False,
                annot_kws={"size":16,"color":"white","weight":"bold"})
    ax.set_title(title,color='white',fontsize=13,fontweight='bold',pad=12)
    ax.set_xlabel('Predicted',color='#8B8FA8',fontsize=11)
    ax.set_ylabel('Actual',color='#8B8FA8',fontsize=11)
    ax.tick_params(colors='white',labelsize=11)
    plt.tight_layout(); return fig

def plot_accuracy(pac_acc, lr_acc):
    fig, ax = plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor('#1A1A2E'); ax.set_facecolor('#1A1A2E')
    bars = ax.bar(['Passive Aggressive\nClassifier','Logistic\nRegression'],
                  [pac_acc,lr_acc], color=['#00D4FF','#7B2FBE'], width=0.45)
    ax.set_ylim(0,115)
    ax.set_ylabel('Accuracy (%)',color='#8B8FA8',fontsize=11)
    ax.set_title('Model Accuracy Comparison',color='white',fontsize=13,fontweight='bold',pad=12)
    ax.tick_params(colors='white',labelsize=11)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['bottom','left']: ax.spines[s].set_color('#444')
    for bar,val in zip(bars,[pac_acc,lr_acc]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                f"{val:.1f}%", ha='center', color='white', fontsize=14, fontweight='bold')
    plt.tight_layout(); return fig

def render_report(report, name):
    rows = []
    for cls in ['FAKE','REAL']:
        r = report.get(cls, report.get(cls.lower(), {}))
        if r:
            rows.append({"Class":cls,"Precision":f"{r['precision']:.3f}",
                         "Recall":f"{r['recall']:.3f}","F1":f"{r['f1-score']:.3f}",
                         "Support":int(r['support'])})
    if 'accuracy' in report:
        rows.append({"Class":"Accuracy","Precision":"—","Recall":"—",
                     "F1":f"{report['accuracy']:.3f}","Support":""})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================
# MAIN APP
# ============================================================
def main():

    # ── Auto-train if models missing ──────────────────────────
    if not os.path.exists("models/pac_model.pkl"):
        with st.spinner("⚙️ First-time setup: Training model... (~30 seconds)"):
            try:
                meta_new = train_and_save()
                st.cache_resource.clear()
                st.success("✅ Model trained successfully! Loading app...")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()

    # ── Load models ───────────────────────────────────────────
    models = load_models()
    if models is None:
        st.error("Could not load models. Please check the logs.")
        st.stop()

    meta = models["metadata"]

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0'>
            <div style='font-size:3rem'>🔍</div>
            <div style='font-size:1.1rem;font-weight:800;color:#00D4FF;letter-spacing:1px;margin-top:.3rem'>
                FAKE NEWS DETECTOR</div>
            <div style='font-size:.75rem;color:#8B8FA8;margin-top:.2rem'>B.Tech Final Year Project</div>
        </div>
        <hr style='border-color:#2D2D44;margin:.8rem 0'>
        """, unsafe_allow_html=True)

        st.markdown("**📊 Model Performance**")
        c1,c2 = st.columns(2)
        c1.markdown(f"<div class='metric-tile'><div class='metric-value'>{meta['pac_accuracy']:.0f}%</div><div class='metric-label'>PAC</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-tile'><div class='metric-value'>{meta['lr_accuracy']:.0f}%</div><div class='metric-label'>LR</div></div>", unsafe_allow_html=True)

        st.markdown("<br>**📁 Dataset Info**", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:.88rem;color:#C0C4D0;line-height:2'>
        🗂️ Total: <b style='color:white'>{meta['total_samples']}</b><br>
        🏋️ Train: <b style='color:white'>{meta['training_samples']}</b><br>
        🧪 Test:  <b style='color:white'>{meta['testing_samples']}</b><br>
        📖 Vocab: <b style='color:white'>{meta['vocabulary_size']:,}</b>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>**⚙️ Tech Stack**", unsafe_allow_html=True)
        for icon,name in [("🐍","Python 3"),("📊","Scikit-learn"),("🌐","Streamlit"),
                          ("🔢","TF-IDF"),("⚡","Passive Aggressive"),("📈","Logistic Regression")]:
            st.markdown(f"<div style='font-size:.85rem;color:#C0C4D0;padding:2px 0'>{icon} {name}</div>", unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>🔍 Fake News Detection System</div>
        <div class='main-subtitle'>TF-IDF Vectorizer · Passive Aggressive Classifier · Logistic Regression</div>
        <div style='display:flex;gap:2rem;margin-top:1.2rem;flex-wrap:wrap'>
            <span style='font-size:.8rem;color:#8B8FA8'>🎓 B.Tech Final Year Major Project</span>
            <span style='font-size:.8rem;color:#8B8FA8'>📚 Computer Science & Engineering</span>
            <span style='font-size:.8rem;color:#8B8FA8'>🤖 NLP · ML Classification</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detection","📊 Model Analysis","📈 Visualizations","📚 Project Info"])

    # ══ TAB 1: DETECTION ══════════════════════════════════════
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1.3,1], gap="large")

        with left:
            st.markdown("**📝 Enter News Article**")
            example = st.selectbox("Load example:", ["— Write your own —","Example: Real News","Example: Fake News"], label_visibility="collapsed")
            examples = {
                "Example: Real News": "The Indian Space Research Organisation successfully launched the Chandrayaan-3 lunar mission. The spacecraft entered lunar orbit and the lander Vikram successfully touched down near the Moon's south pole, making India the first country to achieve a soft landing at this location. Scientists at ISRO confirmed all systems are operating normally.",
                "Example: Fake News": "SHOCKING TRUTH EXPOSED! Government scientists have CONFIRMED that 5G towers are mind control devices installed by the New World Order! A whistleblower from NASA leaked classified documents proving world leaders are actually reptilian aliens hiding in underground bunkers. SHARE before this gets deleted! Big Pharma is hiding the CURE for all diseases!"
            }
            news_text = st.text_area("Article", value=examples.get(example,""),
                placeholder="Paste or type a news article here...\n\nTips:\n• Enter at least 2-3 sentences\n• Works for English news articles\n• Model analyses vocabulary patterns",
                height=260, label_visibility="collapsed")

            wc = len(news_text.split()) if news_text.strip() else 0
            a,b,c = st.columns(3)
            a.metric("Words", wc)
            b.metric("Characters", len(news_text))
            c.metric("Lines", news_text.count('\n')+1 if news_text.strip() else 0)

            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("🔍 ANALYZE NEWS ARTICLE", use_container_width=True)

        with right:
            st.markdown("**📋 Detection Result**")
            if btn:
                if not news_text.strip():
                    st.warning("⚠️ Please enter some text.")
                else:
                    with st.spinner("Analyzing..."):
                        res = predict_news(news_text, models)
                    if "error" in res:
                        st.warning(f"⚠️ {res['error']}")
                    else:
                        if res["label"] == "FAKE":
                            st.markdown(f"""<div class='result-fake'>
                                <div style='font-size:3rem'>❌</div>
                                <div style='font-size:2.4rem;font-weight:900;color:#FF4444;letter-spacing:2px'>FAKE NEWS</div>
                                <div style='font-size:1.1rem;color:#FF8888'>Confidence: {res['confidence']:.1f}%</div>
                                <div style='font-size:.82rem;color:#AA6666;margin-top:.5rem'>Shows characteristics of misinformation.</div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""<div class='result-real'>
                                <div style='font-size:3rem'>✅</div>
                                <div style='font-size:2.4rem;font-weight:900;color:#44FF88;letter-spacing:2px'>REAL NEWS</div>
                                <div style='font-size:1.1rem;color:#88FFB0'>Confidence: {res['confidence']:.1f}%</div>
                                <div style='font-size:.82rem;color:#66AA77;margin-top:.5rem'>Appears to be credible information.</div>
                            </div>""", unsafe_allow_html=True)

                        fig = plot_confidence(res["fake_prob"], res["real_prob"])
                        st.pyplot(fig, use_container_width=True); plt.close(fig)

                        st.markdown("**Fake Probability:**"); st.progress(int(res["fake_prob"]))
                        st.markdown("**Real Probability:**"); st.progress(int(res["real_prob"]))

                        with st.expander("🔬 Cross-Model Comparison", expanded=True):
                            m1,m2 = st.columns(2)
                            m1.metric("Passive Aggressive", "FAKE" if res["pac_pred"]==0 else "REAL", "Main Model")
                            m2.metric("Logistic Regression", res["lr_label"], "Comparison")
                            if res["pac_pred"] == res["lr_pred"]:
                                st.success("✅ Both models agree.")
                            else:
                                st.warning("⚠️ Models disagree — result may be uncertain.")
            else:
                st.markdown("""<div style='display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:350px;opacity:.4'>
                    <div style='font-size:4rem'>🤖</div>
                    <div style='font-size:1.1rem;margin-top:1rem'>Results will appear here</div>
                    <div style='font-size:.85rem;margin-top:.4rem;color:#666'>Enter text and click Analyze</div>
                </div>""", unsafe_allow_html=True)

    # ══ TAB 2: MODEL ANALYSIS ══════════════════════════════════
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🏆 Model Accuracy")
        c1,c2,c3 = st.columns(3)
        c1.metric("Passive Aggressive Classifier", f"{meta['pac_accuracy']:.1f}%", "Main Model")
        c2.metric("Logistic Regression", f"{meta['lr_accuracy']:.1f}%", "Comparison")
        diff = meta['pac_accuracy'] - meta['lr_accuracy']
        c3.metric("Difference", f"{abs(diff):.1f}%", f"PAC {'better' if diff>=0 else 'worse'}")

        st.markdown("<br>#### 🔲 Confusion Matrices")
        cm1,cm2 = st.columns(2)
        with cm1:
            fig = plot_confusion(meta["pac_conf_matrix"],"PAC Model")
            st.pyplot(fig,use_container_width=True); plt.close(fig)
        with cm2:
            fig = plot_confusion(meta["lr_conf_matrix"],"Logistic Regression")
            st.pyplot(fig,use_container_width=True); plt.close(fig)

        st.markdown("<br>#### 📋 Classification Reports")
        r1,r2 = st.columns(2)
        with r1:
            st.markdown("**Passive Aggressive Classifier**")
            render_report(meta["pac_report"],"PAC")
        with r2:
            st.markdown("**Logistic Regression**")
            render_report(meta["lr_report"],"LR")

    # ══ TAB 3: VISUALIZATIONS ══════════════════════════════════
    with tab3:
        st.markdown("<br>#### 📊 Accuracy Comparison")
        v1,v2 = st.columns([1,1.2])
        with v1:
            fig = plot_accuracy(meta["pac_accuracy"], meta["lr_accuracy"])
            st.pyplot(fig,use_container_width=True); plt.close(fig)
        with v2:
            st.markdown("""
            **Key Observations:**
            - Both models trained on identical TF-IDF features
            - PAC is faster and handles live/streaming data
            - LR gives calibrated probability (confidence %)
            - Higher accuracy = better generalisation
            """)

        st.markdown("<br>#### 📈 Precision / Recall / F1 Comparison")
        try:
            fig2, ax = plt.subplots(figsize=(10,5))
            fig2.patch.set_facecolor('#1A1A2E'); ax.set_facecolor('#1A1A2E')
            classes = ['FAKE','REAL']
            x = np.arange(len(classes))
            w = 0.13
            specs = [
                ("pac_report","precision","#00D4FF","PAC Precision",-2.5),
                ("pac_report","recall","#0090BB","PAC Recall",-1.5),
                ("pac_report","f1-score","#005580","PAC F1",-0.5),
                ("lr_report","precision","#FF6B6B","LR Precision",0.5),
                ("lr_report","recall","#BB4040","LR Recall",1.5),
                ("lr_report","f1-score","#882020","LR F1",2.5),
            ]
            for mk,metric,color,label,off in specs:
                vals = [meta[mk].get(cls,meta[mk].get(cls.lower(),{})).get(metric,0)*100 for cls in classes]
                ax.bar(x+off*w, vals, w*0.9, label=label, color=color, edgecolor='none')
            ax.set_xticks(x); ax.set_xticklabels(classes,color='white',fontsize=13)
            ax.set_ylim(0,120); ax.set_ylabel('Score (%)',color='#8B8FA8',fontsize=11)
            ax.set_title('Precision / Recall / F1 — PAC vs LR',color='white',fontsize=13,fontweight='bold',pad=12)
            ax.tick_params(colors='white')
            for s in ['top','right']: ax.spines[s].set_visible(False)
            for s in ['bottom','left']: ax.spines[s].set_color('#444')
            ax.legend(framealpha=.2,facecolor='#1A1A2E',labelcolor='white',fontsize=9,ncol=3)
            plt.tight_layout()
            st.pyplot(fig2,use_container_width=True); plt.close(fig2)
        except Exception as e:
            st.warning(f"Chart error: {e}")

    # ══ TAB 4: PROJECT INFO ════════════════════════════════════
    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📖 Why TF-IDF Vectorizer?", expanded=True):
            st.markdown("""
**TF-IDF = Term Frequency × Inverse Document Frequency**

ML models cannot process raw text — they need numbers. TF-IDF converts each article into a vector where each element represents how important a word is for that specific document.

| Component | Meaning |
|-----------|---------|
| **TF** | How often a word appears in ONE article |
| **IDF** | How rare a word is across ALL articles |
| **TF-IDF** | High score = frequent here but rare elsewhere |

**Advantages:** Fast, low memory, handles large vocabularies.
**Limitations:** Ignores word order and semantic meaning.
            """)

        with st.expander("⚡ Why Passive Aggressive Classifier?", expanded=True):
            st.markdown("""
**PAC is an online learning algorithm for large-scale text classification.**

- 🟢 **Passive**: Correct prediction → keep weights unchanged
- 🔴 **Aggressive**: Wrong prediction → aggressively update weights

| Property | Benefit |
|----------|---------|
| Online learning | Updates in real-time with new articles |
| Speed | Much faster than SVMs or neural networks |
| Sparse data | Works natively with TF-IDF sparse matrices |

**Advantages:** Fast convergence, high-dimensional text, low memory.
**Limitations:** No probability output, sensitive to noisy labels.
            """)

        with st.expander("🎓 Viva Q&A", expanded=False):
            st.markdown("""
**Q: Why not use BERT or Neural Networks?**
A: For this scope, traditional ML provides >90% accuracy with far lower cost. BERT requires GPUs and long training times — impractical for a project demo.

**Q: How is confidence % calculated?**
A: Logistic Regression's `predict_proba()` applies softmax to linear scores, giving probability per class. E.g. [0.82, 0.18] = 82% FAKE confidence.

**Q: What are the limitations?**
A: Adversarially written fake news that mimics credible style may fool it. Context and source credibility are not considered.

**Q: Future improvements?**
A: BERT embeddings, source credibility scoring, browser extension, active learning with user feedback.
            """)

    st.markdown("""<div class='footer'>
        🔍 Fake News Detection · B.Tech Final Year Project · Python + Scikit-learn + Streamlit
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
