# ============================================================
# train_model.py
# Fake News Detection - Model Training Script
# B.Tech Major Project | Final Year
# ============================================================
# This script:
#   1. Loads the news dataset
#   2. Preprocesses text data
#   3. Trains TF-IDF Vectorizer
#   4. Trains Passive Aggressive Classifier (main model)
#   5. Trains Logistic Regression (comparison model)
#   6. Evaluates both models and prints reports
#   7. Saves all trained objects using pickle
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# --- Scikit-learn imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
# PassiveAggressiveClassifier is deprecated in sklearn >= 1.8.
# We use SGDClassifier with equivalent PAC parameters for forward compatibility,
# while falling back to the original class for older sklearn versions.
try:
    import sklearn
    from packaging import version as pkg_version
    _sklearn_new = pkg_version.parse(sklearn.__version__) >= pkg_version.parse("1.8.0")
except Exception:
    _sklearn_new = False

if _sklearn_new:
    from sklearn.linear_model import SGDClassifier as _SGD
    def PassiveAggressiveClassifier(C=1.0, max_iter=1000, random_state=42, tol=1e-4):
        """
        Drop-in replacement for PassiveAggressiveClassifier using SGDClassifier
        with mathematically equivalent parameters (sklearn >= 1.8 compatible).
        """
        return _SGD(
            loss='hinge',
            penalty=None,
            learning_rate='pa1',
            eta0=C,
            max_iter=max_iter,
            random_state=random_state,
            tol=tol
        )
else:
    from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# --- Text preprocessing ---
import re
import string

# --- Visualization (for saving training plots) ---
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("  FAKE NEWS DETECTION — MODEL TRAINING")
print("=" * 60)


# ============================================================
# STEP 1: Load Dataset
# ============================================================
print("\n[STEP 1] Loading dataset...")

DATASET_PATH = os.path.join("dataset", "news.csv")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATASET_PATH}'.\n"
        "Make sure 'news.csv' exists inside the 'dataset/' folder."
    )

df = pd.read_csv(DATASET_PATH)
print(f"  ✔  Loaded {len(df)} rows and {len(df.columns)} columns.")
print(f"  Columns: {list(df.columns)}")
print(f"\n  Label distribution:\n{df['label'].value_counts().to_string()}")


# ============================================================
# STEP 2: Data Cleaning & Preprocessing
# ============================================================
print("\n[STEP 2] Preprocessing text data...")

def clean_text(text):
    """
    Cleans a raw news article text:
    - Lowercases everything (case doesn't matter for meaning)
    - Removes URLs (http/https links)
    - Removes HTML tags
    - Removes punctuation and special characters
    - Removes extra whitespace
    Returns cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs (e.g., https://example.com)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags (e.g., <b>, </div>)
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits (numbers rarely help classify news authenticity)
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace / newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply cleaning to the 'text' column
df['clean_text'] = df['text'].apply(clean_text)

# Drop rows where text became empty after cleaning
df = df[df['clean_text'].str.strip() != '']
df.reset_index(drop=True, inplace=True)

print(f"  ✔  Cleaned. {len(df)} usable rows remaining.")


# ============================================================
# STEP 3: Encode Labels
# ============================================================
print("\n[STEP 3] Encoding labels...")

# REAL → 1, FAKE → 0  (standard binary classification)
df['label_encoded'] = df['label'].map({'REAL': 1, 'FAKE': 0})

# Check for unmapped values
if df['label_encoded'].isna().any():
    unmapped = df[df['label_encoded'].isna()]['label'].unique()
    raise ValueError(
        f"Unknown label values found: {unmapped}. "
        "Labels must be exactly 'REAL' or 'FAKE' (case-sensitive)."
    )

X = df['clean_text']          # Features (text)
y = df['label_encoded']       # Target (0 or 1)

print(f"  ✔  Labels encoded: FAKE=0, REAL=1")


# ============================================================
# STEP 4: Train / Test Split
# ============================================================
print("\n[STEP 4] Splitting data into train and test sets...")

# 80% training, 20% testing
# random_state=42 ensures reproducible splits every time
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y           # Ensures equal class ratio in both splits
)

print(f"  ✔  Training samples : {len(X_train)}")
print(f"  ✔  Testing  samples : {len(X_test)}")


# ============================================================
# STEP 5: TF-IDF Vectorization
# ============================================================
# WHY TF-IDF?
# - TF (Term Frequency): How often a word appears in a document.
# - IDF (Inverse Document Frequency): Penalises words that appear
#   in most documents (e.g., "the", "is") and rewards rare, 
#   informative words.
# - Together, TF-IDF gives each word a score that reflects how
#   important it is for a specific article.
# - It converts raw text into numerical vectors that ML models
#   can understand.
# ============================================================
print("\n[STEP 5] Fitting TF-IDF Vectorizer...")

tfidf = TfidfVectorizer(
    max_df=0.7,           # Ignore words appearing in >70% of docs (too common)
    min_df=2,             # Ignore words appearing in fewer than 2 docs (too rare)
    max_features=10000,   # Keep only the top 10,000 features (words) by frequency
    ngram_range=(1, 2),   # Use single words AND two-word phrases (bigrams)
    stop_words='english', # Remove common English stop words (the, is, are...)
    sublinear_tf=True     # Apply log normalization to TF values
)

# Fit on training data ONLY — never fit on test data (data leakage!)
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform (not fit) test data using the same vocabulary
X_test_tfidf  = tfidf.transform(X_test)

print(f"  ✔  Vocabulary size   : {len(tfidf.vocabulary_)} unique terms")
print(f"  ✔  Train matrix shape: {X_train_tfidf.shape}")
print(f"  ✔  Test  matrix shape: {X_test_tfidf.shape}")


# ============================================================
# STEP 6A: Train Passive Aggressive Classifier (Main Model)
# ============================================================
# WHY PASSIVE AGGRESSIVE CLASSIFIER?
# - Designed for online / streaming learning — great for news feeds
# - "Passive"  : If a prediction is correct, the model doesn't change.
# - "Aggressive": If a prediction is wrong, the model aggressively
#                 updates its parameters to correct the mistake.
# - Very fast and memory efficient — ideal for large text datasets.
# - Works natively with sparse TF-IDF matrices.
# - Proven to perform well on text classification tasks.
# ============================================================
print("\n[STEP 6A] Training Passive Aggressive Classifier...")

pac_model = PassiveAggressiveClassifier(
    C=0.5,                # Regularization parameter (lower = more conservative)
    max_iter=1000,        # Maximum number of training iterations
    random_state=42,      # Reproducibility
    tol=1e-4              # Stop training when improvement is below this threshold
)

pac_model.fit(X_train_tfidf, y_train)

# Evaluate on test set
pac_predictions = pac_model.predict(X_test_tfidf)
pac_accuracy     = accuracy_score(y_test, pac_predictions)
pac_conf_matrix  = confusion_matrix(y_test, pac_predictions)
pac_report       = classification_report(
    y_test, pac_predictions,
    target_names=['FAKE', 'REAL'],
    output_dict=True
)

print(f"  ✔  PAC Accuracy : {pac_accuracy * 100:.2f}%")
print(f"\n  Classification Report (PAC):\n")
print(classification_report(y_test, pac_predictions, target_names=['FAKE', 'REAL']))


# ============================================================
# STEP 6B: Train Logistic Regression (Comparison Model)
# ============================================================
# Logistic Regression is a classic baseline model for NLP.
# It provides probability scores (confidence %) for each class.
# We use it here to compare against PAC.
# ============================================================
print("\n[STEP 6B] Training Logistic Regression (comparison)...")

lr_model = LogisticRegression(
    C=1.0,           # Regularization strength
    max_iter=1000,   # Maximum iterations for solver convergence
    random_state=42,
    solver='lbfgs'   # Limited-memory Broyden-Fletcher optimizer
)

lr_model.fit(X_train_tfidf, y_train)

lr_predictions = lr_model.predict(X_test_tfidf)
lr_accuracy     = accuracy_score(y_test, lr_predictions)
lr_conf_matrix  = confusion_matrix(y_test, lr_predictions)
lr_report       = classification_report(
    y_test, lr_predictions,
    target_names=['FAKE', 'REAL'],
    output_dict=True
)

print(f"  ✔  LR  Accuracy : {lr_accuracy * 100:.2f}%")
print(f"\n  Classification Report (LR):\n")
print(classification_report(y_test, lr_predictions, target_names=['FAKE', 'REAL']))


# ============================================================
# STEP 7: Save Accuracy Comparison Plot
# ============================================================
print("\n[STEP 7] Saving accuracy comparison plot...")

os.makedirs("models", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#0E1117')

# --- Subplot 1: Accuracy Comparison Bar Chart ---
ax1 = axes[0]
ax1.set_facecolor('#1A1A2E')
models_names = ['Passive Aggressive\nClassifier', 'Logistic\nRegression']
accuracies   = [pac_accuracy * 100, lr_accuracy * 100]
colors       = ['#00D4FF', '#FF6B6B']
bars = ax1.bar(models_names, accuracies, color=colors, width=0.5, edgecolor='white', linewidth=0.8)
ax1.set_ylim(0, 110)
ax1.set_ylabel("Accuracy (%)", color='white', fontsize=12)
ax1.set_title("Model Accuracy Comparison", color='white', fontsize=14, fontweight='bold', pad=15)
ax1.tick_params(colors='white')
ax1.spines['bottom'].set_color('#444')
ax1.spines['left'].set_color('#444')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for bar, acc in zip(bars, accuracies):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{acc:.1f}%",
        ha='center', va='bottom', color='white', fontsize=13, fontweight='bold'
    )

# --- Subplot 2: Confusion Matrix for PAC ---
ax2 = axes[1]
ax2.set_facecolor('#1A1A2E')
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=pac_conf_matrix,
    display_labels=['FAKE', 'REAL']
)
cm_display.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title("Confusion Matrix (PAC Model)", color='white', fontsize=14, fontweight='bold', pad=15)
ax2.tick_params(colors='white')
ax2.xaxis.label.set_color('white')
ax2.yaxis.label.set_color('white')
for text in ax2.texts:
    text.set_color('white')

plt.tight_layout(pad=3.0)
plot_path = os.path.join("models", "training_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0E1117')
plt.close()
print(f"  ✔  Plot saved → {plot_path}")


# ============================================================
# STEP 8: Save Models and Metadata with Pickle
# ============================================================
# pickle serializes Python objects to binary files so they can be
# loaded later without retraining — saves time during deployment.
# ============================================================
print("\n[STEP 8] Saving models to disk using pickle...")

def save_pickle(obj, filename):
    """Save a Python object to a .pkl file inside the models/ directory."""
    path = os.path.join("models", filename)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✔  Saved '{filename}'  ({size_kb:.1f} KB)")

# Save TF-IDF Vectorizer (must be saved — needed to transform new text)
save_pickle(tfidf, "tfidf_vectorizer.pkl")

# Save Passive Aggressive Classifier (main model)
save_pickle(pac_model, "pac_model.pkl")

# Save Logistic Regression (comparison model)
save_pickle(lr_model, "lr_model.pkl")

# Save metadata dictionary (accuracy, confusion matrix, reports)
metadata = {
    "pac_accuracy"    : round(pac_accuracy * 100, 2),
    "lr_accuracy"     : round(lr_accuracy  * 100, 2),
    "pac_conf_matrix" : pac_conf_matrix.tolist(),
    "lr_conf_matrix"  : lr_conf_matrix.tolist(),
    "pac_report"      : pac_report,
    "lr_report"       : lr_report,
    "vocabulary_size" : len(tfidf.vocabulary_),
    "training_samples": len(X_train),
    "testing_samples" : len(X_test),
    "total_samples"   : len(df),
}
save_pickle(metadata, "metadata.pkl")


# ============================================================
# TRAINING COMPLETE — Summary
# ============================================================
print("\n" + "=" * 60)
print("  TRAINING COMPLETE ✔")
print("=" * 60)
print(f"  Passive Aggressive Classifier Accuracy : {pac_accuracy * 100:.2f}%")
print(f"  Logistic Regression Accuracy           : {lr_accuracy  * 100:.2f}%")
print(f"  Models saved in : models/")
print(f"  Plot  saved in  : models/training_results.png")
print("=" * 60)
print("\n  Run the app with:  streamlit run app.py\n")
