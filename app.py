"""
Streamlit app for multi-label peer feedback classification.

Expected artifact structure inside ARTIFACT_DIR:
- best_ml_model.joblib
- best_ml_model_metadata.json
- best_dl_model.keras                 optional, needed for DL inference
- best_dl_tokenizer.joblib             optional, needed for DL inference
- best_dl_model_metadata.json          optional, needed for DL inference

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

# Keras/TensorFlow is imported lazily only when DL is used.
# DL models saved with Keras 3 must be loaded with standalone keras, not older tf.keras.


# -----------------------------------------------------------------------------
# Custom classes needed for joblib compatibility
# -----------------------------------------------------------------------------
def labelset_strings(y: np.ndarray) -> np.ndarray:
    """Convert a binary multi-label matrix to label powerset strings."""
    return np.array(["".join(map(str, row.astype(int))) for row in y])


class LabelPowersetClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible label powerset wrapper used by the training pipeline."""

    def __init__(self, base_estimator=None):
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else LogisticRegression(max_iter=2000, class_weight="balanced")
        )

    def fit(self, X, y):
        self.n_labels_ = y.shape[1]
        label_strings = labelset_strings(np.asarray(y))
        self.encoder_ = LabelEncoder().fit(label_strings)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, self.encoder_.transform(label_strings))
        return self

    def predict(self, X):
        pred_strings = self.encoder_.inverse_transform(self.estimator_.predict(X))
        return np.array([[int(ch) for ch in s] for s in pred_strings], dtype=int)


class SafeCalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    """Safe calibration wrapper used by the training pipeline."""

    def __init__(self, estimator=None, cv=3):
        self.estimator = estimator
        self.cv = cv

    def fit(self, X, y):
        y_arr = np.asarray(y)
        _, counts = np.unique(y_arr, return_counts=True)
        estimator = self.estimator if self.estimator is not None else LogisticRegression(max_iter=2000)
        can_calibrate = len(counts) >= 2 and int(counts.min()) >= int(self.cv)

        if can_calibrate:
            self.model_ = CalibratedClassifierCV(estimator=clone(estimator), cv=self.cv)
            self.calibration_mode_ = f"calibrated_cv_{self.cv}"
        else:
            self.model_ = clone(estimator)
            self.calibration_mode_ = "fallback_uncalibrated_due_to_rare_class"

        self.model_.fit(X, y_arr)
        self.classes_ = getattr(self.model_, "classes_", None)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)
        raise AttributeError("The fitted estimator does not provide predict_proba.")

    def decision_function(self, X):
        if hasattr(self.model_, "decision_function"):
            return self.model_.decision_function(X)
        raise AttributeError("The fitted estimator does not provide decision_function.")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_LABELS = ["Appreciation", "Problem", "Suggestion", "Neutral"]
DEFAULT_ARTIFACT_DIR = "artifacts/latest_run"

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "pada", "dalam",
    "adalah", "atau", "juga", "karena", "sebagai", "lebih", "agar", "akan", "sudah", "belum",
    "tidak", "bisa", "dapat", "sangat", "masih", "ada", "jadi", "tersebut", "nya", "the",
    "a", "an", "and", "or", "to", "of", "in", "is", "are", "for", "with", "on", "this",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def preprocess_text(text: object) -> str:
    """Simple text normalization consistent with the modelling pipeline."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ÿ_\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_json(path: Path) -> Dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def thresholds_from_metadata(metadata: Dict, labels: List[str]) -> np.ndarray:
    raw = metadata.get("thresholds", {})
    if isinstance(raw, dict):
        return np.array([float(raw.get(label, 0.5)) for label in labels], dtype=float)
    if isinstance(raw, list):
        arr = np.array(raw, dtype=float)
        if len(arr) == len(labels):
            return arr
    return np.array([0.5] * len(labels), dtype=float)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Map decision scores to [0,1] with sigmoid when needed."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if np.nanmin(scores) < 0 or np.nanmax(scores) > 1:
        scores = 1.0 / (1.0 + np.exp(-scores))
    return scores


def predict_scores_or_none(model, texts: pd.Series) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
        if isinstance(proba, list):
            # MultiOutputClassifier returns a list of arrays, one per label.
            return np.vstack([p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel() for p in proba]).T
        proba = np.asarray(proba)
        if proba.ndim == 3:
            return proba[:, :, 1]
        return proba
    if hasattr(model, "decision_function"):
        return normalize_scores(model.decision_function(texts))
    return None


@st.cache_resource(show_spinner=False)
def load_ml_artifact(artifact_dir: str) -> Tuple[object, Dict, List[str], np.ndarray]:
    artifact_path = Path(artifact_dir)
    model_bundle = joblib.load(artifact_path / "best_ml_model.joblib")
    metadata = read_json(artifact_path / "best_ml_model_metadata.json")

    if isinstance(model_bundle, dict):
        model = model_bundle.get("pipeline", model_bundle.get("model", model_bundle))
        labels = model_bundle.get("labels") or metadata.get("labels") or DEFAULT_LABELS
        thresholds = np.asarray(model_bundle.get("thresholds", thresholds_from_metadata(metadata, labels)), dtype=float)
    else:
        model = model_bundle
        labels = metadata.get("labels") or DEFAULT_LABELS
        thresholds = thresholds_from_metadata(metadata, labels)

    if len(thresholds) != len(labels):
        thresholds = thresholds_from_metadata(metadata, labels)

    return model, metadata, labels, thresholds


def load_keras_model_compat(model_path: Path):
    """Load a .keras model saved by either Keras 3 or tf.keras.

    The training notebook may save models with Keras 3, whose serialized config
    contains modules such as `keras.src.models.functional`. Such models cannot be
    deserialized by older `tf.keras` / `keras<3`. This loader therefore tries
    standalone Keras first, then falls back to TensorFlow Keras.
    """
    errors = []

    try:
        import keras  # Keras 3 standalone package

        return keras.saving.load_model(model_path, compile=False, safe_mode=False)
    except Exception as exc:  # pragma: no cover - shown to the Streamlit user
        errors.append(f"keras.saving.load_model failed: {exc}")

    try:
        from tensorflow import keras as tf_keras

        return tf_keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # pragma: no cover - shown to the Streamlit user
        errors.append(f"tf.keras.models.load_model failed: {exc}")

    raise RuntimeError(
        "Model DL tidak dapat dimuat. Kemungkinan besar versi Keras/TensorFlow "
        "di environment Streamlit tidak sama dengan versi saat model disimpan. "
        "Gunakan requirements terbaru: keras>=3 dan tensorflow>=2.16. Detail: "
        + " | ".join(errors)
    )


def pad_sequences_compat(sequences, max_len: int):
    """Pad token sequences with Keras 3 first, then tf.keras fallback."""
    try:
        import keras

        return keras.utils.pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    except Exception:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")


@st.cache_resource(show_spinner=False)
def load_dl_artifact(artifact_dir: str):
    artifact_path = Path(artifact_dir)
    model_path = artifact_path / "best_dl_model.keras"
    tokenizer_path = artifact_path / "best_dl_tokenizer.joblib"
    metadata_path = artifact_path / "best_dl_model_metadata.json"

    missing = [str(p.name) for p in [model_path, tokenizer_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Artefak DL tidak ditemukan: {', '.join(missing)} di {artifact_path}")

    model = load_keras_model_compat(model_path)
    tokenizer = joblib.load(tokenizer_path)
    metadata = read_json(metadata_path)
    labels = metadata.get("labels") or DEFAULT_LABELS
    thresholds = thresholds_from_metadata(metadata, labels)
    max_len = int(metadata.get("dl_max_len", metadata.get("max_len", 200)))
    return model, tokenizer, metadata, labels, thresholds, max_len


def predict_ml(texts_clean: pd.Series, artifact_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    model, metadata, labels, thresholds = load_ml_artifact(artifact_dir)
    scores = predict_scores_or_none(model, texts_clean)
    if scores is not None:
        scores = normalize_scores(scores)
        y_pred = (scores >= thresholds).astype(int)
    else:
        y_pred = np.asarray(model.predict(texts_clean)).astype(int)
        scores = y_pred.astype(float)
    return y_pred, scores, labels, metadata


def predict_dl(texts_clean: pd.Series, artifact_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    model, tokenizer, metadata, labels, thresholds, max_len = load_dl_artifact(artifact_dir)
    seq = tokenizer.texts_to_sequences(texts_clean.tolist())
    padded = pad_sequences_compat(seq, max_len=max_len)
    scores = np.asarray(model.predict(padded, verbose=0), dtype=float)
    y_pred = (scores >= thresholds).astype(int)
    return y_pred, scores, labels, metadata


def add_prediction_columns(df: pd.DataFrame, y_pred: np.ndarray, scores: np.ndarray, labels: List[str]) -> pd.DataFrame:
    out = df.copy()
    for i, label in enumerate(labels):
        out[f"pred_{label}"] = y_pred[:, i].astype(int)
        out[f"score_{label}"] = np.round(scores[:, i].astype(float), 4)
    out["predicted_labels"] = [", ".join([labels[j] for j, val in enumerate(row) if val == 1]) or "None" for row in y_pred]
    out["n_predicted_labels"] = y_pred.sum(axis=1).astype(int)
    return out


def build_label_summary(y_pred: np.ndarray, labels: List[str]) -> pd.DataFrame:
    n = len(y_pred)
    rows = []
    for i, label in enumerate(labels):
        count = int(y_pred[:, i].sum())
        rows.append({"label": label, "count": count, "percentage": round(100 * count / max(n, 1), 2)})
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def build_combination_summary(y_pred: np.ndarray, labels: List[str]) -> pd.DataFrame:
    combos = []
    for row in y_pred:
        active = [labels[i] for i, v in enumerate(row) if int(v) == 1]
        combos.append(" + ".join(active) if active else "None")
    summary = pd.Series(combos).value_counts().reset_index()
    summary.columns = ["label_combination", "count"]
    summary["percentage"] = (100 * summary["count"] / max(len(y_pred), 1)).round(2)
    return summary


def build_pair_summary(y_pred: np.ndarray, labels: List[str]) -> pd.DataFrame:
    rows = []
    for i, first in enumerate(labels):
        for j, second in enumerate(labels):
            if i < j:
                count = int(np.logical_and(y_pred[:, i] == 1, y_pred[:, j] == 1).sum())
                rows.append({"pair": f"{first} + {second}", "count": count})
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def generate_wordcloud(texts: List[str]) -> Optional[Image.Image]:
    text = " ".join([str(t) for t in texts if str(t).strip()])
    if not text.strip():
        return None
    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=STOPWORDS_ID,
        collocations=False,
        max_words=120,
    ).generate(text)
    return wc.to_image()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def generate_learning_insight_text(
    n_rows: int,
    label_summary: pd.DataFrame,
    combo_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> str:
    top_label = label_summary.iloc[0]
    top_combo = combo_summary.iloc[0]
    problem_count = int(label_summary.loc[label_summary["label"] == "Problem", "count"].sum())
    suggestion_count = int(label_summary.loc[label_summary["label"] == "Suggestion", "count"].sum())
    appreciation_count = int(label_summary.loc[label_summary["label"] == "Appreciation", "count"].sum())
    neutral_count = int(label_summary.loc[label_summary["label"] == "Neutral", "count"].sum())

    # Specific pedagogical patterns
    problem_suggestion = int(pair_summary.loc[pair_summary["pair"] == "Problem + Suggestion", "count"].sum())
    appreciation_problem = int(pair_summary.loc[pair_summary["pair"] == "Appreciation + Problem", "count"].sum())

    lines = [
        f"Sebanyak {n_rows} peer feedback berhasil diklasifikasikan.",
        f"Kategori yang paling dominan adalah {top_label['label']} ({int(top_label['count'])} komentar; {top_label['percentage']}%).",
        f"Kombinasi label paling sering muncul adalah {top_combo['label_combination']} ({int(top_combo['count'])} komentar; {top_combo['percentage']}%).",
        "",
        "Interpretasi pembelajaran:",
    ]

    if suggestion_count > 0:
        lines.append(
            f"- Terdapat {suggestion_count} komentar yang mengandung Suggestion, menunjukkan adanya umpan balik yang dapat ditindaklanjuti untuk perbaikan karya."
        )
    if problem_count > 0:
        lines.append(
            f"- Terdapat {problem_count} komentar yang mengandung Problem, menunjukkan bahwa mahasiswa mampu mengidentifikasi kelemahan atau aspek yang perlu diperbaiki."
        )
    if appreciation_count > 0:
        lines.append(
            f"- Terdapat {appreciation_count} komentar yang mengandung Appreciation, mencerminkan dukungan positif dalam proses peer review."
        )
    if neutral_count > 0:
        lines.append(
            f"- Terdapat {neutral_count} komentar Neutral; proporsi ini dapat menjadi indikator komentar yang kurang evaluatif atau kurang memberikan arahan belajar."
        )
    if problem_suggestion > 0:
        lines.append(
            f"- Kombinasi Problem + Suggestion muncul pada {problem_suggestion} komentar, yang dapat dianggap sebagai feedback konstruktif karena mengidentifikasi masalah sekaligus menawarkan perbaikan."
        )
    if appreciation_problem > 0:
        lines.append(
            f"- Kombinasi Appreciation + Problem muncul pada {appreciation_problem} komentar, menunjukkan pola feedback yang menyeimbangkan penguatan positif dan kritik."
        )

    lines.append("")
    lines.append(
        "Secara umum, hasil ini dapat digunakan dosen untuk memantau kualitas peer review, mengidentifikasi kebutuhan scaffolding, dan mengevaluasi apakah aktivitas peer review mendorong komentar yang konstruktif."
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Peer Feedback Multi-Label Classifier",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Peer Feedback Multi-Label Classifier")
st.caption("Klasifikasi komentar peer review menjadi Appreciation, Problem, Suggestion, dan Neutral, lalu mengubahnya menjadi learning insights.")

with st.sidebar:
    st.header("Pengaturan Model")
    artifact_dir = st.text_input("Folder artefak model", value=DEFAULT_ARTIFACT_DIR)
    model_choice = st.radio("Pilih model terbaik", options=["Machine Learning", "Deep Learning"], horizontal=False)

    st.divider()
    st.header("Pengaturan Data")
    uploaded_file = st.file_uploader("Upload CSV peer feedback", type=["csv"])
    delimiter = st.selectbox("Delimiter CSV", options=[",", ";", "\t"], index=0)
    encoding = st.selectbox("Encoding", options=["utf-8", "utf-8-sig", "latin-1"], index=0)

    st.divider()
    st.write("**Catatan artefak**")
    st.code(
        "best_ml_model.joblib\n"
        "best_ml_model_metadata.json\n"
        "best_dl_model.keras\n"
        "best_dl_tokenizer.joblib\n"
        "best_dl_model_metadata.json",
        language="text",
    )

if uploaded_file is None:
    st.info("Upload file CSV yang berisi kolom teks peer feedback untuk mulai melakukan klasifikasi.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, sep=delimiter, encoding=encoding)
except Exception as exc:
    st.error(f"CSV tidak dapat dibaca: {exc}")
    st.stop()

if df.empty:
    st.error("File CSV kosong.")
    st.stop()

st.subheader("1. Preview Data")
st.dataframe(df.head(20), use_container_width=True)

candidate_text_cols = [c for c in df.columns if df[c].dtype == "object"] or list(df.columns)
default_text_index = 0
for preferred in ["text", "feedback", "comment", "peer_feedback", "peer_comment", "comments"]:
    if preferred in df.columns:
        default_text_index = list(df.columns).index(preferred)
        break

text_col = st.selectbox(
    "Pilih kolom teks peer feedback",
    options=list(df.columns),
    index=default_text_index if default_text_index < len(df.columns) else 0,
)

if st.button("🚀 Jalankan Klasifikasi", type="primary"):
    artifact_path = Path(artifact_dir)
    if not artifact_path.exists():
        st.error(f"Folder artefak tidak ditemukan: {artifact_path}")
        st.stop()

    work_df = df.copy()
    work_df["text_clean"] = work_df[text_col].apply(preprocess_text)

    with st.spinner("Memuat model dan melakukan klasifikasi..."):
        try:
            if model_choice == "Machine Learning":
                y_pred, scores, labels, metadata = predict_ml(work_df["text_clean"], artifact_dir)
            else:
                y_pred, scores, labels, metadata = predict_dl(work_df["text_clean"], artifact_dir)
        except Exception as exc:
            st.error(f"Gagal memuat model atau melakukan prediksi: {exc}")
            st.stop()

    result_df = add_prediction_columns(work_df, y_pred, scores, labels)
    label_summary = build_label_summary(y_pred, labels)
    combo_summary = build_combination_summary(y_pred, labels)
    pair_summary = build_pair_summary(y_pred, labels)
    insight_text = generate_learning_insight_text(len(result_df), label_summary, combo_summary, pair_summary)

    st.success(f"Klasifikasi selesai menggunakan model: {model_choice}")

    st.subheader("2. Ringkasan Learning Insight")
    metric_cols = st.columns(len(labels))
    for idx, label in enumerate(labels):
        count = int(label_summary.loc[label_summary["label"] == label, "count"].iloc[0])
        pct = float(label_summary.loc[label_summary["label"] == label, "percentage"].iloc[0])
        metric_cols[idx].metric(label, f"{count}", f"{pct}%")

    st.text_area("Summary insight otomatis", value=insight_text, height=260)

    st.subheader("3. Distribusi Label")
    left, right = st.columns([1, 1])
    with left:
        st.write("**Jumlah komentar per label**")
        st.dataframe(label_summary, use_container_width=True)
        st.bar_chart(label_summary.set_index("label")["count"])
    with right:
        st.write("**Kombinasi label**")
        st.dataframe(combo_summary, use_container_width=True)
        st.bar_chart(combo_summary.set_index("label_combination")["count"])

    st.subheader("4. Co-occurrence Antar Label")
    st.write("Jumlah komentar yang mendapatkan dua label sekaligus, misalnya Problem + Appreciation atau Problem + Suggestion.")
    st.dataframe(pair_summary, use_container_width=True)
    st.bar_chart(pair_summary.set_index("pair")["count"])

    st.subheader("5. WordCloud per Label")
    wc_cols = st.columns(2)
    for i, label in enumerate(labels):
        with wc_cols[i % 2]:
            st.write(f"**{label}**")
            label_texts = result_df.loc[result_df[f"pred_{label}"] == 1, "text_clean"].tolist()
            image = generate_wordcloud(label_texts)
            if image is None:
                st.info(f"Tidak ada teks yang diklasifikasikan sebagai {label}.")
            else:
                st.image(image, use_container_width=True)

    st.subheader("6. Hasil Klasifikasi")
    st.dataframe(result_df, use_container_width=True)

    st.download_button(
        label="⬇️ Download hasil klasifikasi CSV",
        data=dataframe_to_csv_bytes(result_df),
        file_name="peer_feedback_classification_results.csv",
        mime="text/csv",
    )

    st.download_button(
        label="⬇️ Download ringkasan label CSV",
        data=dataframe_to_csv_bytes(label_summary),
        file_name="peer_feedback_label_summary.csv",
        mime="text/csv",
    )

    st.download_button(
        label="⬇️ Download kombinasi label CSV",
        data=dataframe_to_csv_bytes(combo_summary),
        file_name="peer_feedback_label_combination_summary.csv",
        mime="text/csv",
    )

    st.subheader("7. Metadata Model")
    st.json(metadata)
