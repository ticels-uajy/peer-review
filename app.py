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
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# TensorFlow is imported lazily only when DL is used.


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


@st.cache_resource(show_spinner=False)
def load_dl_artifact(artifact_dir: str):
    from tensorflow.keras.models import load_model

    artifact_path = Path(artifact_dir)
    model = load_model(artifact_path / "best_dl_model.keras")
    tokenizer = joblib.load(artifact_path / "best_dl_tokenizer.joblib")
    metadata = read_json(artifact_path / "best_dl_model_metadata.json")
    labels = metadata.get("labels") or DEFAULT_LABELS
    thresholds = thresholds_from_metadata(metadata, labels)
    max_len = int(metadata.get("dl_max_len", metadata.get("max_len", 120)))
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
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer, metadata, labels, thresholds, max_len = load_dl_artifact(artifact_dir)
    seq = tokenizer.texts_to_sequences(texts_clean.tolist())
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    scores = model.predict(padded, verbose=0)
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
    n = len(y_pred)
    for i, first in enumerate(labels):
        for j, second in enumerate(labels):
            if i < j:
                count = int(np.logical_and(y_pred[:, i] == 1, y_pred[:, j] == 1).sum())
                rows.append({
                    "pair": f"{first} + {second}",
                    "count": count,
                    "percentage": round(100 * count / max(n, 1), 2),
                })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def pct(value: float) -> str:
    return f"{float(value):.2f}%"


def extract_keyphrases(texts: List[str], top_n: int = 8) -> List[str]:
    """Extract readable unigram/bigram keyphrases using TF-IDF."""
    cleaned = [str(t).strip() for t in texts if str(t).strip()]
    if not cleaned:
        return []
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words=list(STOPWORDS_ID),
            token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_\-]{2,}\b",
            max_features=4000,
        )
        X = vectorizer.fit_transform(cleaned)
        scores = np.asarray(X.sum(axis=0)).ravel()
        terms = np.asarray(vectorizer.get_feature_names_out())
        order = scores.argsort()[::-1]
        phrases = []
        seen = set()
        for idx in order:
            term = str(terms[idx]).strip()
            # Prefer clearer terms and avoid near-duplicate unigram/bigram repetitions.
            if not term or term in seen:
                continue
            if len(term) < 3:
                continue
            phrases.append(term)
            seen.add(term)
            if len(phrases) >= top_n:
                break
        return phrases
    except Exception:
        # Safe fallback if the texts are too short or vectorization fails.
        tokens = []
        for text in cleaned:
            tokens.extend([tok for tok in re.findall(r"[a-zA-ZÀ-ÿ]{3,}", text.lower()) if tok not in STOPWORDS_ID])
        if not tokens:
            return []
        return pd.Series(tokens).value_counts().head(top_n).index.tolist()


def build_label_topic_summary(result_df: pd.DataFrame, labels: List[str]) -> Dict[str, Dict]:
    """Create compact per-label topic summaries for the learning insight text."""
    n = len(result_df)
    summary = {}
    for label in labels:
        mask = result_df[f"pred_{label}"] == 1
        label_texts = result_df.loc[mask, "text_clean"].tolist()
        count = int(mask.sum())
        summary[label] = {
            "count": count,
            "percentage": round(100 * count / max(n, 1), 2),
            "keyphrases": extract_keyphrases(label_texts, top_n=6),
        }
    return summary


def label_meaning(label: str) -> str:
    meanings = {
        "Appreciation": "menggambarkan aspek yang dinilai positif, jelas, menarik, atau sudah baik oleh mahasiswa",
        "Problem": "menunjukkan aspek yang dianggap kurang, membingungkan, bermasalah, atau perlu diperbaiki",
        "Suggestion": "menunjukkan rekomendasi perbaikan, tindakan lanjutan, atau masukan konkret",
        "Neutral": "menunjukkan komentar yang cenderung deskriptif, umum, atau belum cukup evaluatif",
    }
    return meanings.get(label, "menggambarkan pola komentar yang terdeteksi pada label ini")


def label_instructional_action(label: str) -> str:
    actions = {
        "Appreciation": "Gunakan aspek ini sebagai contoh praktik baik atau kekuatan karya yang perlu dipertahankan.",
        "Problem": "Gunakan aspek ini untuk mengidentifikasi bagian materi/karya yang perlu klarifikasi, revisi, atau pendampingan.",
        "Suggestion": "Gunakan masukan ini sebagai dasar tindak lanjut karena biasanya berisi arahan perbaikan yang dapat diterapkan.",
        "Neutral": "Berikan rubrik, contoh komentar, atau prompt tambahan agar mahasiswa menulis feedback yang lebih spesifik dan bermanfaat.",
    }
    return actions.get(label, "Gunakan temuan ini sebagai bahan refleksi pembelajaran.")


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
    label_topics: Optional[Dict[str, Dict]] = None,
) -> str:
    """Generate a readable, teacher-oriented learning insight summary in Markdown."""
    label_topics = label_topics or {}
    top_label = label_summary.iloc[0]
    top_combo = combo_summary.iloc[0]

    def label_count(label: str) -> int:
        value = label_summary.loc[label_summary["label"] == label, "count"]
        return int(value.iloc[0]) if len(value) else 0

    def label_pct(label: str) -> float:
        value = label_summary.loc[label_summary["label"] == label, "percentage"]
        return float(value.iloc[0]) if len(value) else 0.0

    def pair_count(pair: str) -> int:
        value = pair_summary.loc[pair_summary["pair"] == pair, "count"]
        return int(value.iloc[0]) if len(value) else 0

    def pair_pct(pair: str) -> float:
        value = pair_summary.loc[pair_summary["pair"] == pair, "percentage"]
        return float(value.iloc[0]) if len(value) else 0.0

    appreciation_count = label_count("Appreciation")
    problem_count = label_count("Problem")
    suggestion_count = label_count("Suggestion")
    neutral_count = label_count("Neutral")
    problem_suggestion = pair_count("Problem + Suggestion")
    appreciation_problem = pair_count("Appreciation + Problem")

    suggestion_gap = max(problem_count - problem_suggestion, 0)

    lines = []
    lines.append("### Ringkasan Eksekutif")
    lines.append(
        f"Sebanyak **{n_rows} peer feedback** berhasil diklasifikasikan. "
        f"Kategori paling dominan adalah **{top_label['label']}** "
        f"(**{int(top_label['count'])} komentar; {pct(top_label['percentage'])}**). "
        f"Kombinasi label yang paling sering muncul adalah **{top_combo['label_combination']}** "
        f"(**{int(top_combo['count'])} komentar; {pct(top_combo['percentage'])}**)."
    )
    lines.append(
        "Karena klasifikasi bersifat **multi-label**, satu komentar dapat masuk ke lebih dari satu kategori. "
        "Oleh karena itu, total persentase antarlabel dapat melebihi 100%."
    )

    lines.append("\n### Profil Feedback per Label")
    for label in ["Appreciation", "Problem", "Suggestion", "Neutral"]:
        if label not in label_summary["label"].tolist():
            continue
        topics = label_topics.get(label, {}).get("keyphrases", [])
        topics_text = ", ".join(topics[:6]) if topics else "belum ada topik dominan yang cukup kuat"
        count = label_count(label)
        percentage = label_pct(label)
        lines.append(
            f"**{label}** — **{count} komentar ({pct(percentage)})**. "
            f"Tema/kata kunci utama: *{topics_text}*. "
            f"Label ini {label_meaning(label)}. {label_instructional_action(label)}"
        )

    lines.append("\n### Pola Kombinasi Label yang Perlu Diperhatikan")
    if problem_suggestion > 0:
        lines.append(
            f"- **Problem + Suggestion** muncul pada **{problem_suggestion} komentar ({pct(pair_pct('Problem + Suggestion'))})**. "
            "Ini merupakan indikator feedback konstruktif karena mahasiswa tidak hanya menemukan masalah, tetapi juga memberikan arah perbaikan."
        )
    else:
        lines.append(
            "- **Problem + Suggestion** belum muncul secara kuat. Ini menunjukkan bahwa kritik mahasiswa belum banyak disertai rekomendasi perbaikan yang eksplisit."
        )

    if appreciation_problem > 0:
        lines.append(
            f"- **Appreciation + Problem** muncul pada **{appreciation_problem} komentar ({pct(pair_pct('Appreciation + Problem'))})**. "
            "Pola ini menunjukkan feedback yang relatif seimbang karena mahasiswa mengakui aspek positif sekaligus menunjukkan bagian yang perlu diperbaiki."
        )

    if problem_count > 0 and suggestion_gap > 0:
        lines.append(
            f"- Terdapat sekitar **{suggestion_gap} komentar Problem** yang tidak terhubung dengan Suggestion. "
            "Pengajar dapat mendorong mahasiswa agar setiap kritik dilengkapi dengan saran yang spesifik dan dapat ditindaklanjuti."
        )

    if suggestion_count < max(1, int(0.1 * n_rows)):
        lines.append(
            f"- Proporsi **Suggestion** masih rendah (**{suggestion_count} komentar; {pct(label_pct('Suggestion'))}**). "
            "Ini mengindikasikan perlunya scaffolding, misalnya template kalimat: 'Bagian yang dapat diperbaiki adalah ... karena ... saran saya ...'."
        )

    if neutral_count / max(n_rows, 1) >= 0.20:
        lines.append(
            f"- **Neutral** cukup tinggi (**{neutral_count} komentar; {pct(label_pct('Neutral'))}**). "
            "Pengajar dapat memberi contoh komentar yang evaluatif agar mahasiswa tidak hanya menulis komentar umum atau deskriptif."
        )

    lines.append("\n### Implikasi untuk Pengajar")
    lines.append(
        "Hasil ini dapat digunakan untuk memantau kualitas peer review pada level kelas. "
        "Jika Appreciation dominan, aktivitas peer review sudah menunjukkan dukungan sosial, tetapi pengajar tetap perlu mendorong komentar yang lebih analitis. "
        "Jika Problem muncul tanpa Suggestion, mahasiswa perlu diarahkan untuk memberikan solusi. "
        "Jika Neutral tinggi, rubrik dan contoh feedback perlu diperjelas agar komentar lebih spesifik, evaluatif, dan bermanfaat bagi perbaikan karya."
    )
    return "\n\n".join(lines)


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
    label_topics = build_label_topic_summary(result_df, labels)
    insight_text = generate_learning_insight_text(
        len(result_df), label_summary, combo_summary, pair_summary, label_topics
    )

    st.success(f"Klasifikasi selesai menggunakan model: {model_choice}")

    st.subheader("2. Ringkasan Learning Insight")
    metric_cols = st.columns(len(labels))
    for idx, label in enumerate(labels):
        count = int(label_summary.loc[label_summary["label"] == label, "count"].iloc[0])
        pct = float(label_summary.loc[label_summary["label"] == label, "percentage"].iloc[0])
        metric_cols[idx].metric(label, f"{count}", f"{pct}%")

    st.markdown(insight_text)
    with st.expander("Lihat/copy versi Markdown ringkasan"):
        st.text_area("Summary insight otomatis", value=insight_text, height=420)

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
