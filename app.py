"""
Streamlit app for multi-label peer feedback classification and learning insight generation.

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
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

# TensorFlow / Keras / Transformers are imported lazily only when needed.

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
    "saya", "kami", "kita", "mereka", "mahasiswa", "komentar", "feedback", "peer", "review",
    "menurut", "menjadi", "sehingga", "secara", "namun", "tetapi", "cukup", "perlu",
}

LABEL_PEDAGOGICAL_MEANING = {
    "Appreciation": "menunjukkan aspek yang dinilai positif, kuat, jelas, menarik, atau sudah baik oleh mahasiswa.",
    "Problem": "menunjukkan aspek yang dianggap masih bermasalah, kurang jelas, kurang lengkap, keliru, atau perlu diperbaiki.",
    "Suggestion": "menunjukkan rekomendasi, arahan perbaikan, atau masukan yang dapat ditindaklanjuti.",
    "Neutral": "menunjukkan komentar yang cenderung deskriptif, umum, atau belum memberikan evaluasi dan arahan perbaikan yang kuat.",
}

LABEL_TEACHER_ACTION = {
    "Appreciation": "Gunakan informasi ini untuk mengidentifikasi aspek karya yang sudah dipahami atau diapresiasi mahasiswa, lalu dorong mereka menjelaskan alasan apresiasinya secara lebih spesifik.",
    "Problem": "Gunakan informasi ini untuk melihat bagian yang paling sering dipermasalahkan dan jadikan sebagai dasar klarifikasi, remediasi, atau diskusi kelas.",
    "Suggestion": "Gunakan informasi ini untuk menilai apakah mahasiswa sudah mampu memberi masukan yang konkret, operasional, dan dapat ditindaklanjuti.",
    "Neutral": "Gunakan informasi ini sebagai sinyal perlunya rubrik, contoh komentar, atau scaffolding agar mahasiswa menghasilkan feedback yang lebih evaluatif.",
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
    artifact_path = Path(artifact_dir)
    model_path = artifact_path / "best_dl_model.keras"

    try:
        import keras
        model = keras.saving.load_model(model_path, compile=False, safe_mode=False)
    except Exception:
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)

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
    try:
        from keras.utils import pad_sequences
    except Exception:
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
        collocations=True,
        max_words=140,
    ).generate(text)
    return wc.to_image()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def make_vectorizer(max_features: int = 1200, ngram_range: Tuple[int, int] = (1, 3)) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words=list(STOPWORDS_ID),
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_\-]{2,}\b",
        sublinear_tf=True,
    )


def extract_keyphrases(texts: List[str], top_n: int = 12, prefer_phrases: bool = True) -> List[Tuple[str, float]]:
    valid_texts = [str(t).strip() for t in texts if str(t).strip()]
    if not valid_texts:
        return []
    try:
        vectorizer = make_vectorizer(max_features=2500, ngram_range=(1, 3))
        X = vectorizer.fit_transform(valid_texts)
        scores = np.asarray(X.mean(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        ranking = pd.DataFrame({"keyphrase": terms, "score": scores})
        ranking["n_words"] = ranking["keyphrase"].str.split().str.len()
        ranking = ranking[ranking["score"] > 0]
        if prefer_phrases:
            multi = ranking[ranking["n_words"] >= 2].sort_values(["score", "n_words"], ascending=[False, False])
            uni = ranking[ranking["n_words"] == 1].sort_values("score", ascending=False)
            combined = pd.concat([multi, uni], ignore_index=True)
        else:
            combined = ranking.sort_values(["score", "n_words"], ascending=[False, False])
        combined = combined.drop_duplicates("keyphrase").head(top_n)
        return list(zip(combined["keyphrase"].tolist(), combined["score"].round(5).tolist()))
    except Exception:
        return []


def keyphrase_dataframe_by_label(result_df: pd.DataFrame, labels: List[str], text_col: str, top_n: int = 15) -> pd.DataFrame:
    rows = []
    for label in labels:
        texts = result_df.loc[result_df[f"pred_{label}"] == 1, text_col].tolist()
        for rank, (phrase, score) in enumerate(extract_keyphrases(texts, top_n=top_n, prefer_phrases=True), start=1):
            rows.append({"label": label, "rank": rank, "keyphrase": phrase, "score": score, "n_words": len(phrase.split())})
    return pd.DataFrame(rows)


def representative_comments(result_df: pd.DataFrame, label: str, original_text_col: str, top_n: int = 5) -> pd.DataFrame:
    subset = result_df.loc[result_df[f"pred_{label}"] == 1].copy()
    if subset.empty:
        return pd.DataFrame(columns=[original_text_col, f"score_{label}", "predicted_labels"])
    subset = subset.sort_values(f"score_{label}", ascending=False)
    cols = [original_text_col, f"score_{label}", "predicted_labels"]
    existing = [c for c in cols if c in subset.columns]
    return subset[existing].head(top_n)


def fallback_abstractive_summary(label: str, n_label: int, n_total: int, keyphrases: List[Tuple[str, float]], combo_summary: pd.DataFrame) -> str:
    pct = 100 * n_label / max(n_total, 1)
    phrase_text = ", ".join([p for p, _ in keyphrases[:6]]) if keyphrases else "belum ada tema dominan yang cukup kuat"
    meaning = LABEL_PEDAGOGICAL_MEANING.get(label, "merepresentasikan pola komentar tertentu.")
    action = LABEL_TEACHER_ACTION.get(label, "Gunakan informasi ini untuk mendukung analisis pembelajaran.")
    related_combos = combo_summary[combo_summary["label_combination"].str.contains(label, regex=False, na=False)].head(3)
    combo_text = "; ".join(
        [f"{r.label_combination} ({int(r['count'])} komentar)" for _, r in related_combos.iterrows()]
    ) or "tidak ada kombinasi label dominan"

    return (
        f"Pada label {label}, terdapat {n_label} dari {n_total} komentar ({pct:.2f}%). "
        f"Tema utama yang muncul berkaitan dengan {phrase_text}. Secara pedagogis, label ini {meaning} "
        f"Pola kombinasi yang relevan adalah {combo_text}. {action}"
    )


@st.cache_resource(show_spinner=False)
def load_summarizer_pipeline(model_name: str):
    from transformers import pipeline
    return pipeline("summarization", model=model_name, tokenizer=model_name)


def transformer_summary(texts: List[str], model_name: str, max_chars: int = 3500) -> Optional[str]:
    joined = " ".join([str(t).strip() for t in texts if str(t).strip()])[:max_chars]
    if len(joined.split()) < 20:
        return None
    try:
        summarizer = load_summarizer_pipeline(model_name)
        out = summarizer(joined, max_length=160, min_length=35, do_sample=False)
        return out[0].get("summary_text", "").strip() or None
    except Exception:
        return None


def label_summary_markdown(
    result_df: pd.DataFrame,
    labels: List[str],
    label_summary: pd.DataFrame,
    combo_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    keyphrase_df: pd.DataFrame,
) -> str:
    n_rows = len(result_df)
    top_label = label_summary.iloc[0]
    top_combo = combo_summary.iloc[0]

    lines = [
        "### Ringkasan Eksekutif",
        f"Sebanyak **{n_rows} peer feedback** berhasil diklasifikasikan. Karena ini adalah **multi-label classification**, satu komentar dapat memiliki lebih dari satu label sehingga total persentase antarlabel dapat melebihi 100%.",
        f"Kategori paling dominan adalah **{top_label['label']}** dengan **{int(top_label['count'])} komentar ({float(top_label['percentage']):.2f}%)**.",
        f"Kombinasi label yang paling sering muncul adalah **{top_combo['label_combination']}** dengan **{int(top_combo['count'])} komentar ({float(top_combo['percentage']):.2f}%)**.",
        "",
        "### Profil Feedback per Label",
    ]

    for label in labels:
        row = label_summary[label_summary["label"] == label].iloc[0]
        kps = keyphrase_df[keyphrase_df["label"] == label]["keyphrase"].head(7).tolist()
        kp_text = ", ".join(kps) if kps else "belum tersedia"
        lines.extend([
            f"**{label}** — {int(row['count'])} komentar ({float(row['percentage']):.2f}%).",
            f"Tema/kata kunci utama: {kp_text}.",
            f"Makna pedagogis: label ini {LABEL_PEDAGOGICAL_MEANING.get(label, '')}",
            f"Tindak lanjut pengajar: {LABEL_TEACHER_ACTION.get(label, '')}",
            "",
        ])

    lines.append("### Pola Kombinasi Label yang Perlu Diperhatikan")
    if not pair_summary.empty:
        for _, r in pair_summary[pair_summary["count"] > 0].head(6).iterrows():
            pair = r["pair"]
            count = int(r["count"])
            if pair == "Problem + Suggestion":
                interpretation = "menunjukkan feedback konstruktif karena mahasiswa menemukan masalah sekaligus memberi arah perbaikan."
            elif pair == "Appreciation + Problem":
                interpretation = "menunjukkan feedback yang relatif seimbang antara penguatan positif dan kritik."
            elif pair == "Appreciation + Suggestion":
                interpretation = "menunjukkan apresiasi yang disertai masukan perbaikan."
            else:
                interpretation = "menunjukkan adanya lebih dari satu fungsi feedback dalam komentar yang sama."
            lines.append(f"- **{pair}**: {count} komentar; {interpretation}")
    else:
        lines.append("- Belum ada pola kombinasi label yang dapat dihitung.")

    neutral_count = int(label_summary.loc[label_summary["label"] == "Neutral", "count"].sum())
    suggestion_count = int(label_summary.loc[label_summary["label"] == "Suggestion", "count"].sum())
    appreciation_count = int(label_summary.loc[label_summary["label"] == "Appreciation", "count"].sum())
    lines.extend(["", "### Implikasi untuk Pengajar"])
    if appreciation_count > suggestion_count:
        lines.append("- Apresiasi lebih dominan daripada saran. Pengajar dapat mendorong mahasiswa melengkapi pujian dengan alasan spesifik dan rekomendasi yang dapat ditindaklanjuti.")
    if neutral_count / max(n_rows, 1) >= 0.20:
        lines.append("- Proporsi Neutral cukup tinggi. Pertimbangkan pemberian contoh komentar, sentence starter, atau rubrik agar feedback lebih evaluatif dan informatif.")
    lines.append("- Gunakan tema/kata kunci per label untuk mengidentifikasi aspek materi, produk, atau kinerja yang paling sering diapresiasi, dikritik, atau disarankan untuk diperbaiki.")
    return "\n".join(lines)


def run_topic_modeling(texts: List[str], labels_for_docs: List[str], n_topics: int = 5, top_terms: int = 10):
    valid = pd.DataFrame({"text": texts, "predicted_labels": labels_for_docs})
    valid["text"] = valid["text"].fillna("").astype(str)
    valid = valid[valid["text"].str.strip() != ""].reset_index(drop=True)
    if len(valid) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    n_topics = max(1, min(int(n_topics), len(valid), 10))
    vectorizer = make_vectorizer(max_features=2000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(valid["text"].tolist())
    if X.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=600)
    W = model.fit_transform(X)
    H = model.components_
    terms = np.array(vectorizer.get_feature_names_out())

    topic_rows = []
    for topic_idx, comp in enumerate(H):
        top_idx = comp.argsort()[::-1][:top_terms]
        topic_terms = terms[top_idx].tolist()
        topic_rows.append({
            "topic_id": int(topic_idx + 1),
            "top_keyphrases": ", ".join(topic_terms),
            "topic_label": " / ".join(topic_terms[:3]),
        })
    topics_df = pd.DataFrame(topic_rows)

    dominant = W.argmax(axis=1) + 1
    confidence = W.max(axis=1) / np.maximum(W.sum(axis=1), 1e-12)
    doc_topics = valid.copy()
    doc_topics["topic_id"] = dominant.astype(int)
    doc_topics["topic_confidence"] = np.round(confidence, 4)
    doc_topics = doc_topics.merge(topics_df, on="topic_id", how="left")

    topic_distribution = (
        doc_topics.groupby(["topic_id", "topic_label"], as_index=False)
        .agg(count=("text", "size"), dominant_predicted_labels=("predicted_labels", lambda x: x.value_counts().index[0]))
        .sort_values("count", ascending=False)
    )
    topic_distribution["percentage"] = (100 * topic_distribution["count"] / len(doc_topics)).round(2)
    return topics_df, topic_distribution, doc_topics

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Peer Feedback Multi-Label Classifier", page_icon="🧠", layout="wide")

st.title("🧠 Peer Feedback Multi-Label Classifier")
st.caption("Klasifikasi komentar peer review menjadi Appreciation, Problem, Suggestion, dan Neutral, lalu mengubahnya menjadi learning insights.")

with st.sidebar:
    st.header("Pengaturan Model")
    artifact_dir = st.text_input("Folder artefak model", value=DEFAULT_ARTIFACT_DIR)
    model_choice = st.radio("Pilih model terbaik", options=["Machine Learning", "Deep Learning"], horizontal=False)

    st.divider()
    st.header("Pengaturan Analisis")
    top_n_keyphrases = st.slider("Jumlah keyphrase per label", 5, 30, 15)
    n_topics = st.slider("Jumlah topik global NMF", 2, 10, 5)
    use_transformer_summary = st.checkbox("Gunakan Transformer abstractive summarization jika tersedia", value=False)
    summarizer_model = st.text_input("Model summarization", value="cahya/t5-base-indonesian-summarization-cased")

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
    keyphrase_df = keyphrase_dataframe_by_label(result_df, labels, "text_clean", top_n=top_n_keyphrases)
    readable_summary = label_summary_markdown(result_df, labels, label_summary, combo_summary, pair_summary, keyphrase_df)

    topics_df, topic_distribution, doc_topics = run_topic_modeling(
        result_df["text_clean"].tolist(),
        result_df["predicted_labels"].tolist(),
        n_topics=n_topics,
        top_terms=12,
    )

    st.success(f"Klasifikasi selesai menggunakan model: {model_choice}")

    st.subheader("2. Ringkasan Learning Insight")
    metric_cols = st.columns(len(labels))
    for idx, label in enumerate(labels):
        count = int(label_summary.loc[label_summary["label"] == label, "count"].iloc[0])
        pct = float(label_summary.loc[label_summary["label"] == label, "percentage"].iloc[0])
        metric_cols[idx].metric(label, f"{count}", f"{pct}%")
    st.markdown(readable_summary)

    st.subheader("3. Distribusi dan Kombinasi Label")
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

    st.subheader("5. Tab Analisis per Label")
    st.caption("Setiap label memiliki summary, wordcloud, keyphrases/topik utama, dan contoh komentar representatif.")
    label_tabs = st.tabs(labels)
    for tab, label in zip(label_tabs, labels):
        with tab:
            label_texts_clean = result_df.loc[result_df[f"pred_{label}"] == 1, "text_clean"].tolist()
            label_texts_original = result_df.loc[result_df[f"pred_{label}"] == 1, text_col].fillna("").astype(str).tolist()
            label_count = len(label_texts_clean)
            label_kps = [
                (r["keyphrase"], r["score"])
                for _, r in keyphrase_df[keyphrase_df["label"] == label].head(top_n_keyphrases).iterrows()
            ]
            sub_summary, sub_wc, sub_keyphrases, sub_examples = st.tabs(["Summary", "WordCloud", "Keyphrases/Topik Utama", "Contoh Komentar"])

            with sub_summary:
                st.markdown(f"#### Summary {label}")
                if label_count == 0:
                    st.info(f"Tidak ada komentar yang diklasifikasikan sebagai {label}.")
                else:
                    transformer_text = None
                    if use_transformer_summary:
                        with st.spinner(f"Membuat abstractive summary Transformer untuk {label}..."):
                            transformer_text = transformer_summary(label_texts_original, summarizer_model)
                    if transformer_text:
                        st.markdown("**Abstractive summary berbasis Transformer:**")
                        st.write(transformer_text)
                        st.markdown("**Interpretasi pedagogis:**")
                        st.write(fallback_abstractive_summary(label, label_count, len(result_df), label_kps, combo_summary))
                    else:
                        st.markdown("**Abstractive topic-guided summary:**")
                        st.write(fallback_abstractive_summary(label, label_count, len(result_df), label_kps, combo_summary))
                    st.info(LABEL_TEACHER_ACTION.get(label, ""))

            with sub_wc:
                st.markdown(f"#### WordCloud {label}")
                image = generate_wordcloud(label_texts_clean)
                if image is None:
                    st.info(f"Tidak ada teks yang diklasifikasikan sebagai {label}.")
                else:
                    st.image(image, use_container_width=True)

            with sub_keyphrases:
                st.markdown(f"#### Keyphrases dan Topik Utama {label}")
                label_kp_df = keyphrase_df[keyphrase_df["label"] == label].copy()
                if label_kp_df.empty:
                    st.info(f"Belum ada keyphrase untuk label {label}.")
                else:
                    st.write("Keyphrase diekstrak dengan TF-IDF unigram, bigram, dan trigram. Frasa multi-kata diprioritaskan agar informasi tidak hanya berupa satu kata.")
                    st.dataframe(label_kp_df, use_container_width=True)
                    chart_df = label_kp_df.head(12).set_index("keyphrase")["score"]
                    st.bar_chart(chart_df)

            with sub_examples:
                st.markdown(f"#### Contoh Komentar Representatif {label}")
                examples = representative_comments(result_df, label, text_col, top_n=8)
                if examples.empty:
                    st.info(f"Tidak ada contoh komentar untuk label {label}.")
                else:
                    st.dataframe(examples, use_container_width=True)

    st.subheader("6. Keyphrase Extraction & Topic Modelling per Label")
    st.caption(
        "Analisis keyphrase dan topic modelling dikelompokkan per label agar pengajar dapat melihat tema utama "
        "yang muncul pada Appreciation, Problem, Suggestion, dan Neutral secara terpisah. "
        "Keyphrase menggunakan TF-IDF unigram, bigram, dan trigram; frasa multi-kata diprioritaskan."
    )

    # Container untuk menggabungkan semua hasil topic modelling per label agar bisa diunduh.
    all_label_topics = []
    all_label_topic_distributions = []
    all_label_doc_topics = []

    per_label_topic_tabs = st.tabs(labels)
    for label_tab, label in zip(per_label_topic_tabs, labels):
        with label_tab:
            label_subset = result_df.loc[result_df[f"pred_{label}"] == 1].copy()
            label_texts = label_subset["text_clean"].fillna("").astype(str).tolist()
            label_predicted_labels = label_subset["predicted_labels"].fillna("").astype(str).tolist()
            label_kp_df = keyphrase_df[keyphrase_df["label"] == label].copy()

            st.markdown(f"#### {label}")
            st.write(
                f"Bagian ini menampilkan keyphrase dan topik utama dari **{len(label_subset)} komentar** "
                f"yang diklasifikasikan sebagai **{label}**."
            )

            kp_tab, topic_tab, assign_tab = st.tabs([
                "Keyphrase per Label",
                "NMF Topics per Label",
                "Assignment Topik Dokumen",
            ])

            with kp_tab:
                st.markdown("##### Keyphrase utama")
                if label_kp_df.empty:
                    st.info(f"Belum ada keyphrase untuk label {label}.")
                else:
                    st.write(
                        "Keyphrase diekstrak dari komentar pada label ini menggunakan TF-IDF unigram, bigram, "
                        "dan trigram. Kolom `n_words` membantu membedakan kata tunggal dan frasa multi-kata."
                    )
                    st.dataframe(label_kp_df, use_container_width=True)
                    st.bar_chart(label_kp_df.head(15).set_index("keyphrase")["score"])
                    st.download_button(
                        label=f"⬇️ Download keyphrase {label} CSV",
                        data=dataframe_to_csv_bytes(label_kp_df),
                        file_name=f"peer_feedback_keyphrases_{label.lower()}.csv",
                        mime="text/csv",
                    )

            with topic_tab:
                st.markdown("##### Topic modelling NMF khusus label")
                st.write(
                    "Topic modelling pada tab ini hanya menggunakan komentar yang termasuk label ini, "
                    "sehingga topik yang muncul lebih spesifik dibanding topic modelling global."
                )
                label_n_topics = min(n_topics, max(1, len(label_subset)))
                label_topics_df, label_topic_distribution, label_doc_topics = run_topic_modeling(
                    label_texts,
                    label_predicted_labels,
                    n_topics=label_n_topics,
                    top_terms=12,
                )

                if label_topics_df.empty:
                    st.info(
                        f"Topic modelling untuk label {label} belum dapat dibuat. "
                        "Kemungkinan jumlah komentar atau variasi kata terlalu sedikit."
                    )
                else:
                    label_topics_df = label_topics_df.copy()
                    label_topic_distribution = label_topic_distribution.copy()
                    label_doc_topics = label_doc_topics.copy()
                    label_topics_df.insert(0, "label", label)
                    label_topic_distribution.insert(0, "label", label)
                    label_doc_topics.insert(0, "label", label)

                    all_label_topics.append(label_topics_df)
                    all_label_topic_distributions.append(label_topic_distribution)
                    all_label_doc_topics.append(label_doc_topics)

                    st.write("**Topik utama pada label ini**")
                    st.dataframe(label_topics_df, use_container_width=True)
                    st.write("**Distribusi topik pada label ini**")
                    st.dataframe(label_topic_distribution, use_container_width=True)
                    st.bar_chart(label_topic_distribution.set_index("topic_label")["count"])

            with assign_tab:
                st.markdown("##### Assignment topik untuk dokumen pada label ini")
                # Jika topic modelling belum berhasil dijalankan pada tab topic, jalankan ulang ringan di sini.
                if 'label_doc_topics' not in locals() or label_doc_topics.empty:
                    tmp_topics, tmp_dist, tmp_docs = run_topic_modeling(
                        label_texts,
                        label_predicted_labels,
                        n_topics=min(n_topics, max(1, len(label_subset))),
                        top_terms=12,
                    )
                    label_doc_topics_to_show = tmp_docs
                else:
                    label_doc_topics_to_show = label_doc_topics

                if label_doc_topics_to_show.empty:
                    st.info(f"Assignment topik untuk label {label} belum tersedia.")
                else:
                    show_cols = ["text", "predicted_labels", "topic_id", "topic_label", "topic_confidence"]
                    existing_cols = [c for c in show_cols if c in label_doc_topics_to_show.columns]
                    st.dataframe(label_doc_topics_to_show[existing_cols], use_container_width=True)
                    st.download_button(
                        label=f"⬇️ Download assignment topik {label} CSV",
                        data=dataframe_to_csv_bytes(label_doc_topics_to_show),
                        file_name=f"peer_feedback_topic_assignment_{label.lower()}.csv",
                        mime="text/csv",
                    )

    st.markdown("#### Download gabungan hasil Keyphrase & Topic Modelling")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            label="⬇️ Semua keyphrase per label",
            data=dataframe_to_csv_bytes(keyphrase_df),
            file_name="peer_feedback_keyphrases_by_label.csv",
            mime="text/csv",
        )
    with col_dl2:
        if all_label_topics:
            combined_topics = pd.concat(all_label_topics, ignore_index=True)
            st.download_button(
                label="⬇️ Semua topik per label",
                data=dataframe_to_csv_bytes(combined_topics),
                file_name="peer_feedback_nmf_topics_by_label.csv",
                mime="text/csv",
            )
        else:
            st.button("⬇️ Semua topik per label", disabled=True)
    with col_dl3:
        if all_label_doc_topics:
            combined_doc_topics = pd.concat(all_label_doc_topics, ignore_index=True)
            st.download_button(
                label="⬇️ Semua assignment topik",
                data=dataframe_to_csv_bytes(combined_doc_topics),
                file_name="peer_feedback_topic_assignment_by_label.csv",
                mime="text/csv",
            )
        else:
            st.button("⬇️ Semua assignment topik", disabled=True)

    with st.expander("Lihat juga topic modelling global seluruh komentar", expanded=False):
        st.write(
            "Bagian ini bersifat tambahan. Berbeda dari analisis per label, topic modelling global menggunakan semua komentar sekaligus."
        )
        if topics_df.empty:
            st.info("Topic modelling global belum dapat dibuat karena jumlah teks/fitur tidak mencukupi.")
        else:
            st.write("**NMF Global Topics**")
            st.dataframe(topics_df, use_container_width=True)
            st.write("**Distribusi topik global**")
            st.dataframe(topic_distribution, use_container_width=True)
            st.bar_chart(topic_distribution.set_index("topic_label")["count"])
            st.write("**Assignment topik global per dokumen**")
            show_cols = ["text", "predicted_labels", "topic_id", "topic_label", "topic_confidence"]
            st.dataframe(doc_topics[show_cols], use_container_width=True)
            st.download_button(
                label="⬇️ Download assignment topik global CSV",
                data=dataframe_to_csv_bytes(doc_topics),
                file_name="peer_feedback_topic_assignment_global.csv",
                mime="text/csv",
            )

    st.subheader("7. Hasil Klasifikasi")
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

    st.subheader("8. Metadata Model")
    st.json(metadata)
