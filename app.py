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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

# TensorFlow/Keras is imported lazily only when DL is used.


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
    "tidak", "bisa", "dapat", "sangat", "masih", "ada", "jadi", "tersebut", "nya", "nyaa",
    "saya", "aku", "kamu", "anda", "dia", "mereka", "kami", "kita", "the", "a", "an", "and",
    "or", "to", "of", "in", "is", "are", "for", "with", "on", "this", "that", "it", "be",
    "peer", "feedback", "review", "komentar", "menurut", "mungkin", "cukup", "bagian",
}

LABEL_PEDAGOGICAL_HINTS = {
    "Appreciation": "aspek yang dipersepsi positif atau sudah baik oleh mahasiswa",
    "Problem": "aspek yang dianggap bermasalah, kurang jelas, salah, atau perlu diperbaiki",
    "Suggestion": "arah perbaikan, rekomendasi tindakan, atau masukan konkret dari mahasiswa",
    "Neutral": "komentar yang cenderung umum, deskriptif, atau belum memberikan informasi evaluatif yang kuat",
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
    """Load Keras 3 or TensorFlow/Keras model as robustly as possible."""
    artifact_path = Path(artifact_dir)
    model_path = artifact_path / "best_dl_model.keras"

    try:
        import keras
        model = keras.saving.load_model(model_path, compile=False, safe_mode=False)
        pad_sequences = keras.utils.pad_sequences
    except Exception:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        model = load_model(model_path, compile=False)

    tokenizer = joblib.load(artifact_path / "best_dl_tokenizer.joblib")
    metadata = read_json(artifact_path / "best_dl_model_metadata.json")
    labels = metadata.get("labels") or DEFAULT_LABELS
    thresholds = thresholds_from_metadata(metadata, labels)
    max_len = int(metadata.get("dl_max_len", metadata.get("max_len", 120)))
    return model, tokenizer, metadata, labels, thresholds, max_len, pad_sequences


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
    model, tokenizer, metadata, labels, thresholds, max_len, pad_sequences = load_dl_artifact(artifact_dir)
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
        collocations=False,
        max_words=120,
    ).generate(text)
    return wc.to_image()


@st.cache_resource(show_spinner=False)
def load_summarization_pipeline(model_name: str):
    """Load an optional abstractive summarization model.

    The app still works without this model. If the model cannot be loaded
    because transformers/torch are unavailable or the server has no internet,
    the app falls back to a topic-guided abstractive synthesis.
    """
    from transformers import pipeline

    return pipeline("summarization", model=model_name, tokenizer=model_name)


def compact_text_for_summary(texts: List[str], max_comments: int = 40, max_chars: int = 4500) -> str:
    """Select representative text snippets and keep the summarization input bounded."""
    valid = _valid_texts(texts)
    if not valid:
        return ""
    # Prefer diverse medium-length comments rather than a single very long text.
    valid = sorted(valid, key=lambda x: min(len(x), 500), reverse=True)[:max_comments]
    merged = " ".join(valid)
    return merged[:max_chars]


def fallback_abstractive_label_summary(
    label: str,
    count: int,
    pct: float,
    topic_df: pd.DataFrame,
    combo_summary: Optional[pd.DataFrame] = None,
) -> str:
    """Generate a concise abstractive, topic-guided teaching summary without copying comments."""
    topics = topic_df[(topic_df.get("label") == label) & (topic_df.get("topic") != "-")].head(8)
    topic_terms = topics["topic"].astype(str).tolist() if not topics.empty else []
    topic_phrase = ", ".join(topic_terms[:6]) if topic_terms else "belum menunjukkan tema yang dominan"

    if label == "Appreciation":
        core = (
            f"Sebanyak {count} komentar ({pct:.2f}%) menunjukkan apresiasi. "
            f"Tema yang paling menonjol berkaitan dengan {topic_phrase}. "
            "Hal ini mengindikasikan aspek pekerjaan atau proses belajar yang sudah dipersepsi positif oleh mahasiswa. "
            "Pengajar dapat menggunakan informasi ini untuk mengidentifikasi elemen yang perlu dipertahankan atau dijadikan contoh praktik baik."
        )
    elif label == "Problem":
        core = (
            f"Sebanyak {count} komentar ({pct:.2f}%) mengandung identifikasi masalah. "
            f"Isu yang sering muncul berkaitan dengan {topic_phrase}. "
            "Komentar pada label ini penting untuk membaca bagian pekerjaan yang dianggap kurang jelas, belum tepat, atau membutuhkan perbaikan. "
            "Pengajar dapat menindaklanjuti temuan ini dengan memberi klarifikasi, contoh, atau rubrik yang lebih eksplisit."
        )
    elif label == "Suggestion":
        core = (
            f"Sebanyak {count} komentar ({pct:.2f}%) berisi saran perbaikan. "
            f"Arah saran yang dominan berkaitan dengan {topic_phrase}. "
            "Label ini merepresentasikan potensi feedback yang paling actionable karena mahasiswa tidak hanya menilai, tetapi juga menawarkan langkah perbaikan. "
            "Pengajar dapat memperkuat kemampuan ini melalui latihan memberi masukan yang spesifik dan dapat ditindaklanjuti."
        )
    elif label == "Neutral":
        core = (
            f"Sebanyak {count} komentar ({pct:.2f}%) diklasifikasikan sebagai netral. "
            f"Topik yang muncul berkaitan dengan {topic_phrase}. "
            "Komentar netral cenderung belum memberikan evaluasi atau saran yang kuat, sehingga nilai formatifnya lebih terbatas. "
            "Jika proporsinya tinggi, pengajar perlu memberikan scaffolding tentang kriteria komentar yang informatif, spesifik, dan konstruktif."
        )
    else:
        core = (
            f"Sebanyak {count} komentar ({pct:.2f}%) diklasifikasikan sebagai {label}. "
            f"Tema utama yang muncul adalah {topic_phrase}. "
            "Informasi ini dapat digunakan untuk membaca kecenderungan isi peer feedback pada kategori tersebut."
        )
    return core


def generate_abstractive_label_summary(
    label: str,
    texts: List[str],
    count: int,
    pct: float,
    topic_summary: pd.DataFrame,
    use_transformer: bool,
    summarizer_model_name: str,
    max_comments: int,
) -> Tuple[str, str]:
    """Generate an abstractive label-level summary.

    Returns (summary, method). When a Transformer summarizer is unavailable,
    the function produces a topic-guided abstractive synthesis so the app remains usable.
    """
    label_topics = topic_summary[topic_summary["label"] == label].copy() if not topic_summary.empty else pd.DataFrame()
    fallback = fallback_abstractive_label_summary(label, count, pct, topic_summary)

    if not use_transformer:
        return fallback, "topic-guided abstractive synthesis"

    source_text = compact_text_for_summary(texts, max_comments=max_comments)
    if not source_text.strip():
        return fallback, "fallback: no text available"

    try:
        summarizer = load_summarization_pipeline(summarizer_model_name)
        # Indonesian T5-style models usually work better with a short instruction prefix.
        input_text = (
            f"Ringkas secara abstraktif isi komentar peer feedback berikut untuk label {label}. "
            f"Fokus pada topik utama dan implikasi untuk pengajar: {source_text}"
        )
        output = summarizer(
            input_text,
            max_length=180,
            min_length=45,
            do_sample=False,
            truncation=True,
        )
        generated = output[0].get("summary_text", "").strip()
        if not generated:
            return fallback, "fallback: empty transformer output"

        topics = top_topics_text(topic_summary, label, n=5)
        pedagogical_note = (
            f"\n\nInterpretasi untuk pengajar: label {label} merepresentasikan "
            f"{LABEL_PEDAGOGICAL_HINTS.get(label, 'pola komentar pada kategori tersebut')}. "
            f"Topik dominan yang terdeteksi: {topics}."
        )
        return generated + pedagogical_note, f"Transformer abstractive summarization ({summarizer_model_name})"
    except Exception as exc:
        return fallback + f"\n\nCatatan sistem: model abstractive summarization tidak dapat dimuat, sehingga aplikasi memakai fallback berbasis topik. Detail: {exc}", "fallback: transformer unavailable"


def build_topic_model(
    texts: List[str],
    result_df: pd.DataFrame,
    labels: List[str],
    n_topics: int = 5,
    top_words: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run lightweight NMF topic modelling over uploaded feedback texts."""
    valid_idx = [i for i, t in enumerate(texts) if str(t).strip()]
    valid_texts = [str(texts[i]) for i in valid_idx]
    if len(valid_texts) < 3:
        return (
            pd.DataFrame(columns=["topic_id", "top_terms", "document_count", "dominant_predicted_labels"]),
            pd.DataFrame(columns=["topic_id", "feedback_index", "dominant_topic_score", "predicted_labels"]),
        )

    n_topics = max(1, min(int(n_topics), len(valid_texts), 10))
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            max_features=2000,
            stop_words=list(STOPWORDS_ID),
            token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_-]{2,}\b",
        )
        X = vectorizer.fit_transform(valid_texts)
        if X.shape[1] < n_topics:
            n_topics = max(1, X.shape[1])
        nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=500)
        W = nmf.fit_transform(X)
        H = nmf.components_
        terms = np.array(vectorizer.get_feature_names_out())
        topic_assignments = W.argmax(axis=1)
        topic_scores = W.max(axis=1)

        topic_rows = []
        doc_rows = []
        tmp_df = result_df.iloc[valid_idx].copy().reset_index(drop=False)
        for topic_id in range(n_topics):
            term_ids = H[topic_id].argsort()[::-1][:top_words]
            top_terms = terms[term_ids].tolist()
            member_mask = topic_assignments == topic_id
            member_count = int(member_mask.sum())
            if member_count > 0:
                member_df = tmp_df.loc[member_mask]
                label_counts = {}
                for label in labels:
                    col = f"pred_{label}"
                    if col in member_df.columns:
                        label_counts[label] = int(member_df[col].sum())
                dominant_labels = ", ".join([k for k, v in sorted(label_counts.items(), key=lambda x: x[1], reverse=True) if v > 0][:3]) or "None"
            else:
                dominant_labels = "None"
            topic_rows.append({
                "topic_id": f"Topic {topic_id + 1}",
                "top_terms": ", ".join(top_terms),
                "document_count": member_count,
                "dominant_predicted_labels": dominant_labels,
            })

        for local_i, original_i in enumerate(valid_idx):
            doc_rows.append({
                "topic_id": f"Topic {int(topic_assignments[local_i]) + 1}",
                "feedback_index": int(original_i),
                "dominant_topic_score": round(float(topic_scores[local_i]), 4),
                "predicted_labels": result_df.iloc[original_i].get("predicted_labels", ""),
            })
        return pd.DataFrame(topic_rows), pd.DataFrame(doc_rows)
    except Exception as exc:
        return (
            pd.DataFrame([{"topic_id": "Error", "top_terms": str(exc), "document_count": 0, "dominant_predicted_labels": "-"}]),
            pd.DataFrame(columns=["topic_id", "feedback_index", "dominant_topic_score", "predicted_labels"]),
        )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _valid_texts(texts: List[str]) -> List[str]:
    return [str(t).strip() for t in texts if str(t).strip()]


def extract_topics_for_texts(texts: List[str], top_n: int = 10) -> pd.DataFrame:
    """Extract key unigram/bigram topics from a list of texts using mean TF-IDF."""
    texts = _valid_texts(texts)
    if not texts:
        return pd.DataFrame(columns=["topic", "score", "document_count"])

    # If there is only one very short text, TF-IDF may fail after stopword removal.
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=1000,
            stop_words=list(STOPWORDS_ID),
            token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_-]{2,}\b",
        )
        X = vectorizer.fit_transform(texts)
        terms = np.array(vectorizer.get_feature_names_out())
        scores = np.asarray(X.mean(axis=0)).ravel()
        doc_counts = np.asarray((X > 0).sum(axis=0)).ravel()
        order = np.argsort(scores)[::-1]
        rows = []
        for idx in order[:top_n]:
            rows.append({
                "topic": terms[idx],
                "score": round(float(scores[idx]), 4),
                "document_count": int(doc_counts[idx]),
            })
        return pd.DataFrame(rows)
    except ValueError:
        tokens = []
        for text in texts:
            tokens.extend([t for t in re.findall(r"[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_-]{2,}", text.lower()) if t not in STOPWORDS_ID])
        counts = Counter(tokens)
        rows = [{"topic": term, "score": float(count), "document_count": int(count)} for term, count in counts.most_common(top_n)]
        return pd.DataFrame(rows)


def build_label_topic_summary(
    result_df: pd.DataFrame,
    labels: List[str],
    raw_text_col: str,
    top_n: int = 8,
    max_examples: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build topic/keyword summaries and representative examples for each predicted label."""
    topic_frames = []
    example_rows = []

    for label in labels:
        mask = result_df[f"pred_{label}"] == 1
        label_df = result_df.loc[mask].copy()
        texts = label_df["text_clean"].tolist()
        topics = extract_topics_for_texts(texts, top_n=top_n)
        if topics.empty:
            topic_frames.append(pd.DataFrame([{
                "label": label,
                "rank": None,
                "topic": "-",
                "score": 0.0,
                "document_count": 0,
            }]))
        else:
            topics.insert(0, "rank", range(1, len(topics) + 1))
            topics.insert(0, "label", label)
            topic_frames.append(topics)

        if not label_df.empty:
            score_col = f"score_{label}"
            if score_col in label_df.columns:
                label_df = label_df.sort_values(score_col, ascending=False)
            for _, row in label_df.head(max_examples).iterrows():
                example_rows.append({
                    "label": label,
                    "score": row.get(score_col, np.nan),
                    "example_feedback": str(row.get(raw_text_col, ""))[:350],
                })

    topic_summary = pd.concat(topic_frames, ignore_index=True) if topic_frames else pd.DataFrame()
    examples = pd.DataFrame(example_rows)
    return topic_summary, examples


def top_topics_text(topic_summary: pd.DataFrame, label: str, n: int = 5) -> str:
    rows = topic_summary[(topic_summary["label"] == label) & (topic_summary["topic"] != "-")].head(n)
    if rows.empty:
        return "belum ada topik dominan"
    return ", ".join(rows["topic"].astype(str).tolist())


def generate_learning_insight_text(
    n_rows: int,
    label_summary: pd.DataFrame,
    combo_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    topic_summary: pd.DataFrame,
) -> str:
    top_label = label_summary.iloc[0]
    top_combo = combo_summary.iloc[0]
    get_count = lambda label: int(label_summary.loc[label_summary["label"] == label, "count"].sum())
    get_pct = lambda label: float(label_summary.loc[label_summary["label"] == label, "percentage"].sum())

    problem_count = get_count("Problem")
    suggestion_count = get_count("Suggestion")
    appreciation_count = get_count("Appreciation")
    neutral_count = get_count("Neutral")
    problem_suggestion = int(pair_summary.loc[pair_summary["pair"] == "Problem + Suggestion", "count"].sum())
    appreciation_problem = int(pair_summary.loc[pair_summary["pair"] == "Appreciation + Problem", "count"].sum())

    lines = [
        f"Sebanyak {n_rows} peer feedback berhasil diklasifikasikan.",
        f"Kategori dominan adalah {top_label['label']} ({int(top_label['count'])} komentar; {top_label['percentage']}%).",
        f"Kombinasi label paling sering muncul adalah {top_combo['label_combination']} ({int(top_combo['count'])} komentar; {top_combo['percentage']}%).",
        "",
        "Ringkasan topik per label:",
    ]

    for label in ["Appreciation", "Problem", "Suggestion", "Neutral"]:
        if label in label_summary["label"].values:
            topics = top_topics_text(topic_summary, label, n=5)
            lines.append(
                f"- {label}: {get_count(label)} komentar ({get_pct(label):.2f}%). Topik/kata kunci utama: {topics}. "
                f"Ini merepresentasikan {LABEL_PEDAGOGICAL_HINTS.get(label, 'pola komentar pada label tersebut')}."
            )

    lines.append("")
    lines.append("Interpretasi pedagogis untuk pengajar:")

    if problem_suggestion > 0:
        lines.append(
            f"- {problem_suggestion} komentar memuat Problem + Suggestion. Ini merupakan sinyal feedback konstruktif karena mahasiswa tidak hanya menemukan masalah, tetapi juga memberi arah perbaikan."
        )
    elif problem_count > 0 and suggestion_count == 0:
        lines.append(
            "- Komentar bermuatan Problem muncul, tetapi belum disertai Suggestion. Pengajar dapat memberi scaffolding agar mahasiswa tidak hanya mengkritik, tetapi juga menawarkan solusi konkret."
        )

    if appreciation_problem > 0:
        lines.append(
            f"- {appreciation_problem} komentar memuat Appreciation + Problem. Pola ini menunjukkan feedback yang relatif seimbang antara penguatan positif dan kritik."
        )

    if appreciation_count > suggestion_count and suggestion_count > 0:
        lines.append(
            "- Appreciation lebih dominan daripada Suggestion. Pengajar dapat mendorong mahasiswa untuk melengkapi pujian dengan masukan yang lebih spesifik dan dapat ditindaklanjuti."
        )

    neutral_pct = get_pct("Neutral") if "Neutral" in label_summary["label"].values else 0.0
    if neutral_pct >= 20:
        lines.append(
            f"- Neutral mencapai {neutral_pct:.2f}%. Proporsi ini menunjukkan perlunya contoh rubrik atau panduan komentar agar feedback lebih evaluatif dan bermanfaat."
        )

    lines.append(
        "- Topik/kata kunci per label dapat digunakan untuk mengidentifikasi aspek materi, produk, atau kinerja yang sering diapresiasi, dikritik, atau disarankan untuk diperbaiki."
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
    st.header("Pengaturan Insight")
    top_n_topics = st.slider("Jumlah topik/kata kunci per label", min_value=5, max_value=20, value=10, step=1)
    max_examples = st.slider("Jumlah contoh komentar representatif per label", min_value=1, max_value=5, value=3, step=1)
    n_topic_model = st.slider("Jumlah topik global/NMF", min_value=2, max_value=8, value=4, step=1)

    st.write("**Abstractive summarization**")
    use_transformer_summary = st.checkbox(
        "Gunakan model Transformer jika tersedia",
        value=False,
        help="Jika aktif, aplikasi mencoba memuat model summarization dari Hugging Face. Jika gagal, otomatis memakai fallback abstractive berbasis topik.",
    )
    summarizer_model_name = st.text_input(
        "Model summarization",
        value="cahya/t5-base-indonesian-summarization-cased",
        help="Gunakan model summarization Bahasa Indonesia yang tersedia di environment atau dapat diunduh oleh server.",
    )
    max_summary_comments = st.slider("Maksimal komentar per label untuk summary", min_value=10, max_value=80, value=40, step=5)

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
    topic_summary, representative_examples = build_label_topic_summary(
        result_df=result_df,
        labels=labels,
        raw_text_col=text_col,
        top_n=top_n_topics,
        max_examples=max_examples,
    )
    insight_text = generate_learning_insight_text(len(result_df), label_summary, combo_summary, pair_summary, topic_summary)
    topic_model_summary, topic_doc_assignments = build_topic_model(
        texts=result_df["text_clean"].tolist(),
        result_df=result_df,
        labels=labels,
        n_topics=n_topic_model,
        top_words=10,
    )

    # Generate label-level abstractive summaries once, after prediction.
    label_abstractive_summaries = {}
    for label in labels:
        label_texts = result_df.loc[result_df[f"pred_{label}"] == 1, text_col].astype(str).tolist()
        count = int(label_summary.loc[label_summary["label"] == label, "count"].iloc[0])
        pct = float(label_summary.loc[label_summary["label"] == label, "percentage"].iloc[0])
        summary, method = generate_abstractive_label_summary(
            label=label,
            texts=label_texts,
            count=count,
            pct=pct,
            topic_summary=topic_summary,
            use_transformer=use_transformer_summary,
            summarizer_model_name=summarizer_model_name,
            max_comments=max_summary_comments,
        )
        label_abstractive_summaries[label] = {"summary": summary, "method": method}

    st.success(f"Klasifikasi selesai menggunakan model: {model_choice}")

    st.subheader("2. Ringkasan Learning Insight")
    metric_cols = st.columns(len(labels))
    for idx, label in enumerate(labels):
        count = int(label_summary.loc[label_summary["label"] == label, "count"].iloc[0])
        pct = float(label_summary.loc[label_summary["label"] == label, "percentage"].iloc[0])
        metric_cols[idx].metric(label, f"{count}", f"{pct}%")

    st.text_area("Summary insight global untuk pengajar", value=insight_text, height=320)

    st.subheader("3. Analisis per Label")
    st.write(
        "Setiap tab label berisi summary abstraktif, wordcloud, topik/kata kunci utama, dan contoh komentar representatif. "
        "Bagian ini dirancang agar pengajar dapat memahami isi feedback, bukan hanya jumlah prediksi label."
    )

    label_tabs = st.tabs(labels)
    for tab, label in zip(label_tabs, labels):
        with tab:
            label_topics = topic_summary[topic_summary["label"] == label].copy()
            label_examples = representative_examples[representative_examples["label"] == label].copy()
            count = int(label_summary.loc[label_summary["label"] == label, "count"].iloc[0])
            pct = float(label_summary.loc[label_summary["label"] == label, "percentage"].iloc[0])
            label_texts_clean = result_df.loc[result_df[f"pred_{label}"] == 1, "text_clean"].tolist()

            st.markdown(f"### {label}: {count} komentar ({pct}%)")
            st.caption(LABEL_PEDAGOGICAL_HINTS.get(label, "Topik dominan pada label ini."))

            sub_tabs = st.tabs(["Summary", "WordCloud", "Keyphrases", "Contoh Komentar"])
            with sub_tabs[0]:
                method = label_abstractive_summaries[label]["method"]
                st.caption(f"Metode summary: {method}")
                st.markdown(label_abstractive_summaries[label]["summary"])

                if label == "Problem":
                    ps_count = int(pair_summary.loc[pair_summary["pair"] == "Problem + Suggestion", "count"].sum())
                    if count > 0:
                        st.info(
                            f"Dari {count} komentar Problem, {ps_count} juga memuat Suggestion. "
                            "Semakin tinggi kombinasi ini, semakin konstruktif pola feedback mahasiswa."
                        )
                if label == "Neutral" and pct >= 20:
                    st.warning(
                        "Proporsi Neutral cukup tinggi. Pengajar dapat mempertimbangkan pemberian contoh komentar, rubrik, "
                        "atau kalimat pemantik agar feedback mahasiswa lebih spesifik dan actionable."
                    )

            with sub_tabs[1]:
                image = generate_wordcloud(label_texts_clean)
                if image is None:
                    st.info(f"Tidak ada teks yang diklasifikasikan sebagai {label}.")
                else:
                    st.image(image, use_container_width=True)

            with sub_tabs[2]:
                if label_topics.empty or label_topics["topic"].iloc[0] == "-":
                    st.info(f"Belum ada keyphrase/topik dominan untuk label {label}.")
                else:
                    st.dataframe(label_topics, use_container_width=True)
                    st.bar_chart(label_topics.set_index("topic")["score"])

            with sub_tabs[3]:
                if label_examples.empty:
                    st.info(f"Tidak ada komentar yang diprediksi sebagai {label}.")
                else:
                    st.dataframe(label_examples[["score", "example_feedback"]], use_container_width=True)

    st.subheader("4. Keyphrase Extraction & Topic Modelling")
    topic_tabs = st.tabs(["Keyphrase per Label", "Topic Modelling Global", "Distribusi Topik per Dokumen"])
    with topic_tabs[0]:
        st.write("Keyphrase diekstrak menggunakan rerata TF-IDF unigram/bigram untuk setiap label prediksi.")
        st.dataframe(topic_summary, use_container_width=True)
        st.download_button(
            label="⬇️ Download keyphrase per label CSV",
            data=dataframe_to_csv_bytes(topic_summary),
            file_name="peer_feedback_keyphrase_per_label.csv",
            mime="text/csv",
        )
    with topic_tabs[1]:
        st.write(
            "Topic modelling global menggunakan NMF pada TF-IDF untuk menemukan tema umum di seluruh komentar yang diupload. "
            "Kolom dominant_predicted_labels membantu membaca label apa yang paling terkait dengan setiap topik."
        )
        st.dataframe(topic_model_summary, use_container_width=True)
        if not topic_model_summary.empty and "document_count" in topic_model_summary.columns:
            chart_df = topic_model_summary[topic_model_summary["topic_id"] != "Error"].set_index("topic_id")["document_count"]
            if not chart_df.empty:
                st.bar_chart(chart_df)
        st.download_button(
            label="⬇️ Download topic modelling summary CSV",
            data=dataframe_to_csv_bytes(topic_model_summary),
            file_name="peer_feedback_topic_modelling_summary.csv",
            mime="text/csv",
        )
    with topic_tabs[2]:
        st.dataframe(topic_doc_assignments, use_container_width=True)
        st.download_button(
            label="⬇️ Download topic assignment CSV",
            data=dataframe_to_csv_bytes(topic_doc_assignments),
            file_name="peer_feedback_topic_assignments.csv",
            mime="text/csv",
        )

    st.subheader("5. Distribusi Label dan Kombinasi")
    left, right = st.columns([1, 1])
    with left:
        st.write("**Jumlah komentar per label**")
        st.dataframe(label_summary, use_container_width=True)
        st.bar_chart(label_summary.set_index("label")["count"])
    with right:
        st.write("**Kombinasi label**")
        st.dataframe(combo_summary, use_container_width=True)
        st.bar_chart(combo_summary.set_index("label_combination")["count"])

    st.subheader("6. Co-occurrence Antar Label")
    st.write("Jumlah komentar yang mendapatkan dua label sekaligus, misalnya Problem + Appreciation atau Problem + Suggestion.")
    st.dataframe(pair_summary, use_container_width=True)
    st.bar_chart(pair_summary.set_index("pair")["count"])

    st.subheader("7. Hasil Klasifikasi")
    st.dataframe(result_df, use_container_width=True)

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button(
            label="⬇️ Download hasil klasifikasi CSV",
            data=dataframe_to_csv_bytes(result_df),
            file_name="peer_feedback_classification_results.csv",
            mime="text/csv",
        )
    with dl_col2:
        st.download_button(
            label="⬇️ Download ringkasan topik CSV",
            data=dataframe_to_csv_bytes(topic_summary),
            file_name="peer_feedback_label_topic_summary.csv",
            mime="text/csv",
        )
    with dl_col3:
        st.download_button(
            label="⬇️ Download contoh representatif CSV",
            data=dataframe_to_csv_bytes(representative_examples),
            file_name="peer_feedback_representative_examples.csv",
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
