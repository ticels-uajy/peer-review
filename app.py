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
import zipfile
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
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, normalize
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
DEFAULT_ARTIFACT_DIR = "models"

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "pada", "dalam",
    "adalah", "atau", "juga", "karena", "sebagai", "lebih", "agar", "akan", "sudah", "belum",
    "tidak", "bisa", "dapat", "sangat", "masih", "ada", "jadi", "tersebut", "nya", "pun", "para",
    "kami", "kita", "saya", "aku", "mereka", "dia", "ia", "anda", "kamu", "secara", "bahwa",
    "the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "for", "with", "on", "this",
}

LABEL_MEANING = {
    "Appreciation": "aspek yang dipersepsi positif, jelas, menarik, atau sudah baik oleh mahasiswa",
    "Problem": "aspek yang dianggap bermasalah, kurang jelas, kurang lengkap, salah, atau perlu diperbaiki",
    "Suggestion": "arah perbaikan, rekomendasi tindakan, atau masukan konkret dari mahasiswa",
    "Neutral": "komentar yang cenderung umum, deskriptif, atau belum memberikan informasi evaluatif yang kuat",
}

LABEL_ACTION = {
    "Appreciation": "Pertahankan aspek yang diapresiasi, tetapi dorong mahasiswa menjelaskan alasan apresiasi secara lebih spesifik.",
    "Problem": "Gunakan tema problem sebagai dasar klarifikasi materi, revisi instruksi, atau diskusi kelas mengenai kesalahan umum.",
    "Suggestion": "Jadikan saran mahasiswa sebagai daftar aksi perbaikan dan contoh feedback konstruktif untuk aktivitas berikutnya.",
    "Neutral": "Berikan rubrik, contoh komentar, atau sentence starters agar komentar lebih evaluatif dan bermanfaat.",
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


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def markdown_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def generate_wordcloud_image(texts: List[str]) -> Optional[Image.Image]:
    text = " ".join([str(t) for t in texts if str(t).strip()])
    if not text.strip():
        return None
    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=STOPWORDS_ID,
        collocations=False,
        collocation_threshold = 3
        max_words=120,
    ).generate(text)
    return wc.to_image()


def image_to_png_bytes(image: Optional[Image.Image]) -> Optional[bytes]:
    if image is None:
        return None
    bio = BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


def phrase_len(phrase: str) -> int:
    return len(str(phrase).split())


def extract_keyphrases(texts: List[str], top_n: int = 15, prefer_multiword: bool = True) -> pd.DataFrame:
    clean_texts = [preprocess_text(t) for t in texts if str(t).strip()]
    if not clean_texts:
        return pd.DataFrame(columns=["rank", "keyphrase", "ngram", "tfidf_score"])

    vectorizer = TfidfVectorizer(
        stop_words=list(STOPWORDS_ID),
        ngram_range=(1, 3),
        min_df=1,
        max_features=5000,
        token_pattern=r"(?u)\b\w[\w\-]+\b",
    )
    try:
        X = vectorizer.fit_transform(clean_texts)
    except ValueError:
        return pd.DataFrame(columns=["rank", "keyphrase", "ngram", "tfidf_score"])

    terms = np.array(vectorizer.get_feature_names_out())
    scores = np.asarray(X.mean(axis=0)).ravel()
    df = pd.DataFrame({"keyphrase": terms, "tfidf_score": scores})
    df["ngram"] = df["keyphrase"].apply(phrase_len)
    df = df[df["tfidf_score"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["rank", "keyphrase", "ngram", "tfidf_score"])

    # Prioritize meaningful multi-word phrases, but keep unigram fallback when data is sparse.
    if prefer_multiword and (df["ngram"] >= 2).sum() >= max(3, min(top_n, 8)):
        multi = df[df["ngram"] >= 2].copy()
        multi["ranking_score"] = multi["tfidf_score"] * (1.0 + 0.15 * multi["ngram"])
        selected = multi.sort_values(["ranking_score", "tfidf_score"], ascending=False).head(top_n)
    else:
        df["ranking_score"] = df["tfidf_score"] * (1.0 + 0.10 * df["ngram"])
        selected = df.sort_values(["ranking_score", "tfidf_score"], ascending=False).head(top_n)

    selected = selected.drop(columns=["ranking_score"], errors="ignore").reset_index(drop=True)
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    selected["tfidf_score"] = selected["tfidf_score"].round(6)
    return selected[["rank", "keyphrase", "ngram", "tfidf_score"]]


def run_nmf_topic_modeling(texts: List[str], n_topics: int = 3, top_n_terms: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean_texts = [preprocess_text(t) for t in texts]
    indexed_texts = [(i, t) for i, t in enumerate(clean_texts) if t.strip()]
    if len(indexed_texts) < 2:
        return (
            pd.DataFrame(columns=["topic_id", "top_terms", "top_terms_list"]),
            pd.DataFrame(columns=["doc_index_within_subset", "topic_id", "topic_weight"]),
        )

    valid_indices = [i for i, _ in indexed_texts]
    valid_texts = [t for _, t in indexed_texts]
    vectorizer = TfidfVectorizer(
        stop_words=list(STOPWORDS_ID),
        ngram_range=(1, 3),
        min_df=1,
        max_features=3000,
        token_pattern=r"(?u)\b\w[\w\-]+\b",
    )
    try:
        X = vectorizer.fit_transform(valid_texts)
    except ValueError:
        return (
            pd.DataFrame(columns=["topic_id", "top_terms", "top_terms_list"]),
            pd.DataFrame(columns=["doc_index_within_subset", "topic_id", "topic_weight"]),
        )

    if X.shape[1] < 2:
        return (
            pd.DataFrame(columns=["topic_id", "top_terms", "top_terms_list"]),
            pd.DataFrame(columns=["doc_index_within_subset", "topic_id", "topic_weight"]),
        )

    k = int(max(1, min(n_topics, len(valid_texts), X.shape[1])))
    try:
        nmf = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=500)
        W = nmf.fit_transform(X)
    except Exception:
        nmf = NMF(n_components=k, random_state=42, init="random", max_iter=500)
        W = nmf.fit_transform(X)

    terms = np.array(vectorizer.get_feature_names_out())
    topic_rows = []
    for topic_id, component in enumerate(nmf.components_, start=1):
        top_idx = component.argsort()[::-1][:top_n_terms]
        top_terms = terms[top_idx].tolist()
        topic_rows.append({
            "topic_id": topic_id,
            "top_terms": ", ".join(top_terms),
            "top_terms_list": top_terms,
        })

    assignment_rows = []
    for local_idx, doc_idx in enumerate(valid_indices):
        topic_idx = int(np.argmax(W[local_idx])) if W.shape[1] else 0
        weight = float(W[local_idx, topic_idx]) if W.shape[1] else 0.0
        assignment_rows.append({
            "doc_index_within_subset": doc_idx,
            "topic_id": topic_idx + 1,
            "topic_weight": round(weight, 6),
        })

    return pd.DataFrame(topic_rows), pd.DataFrame(assignment_rows)


def get_representative_comments(result_df: pd.DataFrame, label: str, text_col: str, top_n: int = 5) -> pd.DataFrame:
    subset = result_df[result_df[f"pred_{label}"] == 1].copy()
    if subset.empty:
        return pd.DataFrame(columns=["rank", "score", "comment"])
    score_col = f"score_{label}"
    subset = subset.sort_values(score_col, ascending=False).head(top_n)
    out = pd.DataFrame({
        "rank": np.arange(1, len(subset) + 1),
        "score": subset[score_col].round(4).values,
        "comment": subset[text_col].astype(str).values,
    })
    return out


@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str):
    from transformers import pipeline
    return pipeline("summarization", model=model_name, tokenizer=model_name)


def topic_guided_abstractive_summary(
    label: str,
    n_label: int,
    total: int,
    keyphrases_df: pd.DataFrame,
    combo_summary: pd.DataFrame,
    representative_df: pd.DataFrame,
) -> str:
    pct = 100 * n_label / max(total, 1)
    phrases = keyphrases_df["keyphrase"].head(6).tolist() if not keyphrases_df.empty else []
    phrase_text = ", ".join(phrases) if phrases else "belum ada tema dominan yang cukup stabil"
    meaning = LABEL_MEANING.get(label, "pola feedback tertentu")
    action = LABEL_ACTION.get(label, "Gunakan hasil ini sebagai dasar tindak lanjut pembelajaran.")

    extra = ""
    if label == "Problem":
        ps_count = int(combo_summary.loc[combo_summary["label_combination"].str.contains("Problem", regex=False) & combo_summary["label_combination"].str.contains("Suggestion", regex=False), "count"].sum())
        extra = f" Dari komentar berlabel Problem, {ps_count} komentar juga memuat Suggestion sehingga dapat diprioritaskan sebagai feedback yang paling konstruktif."
    elif label == "Suggestion":
        extra = " Saran yang muncul dapat dipakai untuk menyusun daftar perbaikan konkret atau bahan diskusi reflektif setelah peer review."
    elif label == "Neutral":
        extra = " Jika proporsinya tinggi, pengajar sebaiknya memperjelas ekspektasi kualitas komentar dan memberi contoh feedback yang lebih evaluatif."
    elif label == "Appreciation":
        extra = " Apresiasi yang dominan menunjukkan dukungan sosial, tetapi perlu diarahkan agar pujian disertai alasan yang spesifik."

    return (
        f"Pada label {label}, terdapat {n_label} dari {total} komentar ({pct:.2f}%). "
        f"Komentar pada kelompok ini terutama berkaitan dengan {phrase_text}. "
        f"Secara pedagogis, label ini merepresentasikan {meaning}. {extra} {action}"
    )


def transformer_abstractive_summary(texts: List[str], model_name: str) -> Optional[str]:
    joined = " ".join([str(t) for t in texts if str(t).strip()])
    if len(joined.split()) < 25:
        return None
    try:
        summarizer = load_summarizer(model_name)
        truncated = " ".join(joined.split()[:900])
        result = summarizer(truncated, max_length=140, min_length=35, do_sample=False)
        if result and isinstance(result, list):
            return str(result[0].get("summary_text", "")).strip()
    except Exception:
        return None
    return None


def generate_readable_learning_insights(
    n_rows: int,
    labels: List[str],
    label_summary: pd.DataFrame,
    combo_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    keyphrases_by_label: Dict[str, pd.DataFrame],
) -> str:
    top_label = label_summary.iloc[0]
    top_combo = combo_summary.iloc[0]
    lines = [
        "## Ringkasan Learning Insights",
        "",
        "### Ringkasan Eksekutif",
        f"Sebanyak **{n_rows} peer feedback** berhasil diklasifikasikan dengan pendekatan multi-label. Karena satu komentar dapat memiliki lebih dari satu label, total persentase antarlabel dapat melebihi 100%.",
        f"Kategori paling dominan adalah **{top_label['label']}** dengan **{int(top_label['count'])} komentar ({top_label['percentage']}%)**.",
        f"Kombinasi label paling sering muncul adalah **{top_combo['label_combination']}** dengan **{int(top_combo['count'])} komentar ({top_combo['percentage']}%)**.",
        "",
        "### Profil Feedback per Label",
    ]

    for label in labels:
        row = label_summary[label_summary["label"] == label].iloc[0]
        kp = keyphrases_by_label.get(label, pd.DataFrame())
        phrases = kp["keyphrase"].head(5).tolist() if not kp.empty else []
        phrase_text = ", ".join(phrases) if phrases else "belum ada frasa dominan"
        lines.extend([
            f"**{label}** — {int(row['count'])} komentar ({row['percentage']}%).",
            f"Tema/kata kunci utama: {phrase_text}.",
            f"Makna: {LABEL_MEANING.get(label, 'pola feedback tertentu')}.",
            f"Tindak lanjut: {LABEL_ACTION.get(label, 'Gunakan hasil ini sebagai dasar tindak lanjut pembelajaran.')}",
            "",
        ])

    lines.append("### Pola Kombinasi Label yang Perlu Diperhatikan")
    for _, row in pair_summary.head(6).iterrows():
        if int(row["count"]) > 0:
            lines.append(f"- **{row['pair']}** muncul pada **{int(row['count'])} komentar**.")
    if not any(pair_summary["count"] > 0):
        lines.append("- Tidak ditemukan kombinasi dua label yang menonjol pada data ini.")

    lines.extend([
        "",
        "### Implikasi untuk Pengajar",
        "Hasil ini dapat digunakan untuk memantau kualitas peer review pada level kelas. Komentar berlabel **Problem + Suggestion** dapat diprioritaskan sebagai contoh feedback konstruktif. Jika **Appreciation** dominan tetapi **Suggestion** rendah, mahasiswa perlu diarahkan agar pujian dilengkapi alasan dan rekomendasi. Jika **Neutral** cukup tinggi, pengajar dapat menyediakan rubrik, contoh komentar, atau sentence starters agar peer feedback lebih spesifik, evaluatif, dan dapat ditindaklanjuti.",
    ])
    return "\n".join(lines)


def safe_filename(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", label).strip("_")


def create_analysis_zip(
    result_df: pd.DataFrame,
    label_summary: pd.DataFrame,
    combo_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    insight_markdown: str,
    per_label_summary_rows: pd.DataFrame,
    keyphrases_by_label: Dict[str, pd.DataFrame],
    topics_by_label: Dict[str, pd.DataFrame],
    assignments_by_label: Dict[str, pd.DataFrame],
    wordclouds_by_label: Dict[str, Optional[bytes]],
    global_topics_df: pd.DataFrame,
    global_assignments_df: pd.DataFrame,
    metadata: Dict,
) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("01_classification/multilabel_classification_results.csv", dataframe_to_csv_bytes(result_df))
        zf.writestr("02_summaries/label_summary.csv", dataframe_to_csv_bytes(label_summary))
        zf.writestr("02_summaries/label_combination_summary.csv", dataframe_to_csv_bytes(combo_summary))
        zf.writestr("02_summaries/label_pair_cooccurrence.csv", dataframe_to_csv_bytes(pair_summary))
        zf.writestr("02_summaries/learning_insights.md", markdown_bytes(insight_markdown))
        zf.writestr("02_summaries/per_label_abstractive_summaries.csv", dataframe_to_csv_bytes(per_label_summary_rows))
        zf.writestr("metadata/model_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"))

        all_kp = []
        all_topics = []
        all_assignments = []
        for label, df_kp in keyphrases_by_label.items():
            name = safe_filename(label)
            kp = df_kp.copy()
            if not kp.empty:
                kp.insert(0, "label", label)
                all_kp.append(kp)
            zf.writestr(f"03_per_label_keyphrases/{name}_keyphrases.csv", dataframe_to_csv_bytes(df_kp))

            topics = topics_by_label.get(label, pd.DataFrame()).copy()
            if not topics.empty:
                topics.insert(0, "label", label)
                all_topics.append(topics)
            zf.writestr(f"04_per_label_topics/{name}_nmf_topics.csv", dataframe_to_csv_bytes(topics_by_label.get(label, pd.DataFrame())))

            assignments = assignments_by_label.get(label, pd.DataFrame()).copy()
            if not assignments.empty:
                assignments.insert(0, "label", label)
                all_assignments.append(assignments)
            zf.writestr(f"05_per_label_topic_assignments/{name}_topic_assignments.csv", dataframe_to_csv_bytes(assignments_by_label.get(label, pd.DataFrame())))

            png = wordclouds_by_label.get(label)
            if png is not None:
                zf.writestr(f"06_wordclouds/{name}_wordcloud.png", png)

        zf.writestr(
            "03_per_label_keyphrases/ALL_LABELS_keyphrases.csv",
            dataframe_to_csv_bytes(pd.concat(all_kp, ignore_index=True) if all_kp else pd.DataFrame()),
        )
        zf.writestr(
            "04_per_label_topics/ALL_LABELS_nmf_topics.csv",
            dataframe_to_csv_bytes(pd.concat(all_topics, ignore_index=True) if all_topics else pd.DataFrame()),
        )
        zf.writestr(
            "05_per_label_topic_assignments/ALL_LABELS_topic_assignments.csv",
            dataframe_to_csv_bytes(pd.concat(all_assignments, ignore_index=True) if all_assignments else pd.DataFrame()),
        )
        zf.writestr("07_global_topic_modeling/global_nmf_topics.csv", dataframe_to_csv_bytes(global_topics_df))
        zf.writestr("07_global_topic_modeling/global_topic_assignments.csv", dataframe_to_csv_bytes(global_assignments_df))
    return bio.getvalue()


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Peer Feedback Insight Generator",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Peer Feedback Insight Generator")
st.caption("Klasifikasi multi-label peer feedback menjadi Appreciation, Problem, Suggestion, dan Neutral, lalu mengubahnya menjadi learning insights untuk pengajar.")

with st.sidebar:
    st.header("Pengaturan Model")
    artifact_dir = st.text_input("Folder artefak model", value=DEFAULT_ARTIFACT_DIR)
    model_choice = st.radio("Pilih model terbaik", options=["Machine Learning", "Deep Learning"], horizontal=False)

    st.divider()
    st.header("Pengaturan Analisis")
    n_keyphrases = st.slider("Jumlah keyphrase per label", min_value=5, max_value=30, value=15, step=5)
    n_topics = st.slider("Jumlah topik NMF", min_value=2, max_value=8, value=3, step=1)
    n_rep = st.slider("Jumlah contoh komentar representatif", min_value=3, max_value=10, value=5, step=1)
    use_transformer_summary = st.checkbox("Gunakan Transformer untuk abstractive summary jika tersedia", value=False)
    summary_model_name = st.text_input("Model summarization", value="cahya/t5-base-indonesian-summarization-cased")

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

if st.button("🚀 Jalankan Klasifikasi dan Analisis", type="primary"):
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

    # Per-label analysis artifacts
    keyphrases_by_label: Dict[str, pd.DataFrame] = {}
    topics_by_label: Dict[str, pd.DataFrame] = {}
    assignments_by_label: Dict[str, pd.DataFrame] = {}
    wordclouds_by_label: Dict[str, Optional[bytes]] = {}
    summaries_by_label: Dict[str, str] = {}
    representatives_by_label: Dict[str, pd.DataFrame] = {}

    with st.spinner("Membuat keyphrases, topic modelling, wordcloud, dan summary per label..."):
        for label in labels:
            subset = result_df[result_df[f"pred_{label}"] == 1].copy()
            label_texts_clean = subset["text_clean"].tolist()
            label_texts_original = subset[text_col].astype(str).tolist()
            keyphrases_by_label[label] = extract_keyphrases(label_texts_clean, top_n=n_keyphrases)
            topics_df, assignments_df = run_nmf_topic_modeling(label_texts_clean, n_topics=n_topics)
            if not assignments_df.empty:
                subset_indices = subset.index.tolist()
                assignments_df = assignments_df.copy()
                assignments_df["original_row_index"] = assignments_df["doc_index_within_subset"].map(lambda idx: subset_indices[idx] if idx < len(subset_indices) else None)
                assignments_df["comment"] = assignments_df["doc_index_within_subset"].map(lambda idx: label_texts_original[idx] if idx < len(label_texts_original) else "")
            topics_by_label[label] = topics_df
            assignments_by_label[label] = assignments_df
            wordclouds_by_label[label] = image_to_png_bytes(generate_wordcloud_image(label_texts_clean))
            representatives_by_label[label] = get_representative_comments(result_df, label, text_col, top_n=n_rep)

            transformer_summary = None
            if use_transformer_summary and len(label_texts_original) > 0:
                transformer_summary = transformer_abstractive_summary(label_texts_original, summary_model_name)
            summaries_by_label[label] = transformer_summary or topic_guided_abstractive_summary(
                label=label,
                n_label=len(subset),
                total=len(result_df),
                keyphrases_df=keyphrases_by_label[label],
                combo_summary=combo_summary,
                representative_df=representatives_by_label[label],
            )

    per_label_summary_df = pd.DataFrame([
        {
            "label": label,
            "count": int(label_summary.loc[label_summary["label"] == label, "count"].iloc[0]),
            "percentage": float(label_summary.loc[label_summary["label"] == label, "percentage"].iloc[0]),
            "summary": summaries_by_label[label],
        }
        for label in labels
    ])

    insight_markdown = generate_readable_learning_insights(
        len(result_df), labels, label_summary, combo_summary, pair_summary, keyphrases_by_label
    )

    global_topics_df, global_assignments_df = run_nmf_topic_modeling(result_df["text_clean"].tolist(), n_topics=n_topics)
    if not global_assignments_df.empty:
        global_assignments_df = global_assignments_df.copy()
        global_assignments_df["original_row_index"] = global_assignments_df["doc_index_within_subset"]
        global_assignments_df["comment"] = global_assignments_df["doc_index_within_subset"].map(lambda idx: str(result_df.iloc[idx][text_col]) if idx < len(result_df) else "")
        global_assignments_df["predicted_labels"] = global_assignments_df["doc_index_within_subset"].map(lambda idx: str(result_df.iloc[idx]["predicted_labels"]) if idx < len(result_df) else "")

    zip_bytes = create_analysis_zip(
        result_df=result_df,
        label_summary=label_summary,
        combo_summary=combo_summary,
        pair_summary=pair_summary,
        insight_markdown=insight_markdown,
        per_label_summary_rows=per_label_summary_df,
        keyphrases_by_label=keyphrases_by_label,
        topics_by_label=topics_by_label,
        assignments_by_label=assignments_by_label,
        wordclouds_by_label=wordclouds_by_label,
        global_topics_df=global_topics_df,
        global_assignments_df=global_assignments_df,
        metadata=metadata,
    )

    st.success(f"Klasifikasi dan analisis selesai menggunakan model: {model_choice}")

    st.subheader("2. Ringkasan Learning Insights untuk Pengajar")
    metric_cols = st.columns(len(labels))
    for idx, label in enumerate(labels):
        count = int(label_summary.loc[label_summary["label"] == label, "count"].iloc[0])
        pct = float(label_summary.loc[label_summary["label"] == label, "percentage"].iloc[0])
        metric_cols[idx].metric(label, f"{count}", f"{pct}%")
    st.markdown(insight_markdown)

    st.subheader("3. Analisis per Label")
    label_tabs = st.tabs(labels)
    for tab, label in zip(label_tabs, labels):
        with tab:
            count = int(label_summary.loc[label_summary["label"] == label, "count"].iloc[0])
            pct = float(label_summary.loc[label_summary["label"] == label, "percentage"].iloc[0])
            st.markdown(f"### {label}: {count} komentar ({pct:.2f}%)")
            sub_tabs = st.tabs(["Summary", "WordCloud", "Keyphrases/Topik Utama", "Contoh Komentar"])
            with sub_tabs[0]:
                st.markdown(summaries_by_label[label])
                st.download_button(
                    label=f"⬇️ Download summary {label}",
                    data=markdown_bytes(f"# Summary {label}\n\n{summaries_by_label[label]}"),
                    file_name=f"summary_{safe_filename(label)}.md",
                    mime="text/markdown",
                    key=f"download_summary_{label}",
                )
            with sub_tabs[1]:
                if wordclouds_by_label[label] is None:
                    st.info(f"Tidak ada teks untuk wordcloud label {label}.")
                else:
                    st.image(wordclouds_by_label[label], use_container_width=True)
                    st.download_button(
                        label=f"⬇️ Download wordcloud {label}",
                        data=wordclouds_by_label[label],
                        file_name=f"wordcloud_{safe_filename(label)}.png",
                        mime="image/png",
                        key=f"download_wc_{label}",
                    )
            with sub_tabs[2]:
                st.write("Keyphrase menggunakan TF-IDF unigram, bigram, dan trigram dengan prioritas pada frasa multi-kata.")
                st.dataframe(keyphrases_by_label[label], use_container_width=True)
                st.download_button(
                    label=f"⬇️ Download keyphrase {label}",
                    data=dataframe_to_csv_bytes(keyphrases_by_label[label]),
                    file_name=f"keyphrases_{safe_filename(label)}.csv",
                    mime="text/csv",
                    key=f"download_kp_{label}",
                )
            with sub_tabs[3]:
                st.dataframe(representatives_by_label[label], use_container_width=True)
                st.download_button(
                    label=f"⬇️ Download contoh komentar {label}",
                    data=dataframe_to_csv_bytes(representatives_by_label[label]),
                    file_name=f"representative_comments_{safe_filename(label)}.csv",
                    mime="text/csv",
                    key=f"download_rep_{label}",
                )

    st.subheader("4. Keyphrase Extraction & Topic Modelling per Label")
    st.caption("Bagian ini dikelompokkan per label. Setiap label memiliki keyphrase TF-IDF n-gram, NMF topics khusus label tersebut, dan assignment topik untuk setiap dokumen pada label tersebut.")
    topic_label_tabs = st.tabs(labels)
    for tab, label in zip(topic_label_tabs, labels):
        with tab:
            st.markdown(f"### {label}")
            kp_tab, topic_tab, assign_tab = st.tabs(["Keyphrase per Label", "NMF Topics per Label", "Assignment Topik Dokumen"])
            with kp_tab:
                st.dataframe(keyphrases_by_label[label], use_container_width=True)
            with topic_tab:
                topics_df = topics_by_label[label].copy()
                if topics_df.empty:
                    st.info("Jumlah dokumen/fitur belum cukup untuk membentuk topic model pada label ini.")
                else:
                    topics_display = topics_df.drop(columns=["top_terms_list"], errors="ignore")
                    st.dataframe(topics_display, use_container_width=True)
                    st.bar_chart(assignments_by_label[label]["topic_id"].value_counts().sort_index())
            with assign_tab:
                st.dataframe(assignments_by_label[label], use_container_width=True)

    all_keyphrases_df = pd.concat(
        [df.assign(label=label) for label, df in keyphrases_by_label.items() if not df.empty],
        ignore_index=True,
    ) if any(not df.empty for df in keyphrases_by_label.values()) else pd.DataFrame()
    all_topics_df = pd.concat(
        [df.drop(columns=["top_terms_list"], errors="ignore").assign(label=label) for label, df in topics_by_label.items() if not df.empty],
        ignore_index=True,
    ) if any(not df.empty for df in topics_by_label.values()) else pd.DataFrame()
    all_assignments_df = pd.concat(
        [df.assign(label=label) for label, df in assignments_by_label.items() if not df.empty],
        ignore_index=True,
    ) if any(not df.empty for df in assignments_by_label.values()) else pd.DataFrame()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Download semua keyphrase per label", dataframe_to_csv_bytes(all_keyphrases_df), "all_label_keyphrases.csv", "text/csv")
    with c2:
        st.download_button("⬇️ Download semua topik per label", dataframe_to_csv_bytes(all_topics_df), "all_label_nmf_topics.csv", "text/csv")
    with c3:
        st.download_button("⬇️ Download semua assignment topik", dataframe_to_csv_bytes(all_assignments_df), "all_label_topic_assignments.csv", "text/csv")

    st.subheader("5. Topic Modelling Global")
    with st.expander("Lihat topic modelling global sebagai informasi tambahan"):
        st.write("Topic modelling global dihitung dari seluruh komentar yang diupload, tanpa memisahkan label.")
        if global_topics_df.empty:
            st.info("Jumlah dokumen/fitur belum cukup untuk membentuk topic model global.")
        else:
            st.dataframe(global_topics_df.drop(columns=["top_terms_list"], errors="ignore"), use_container_width=True)
            st.bar_chart(global_assignments_df["topic_id"].value_counts().sort_index())
            st.write("**Assignment topik global per dokumen**")
            st.dataframe(global_assignments_df, use_container_width=True)
            gc1, gc2 = st.columns(2)
            with gc1:
                st.download_button("⬇️ Download global topics", dataframe_to_csv_bytes(global_topics_df.drop(columns=["top_terms_list"], errors="ignore")), "global_nmf_topics.csv", "text/csv")
            with gc2:
                st.download_button("⬇️ Download global assignments", dataframe_to_csv_bytes(global_assignments_df), "global_topic_assignments.csv", "text/csv")

    st.subheader("6. Klasifikasi Multi-label untuk Setiap Komentar")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("7. Ringkasan Label dan Kombinasi Label")
    left, right = st.columns([1, 1])
    with left:
        st.write("**Jumlah komentar per label**")
        st.dataframe(label_summary, use_container_width=True)
        st.bar_chart(label_summary.set_index("label")["count"])
    with right:
        st.write("**Kombinasi label**")
        st.dataframe(combo_summary, use_container_width=True)
        st.bar_chart(combo_summary.set_index("label_combination")["count"])

    st.write("**Co-occurrence antar label**")
    st.dataframe(pair_summary, use_container_width=True)

    st.subheader("8. Download Semua Hasil Analisis")
    st.write("Unduh seluruh hasil klasifikasi dan analisis dalam satu file ZIP. ZIP berisi hasil klasifikasi, learning insights, summary per label, wordcloud, keyphrases, NMF topics per label, assignment topik per label, topic modelling global, dan metadata model.")
    st.download_button(
        label="⬇️ Download semua hasil analisis ZIP",
        data=zip_bytes,
        file_name="peer_feedback_complete_analysis.zip",
        mime="application/zip",
        type="primary",
    )

    with st.expander("Download file inti secara terpisah"):
        st.download_button("⬇️ Hasil klasifikasi CSV", dataframe_to_csv_bytes(result_df), "peer_feedback_classification_results.csv", "text/csv")
        st.download_button("⬇️ Ringkasan label CSV", dataframe_to_csv_bytes(label_summary), "peer_feedback_label_summary.csv", "text/csv")
        st.download_button("⬇️ Kombinasi label CSV", dataframe_to_csv_bytes(combo_summary), "peer_feedback_label_combination_summary.csv", "text/csv")
        st.download_button("⬇️ Co-occurrence label CSV", dataframe_to_csv_bytes(pair_summary), "peer_feedback_pair_cooccurrence.csv", "text/csv")
        st.download_button("⬇️ Learning insights Markdown", markdown_bytes(insight_markdown), "learning_insights.md", "text/markdown")
        st.download_button("⬇️ Summary per label CSV", dataframe_to_csv_bytes(per_label_summary_df), "per_label_abstractive_summaries.csv", "text/csv")

    st.subheader("9. Metadata Model")
    st.json(metadata)
