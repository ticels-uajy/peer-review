import io
import json
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import STOPWORDS, WordCloud

# Optional imports for deep learning model.
# The app will still run for ML mode even if TensorFlow/Keras is not installed,
# as long as DL mode is not selected.
try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    import keras as standalone_keras
except Exception:  # pragma: no cover
    standalone_keras = None

try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:  # pragma: no cover
    try:
        from keras.utils import pad_sequences
    except Exception:
        pad_sequences = None


APP_TITLE = "Peer Feedback Multi-label Classification"
DEFAULT_LABELS = ["Problem", "Appreciation", "Suggestion", "Neutral"]
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

# Stopwords used only for wordcloud visualization. Keep this list conservative so
# words that are useful for learning insight, such as "kurang", "baik", "jelas",
# "problem", or "suggestion", are not accidentally removed.
INDONESIAN_WORDCLOUD_STOPWORDS = {
    "ada", "adalah", "agar", "akan", "akhir", "antara", "apa", "apabila",
    "atas", "atau", "bahwa", "bagi", "bagian", "banyak", "baru", "begini",
    "begitu", "belum", "berada", "berikut", "bersama", "berturut", "bila",
    "bisa", "buat", "cara", "cukup", "dalam", "dan", "dapat", "dari",
    "daripada", "dengan", "demi", "demikian", "dengan", "depan", "di", "dia",
    "diri", "dulu", "hal", "hanya", "harus", "hingga", "ia", "ini", "itu",
    "jadi", "jika", "juga", "justru", "kala", "kalau", "kami", "kamu",
    "kan", "karena", "kata", "ke", "kembali", "kemudian", "kepada", "ketika",
    "kita", "lagi", "lain", "lalu", "lewat", "maka", "makin", "malah",
    "mana", "masih", "maupun", "melalui", "memang", "mereka", "meski",
    "misal", "misalnya", "namun", "nanti", "nya", "oleh", "orang", "pada",
    "paling", "para", "per", "perlu", "pun", "saat", "saja", "saling",
    "sama", "sambil", "sampai", "sana", "sangat", "saya", "sebagai",
    "sebelum", "sebuah", "secara", "sedang", "sedangkan", "sedikit",
    "sehingga", "sejak", "sekali", "sekitar", "selain", "selalu", "seluruh",
    "semakin", "semua", "sementara", "sempat", "sendiri", "seolah", "seperti",
    "serta", "setelah", "setiap", "suatu", "sudah", "supaya", "tanpa", "tapi",
    "telah", "tentang", "tentu", "terhadap", "tersebut", "tetapi", "tiap",
    "tidak", "tujuan", "untuk", "usah", "yaitu", "yakni", "yang",
}

DOMAIN_WORDCLOUD_STOPWORDS = {
    "feedback", "peer", "review", "comment", "comments", "student", "students",
    "teks", "text", "label", "labels",
}

DEFAULT_WORDCLOUD_STOPWORDS = (
    set(STOPWORDS)
    | INDONESIAN_WORDCLOUD_STOPWORDS
    | DOMAIN_WORDCLOUD_STOPWORDS
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Session-state helpers
# -----------------------------------------------------------------------------
RESULT_STATE_KEYS = [
    "classification_result_df",
    "label_count_df",
    "combination_df",
    "summary",
    "saved_paths",
    "active_labels",
    "active_text_col",
    "active_model_choice",
]


def clear_classification_results() -> None:
    for key in RESULT_STATE_KEYS:
        st.session_state.pop(key, None)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_custom_stopwords(raw_text: str) -> Set[str]:
    raw_text = normalize_text(raw_text).lower()
    if not raw_text:
        return set()
    return {
        item.strip()
        for item in re.split(r"[\s,;]+", raw_text)
        if item.strip()
    }


def clean_for_wordcloud(text: str, stopwords: Optional[Set[str]] = None) -> str:
    stopwords = {word.lower() for word in (stopwords or set())}
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    words = []
    for word in text.split():
        if word in stopwords:
            continue
        if word.isdigit():
            continue
        if len(word) <= 1:
            continue
        words.append(word)
    return " ".join(words)


def slugify(value: str, default: str = "output") -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or default


def load_labels(path: Path = MODEL_DIR / "labels.json") -> List[str]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, dict) and "labels" in labels:
            labels = labels["labels"]
        return list(labels)
    return DEFAULT_LABELS


def safe_columns(df: pd.DataFrame) -> List[str]:
    return [str(c) for c in df.columns]


def get_probability_columns(labels: List[str]) -> List[str]:
    return [f"prob_{label}" for label in labels]


def get_prediction_columns(labels: List[str]) -> List[str]:
    return [f"pred_{label}" for label in labels]


def predictions_to_label_string(predictions: np.ndarray, labels: List[str]) -> List[str]:
    results = []
    for row in predictions:
        active = [labels[i] for i, value in enumerate(row) if int(value) == 1]
        results.append(", ".join(active) if active else "Unclassified")
    return results


def enforce_neutral_rule(predictions: np.ndarray, labels: List[str]) -> np.ndarray:
    if "Neutral" not in labels:
        return predictions

    predictions = predictions.copy().astype(int)
    neutral_idx = labels.index("Neutral")
    non_neutral_indices = [i for i, label in enumerate(labels) if label != "Neutral"]

    non_neutral_sum = predictions[:, non_neutral_indices].sum(axis=1)
    predictions[non_neutral_sum > 0, neutral_idx] = 0
    predictions[non_neutral_sum == 0, neutral_idx] = 1
    return predictions


def probabilities_to_predictions(
    probabilities: np.ndarray,
    labels: List[str],
    threshold: float,
    neutral_rule: bool = True,
) -> np.ndarray:
    predictions = (probabilities >= threshold).astype(int)
    if neutral_rule:
        predictions = enforce_neutral_rule(predictions, labels)
    return predictions


def infer_probabilities_from_predict_output(y_pred, labels: List[str]) -> np.ndarray:
    # Some sklearn multi-output predict_proba returns list of arrays: one array per label.
    if isinstance(y_pred, list):
        probs = []
        for arr in y_pred:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                probs.append(arr[:, 1])
            else:
                probs.append(arr.reshape(-1))
        return np.vstack(probs).T

    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        # Single-label fallback; not ideal for multi-label, but keeps app safe.
        out = np.zeros((len(y_pred), len(labels)), dtype=float)
        for i, value in enumerate(y_pred):
            if isinstance(value, str) and value in labels:
                out[i, labels.index(value)] = 1.0
            else:
                try:
                    idx = int(value)
                    if 0 <= idx < len(labels):
                        out[i, idx] = 1.0
                except Exception:
                    pass
        return out

    if y_pred.ndim == 2:
        if y_pred.shape[1] == len(labels):
            return y_pred.astype(float)
        raise ValueError(
            f"Model output has {y_pred.shape[1]} columns, but labels.json has {len(labels)} labels."
        )

    raise ValueError("Unsupported prediction/probability output format.")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ml_model(model_path: str):
    return joblib.load(model_path)


def _version_of(module) -> str:
    return getattr(module, "__version__", "unknown") if module is not None else "not installed"


def _major_version(version: str) -> Optional[int]:
    try:
        return int(str(version).split(".")[0])
    except Exception:
        return None


def _is_keras3_serialized_model(model_path: str) -> bool:
    path = Path(model_path)
    if path.suffix.lower() != ".keras" or not path.exists():
        return False
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if "config.json" not in names:
                return False
            config_text = zf.read("config.json").decode("utf-8", errors="ignore")
        return "keras.src" in config_text or "DTypePolicy" in config_text
    except Exception:
        return False


def _ensure_compatible_dl_runtime(model_path: str) -> None:
    keras_version = _version_of(standalone_keras)
    keras_major = _major_version(keras_version)
    if _is_keras3_serialized_model(model_path) and (keras_major is None or keras_major < 3):
        raise RuntimeError(
            "Model DL ini tampaknya disimpan dengan format Keras 3 `.keras`, "
            "tetapi environment deploy masih memakai Keras/TensorFlow lama.\n\n"
            f"Versi terdeteksi: tensorflow={_version_of(tf)}, keras={keras_version}.\n\n"
            "Perbaikan yang perlu dilakukan:\n"
            "1. Pastikan file `requirements.txt` di GitHub berisi `tensorflow==2.16.2` dan `keras==3.4.1`.\n"
            "2. Deploy dengan Python 3.11 di Streamlit Community Cloud.\n"
            "3. Jika app lama masih memakai TensorFlow/Keras 2.15, delete app lalu deploy ulang, jangan hanya reboot.\n"
            "4. Setelah deploy, cek kembali log. Seharusnya tidak lagi tertulis `tensorflow=2.15.1` dan `keras=2.15.0`."
        )


def _load_keras_model_robust(model_path: str):
    _ensure_compatible_dl_runtime(model_path)
    loaders = []
    if standalone_keras is not None:
        loaders.append(("keras.models.load_model", standalone_keras.models.load_model))
    if tf is not None:
        loaders.append(("tf.keras.models.load_model", tf.keras.models.load_model))

    if not loaders:
        raise ImportError(
            "TensorFlow/Keras is not installed. Install TensorFlow or Keras to use DL mode."
        )

    attempts = []
    kwargs_candidates = [
        {"compile": False, "safe_mode": False},
        {"compile": False},
        {},
    ]

    for loader_name, loader in loaders:
        for kwargs in kwargs_candidates:
            try:
                return loader(model_path, **kwargs)
            except TypeError as exc:
                # Some older loaders do not accept safe_mode. Try the next kwargs.
                attempts.append(f"{loader_name}{kwargs}: {type(exc).__name__}: {exc}")
            except Exception as exc:
                attempts.append(f"{loader_name}{kwargs}: {type(exc).__name__}: {exc}")

    detail = "\n".join(f"- {item}" for item in attempts[-6:])
    raise RuntimeError(
        "Model DL gagal dimuat. Kemungkinan besar file model disimpan dengan "
        "versi Keras/TensorFlow yang berbeda dari environment deploy.\n\n"
        f"Versi terdeteksi: tensorflow={_version_of(tf)}, keras={_version_of(standalone_keras)}.\n\n"
        "Yang bisa dilakukan:\n"
        "1. Gunakan requirements.txt versi Keras 3/TensorFlow 2.16+ yang disertakan pada ZIP ini.\n"
        "2. Deploy dengan Python 3.11.\n"
        "3. Jika masih gagal, simpan ulang model dari environment training dengan `model.save('best_dl_model.keras')` "
        "lalu gunakan versi TensorFlow/Keras yang sama saat deploy.\n\n"
        "Ringkasan percobaan loader terakhir:\n"
        f"{detail}"
    )


@st.cache_resource(show_spinner=False)
def load_dl_assets(model_path: str, tokenizer_path: str):
    if pad_sequences is None:
        raise ImportError(
            "pad_sequences tidak tersedia. Pastikan TensorFlow/Keras terinstall dengan benar."
        )
    model = _load_keras_model_robust(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer


def _has_predict_interface(obj) -> bool:
    return any(
        hasattr(obj, attr)
        for attr in ("predict_proba", "decision_function", "predict")
    )


def _get_first_available(mapping: Dict, keys: List[str]):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def unpack_ml_model_artifact(artifact):
    if not isinstance(artifact, dict):
        return artifact, None

    vectorizer_keys = [
        "vectorizer",
        "tfidf",
        "tfidf_vectorizer",
        "count_vectorizer",
        "cv",
        "vect",
    ]
    model_keys = [
        "model",
        "classifier",
        "clf",
        "estimator",
        "pipeline",
        "best_model",
        "final_model",
        "ml_model",
        "sklearn_model",
        "text_clf",
    ]

    vectorizer = _get_first_available(artifact, vectorizer_keys)
    classifier = _get_first_available(artifact, model_keys)

    # Fallback: if the dict contains exactly one sklearn-like object, use it.
    if classifier is None:
        predict_like_items = [
            (key, value)
            for key, value in artifact.items()
            if _has_predict_interface(value)
        ]
        if len(predict_like_items) == 1:
            classifier = predict_like_items[0][1]

    # Fallback: prefer a sklearn Pipeline-like object if present among many objects.
    if classifier is None:
        for _, value in artifact.items():
            if _has_predict_interface(value) and hasattr(value, "steps"):
                classifier = value
                break

    if classifier is None:
        available_keys = ", ".join(map(str, artifact.keys()))
        raise ValueError(
            "ML model artifact is a dictionary, but no sklearn-compatible model was found. "
            "Please save the model as a full sklearn Pipeline, or use one of these keys: "
            f"{', '.join(model_keys)}. Available keys in your file: {available_keys}"
        )

    return classifier, vectorizer


def predict_with_ml(model, texts: List[str], labels: List[str]) -> np.ndarray:
    model, vectorizer = unpack_ml_model_artifact(model)
    x = texts

    if vectorizer is not None:
        x = vectorizer.transform(texts)

    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x)
        probabilities = infer_probabilities_from_predict_output(raw, labels)
    elif hasattr(model, "decision_function"):
        scores = infer_probabilities_from_predict_output(model.decision_function(x), labels)
        probabilities = 1 / (1 + np.exp(-scores))
    elif hasattr(model, "predict"):
        raw = model.predict(x)
        probabilities = infer_probabilities_from_predict_output(raw, labels)
    else:
        raise ValueError(
            "Unsupported ML model. Provide a sklearn-compatible model, pipeline, "
            "or dictionary containing one."
        )

    return np.clip(probabilities, 0, 1)


def predict_with_dl(
    model,
    tokenizer,
    texts: List[str],
    labels: List[str],
    max_len: int,
    batch_size: int,
) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    probabilities = model.predict(padded, batch_size=batch_size, verbose=0)
    probabilities = infer_probabilities_from_predict_output(probabilities, labels)
    return np.clip(probabilities, 0, 1)


# -----------------------------------------------------------------------------
# Insight, output saving, and visualization
# -----------------------------------------------------------------------------
def build_insight_tables(df_pred: pd.DataFrame, labels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_cols = get_prediction_columns(labels)

    label_counts = []
    total = len(df_pred)
    for label, col in zip(labels, pred_cols):
        count = int(df_pred[col].sum())
        label_counts.append(
            {
                "Label": label,
                "Count": count,
                "Percentage": round((count / total * 100) if total else 0, 2),
            }
        )
    label_count_df = pd.DataFrame(label_counts)

    combination_counter = Counter(df_pred["predicted_labels"].fillna("Unclassified"))
    combination_df = pd.DataFrame(
        [
            {
                "Label combination": combo,
                "Count": count,
                "Percentage": round((count / total * 100) if total else 0, 2),
            }
            for combo, count in combination_counter.most_common()
        ]
    )

    return label_count_df, combination_df


def generate_summary(label_count_df: pd.DataFrame, combination_df: pd.DataFrame, total: int) -> str:
    if total == 0 or label_count_df.empty:
        return "Belum ada data yang dapat dirangkum."

    top_label = label_count_df.sort_values("Count", ascending=False).iloc[0]
    low_label = label_count_df.sort_values("Count", ascending=True).iloc[0]
    top_combo = combination_df.iloc[0] if not combination_df.empty else None

    problem_count = int(label_count_df.loc[label_count_df["Label"] == "Problem", "Count"].sum())
    appreciation_count = int(label_count_df.loc[label_count_df["Label"] == "Appreciation", "Count"].sum())
    suggestion_count = int(label_count_df.loc[label_count_df["Label"] == "Suggestion", "Count"].sum())
    neutral_count = int(label_count_df.loc[label_count_df["Label"] == "Neutral", "Count"].sum())

    sentences = [
        f"Dari {total} peer feedback yang dianalisis, label yang paling dominan adalah {top_label['Label']} sebanyak {int(top_label['Count'])} data ({top_label['Percentage']}%).",
        f"Label dengan kemunculan paling rendah adalah {low_label['Label']} sebanyak {int(low_label['Count'])} data ({low_label['Percentage']}%).",
    ]

    if top_combo is not None:
        sentences.append(
            f"Kombinasi label yang paling sering muncul adalah {top_combo['Label combination']} sebanyak {int(top_combo['Count'])} data ({top_combo['Percentage']}%)."
        )

    if problem_count > suggestion_count:
        sentences.append(
            "Jumlah feedback berkategori Problem lebih tinggi daripada Suggestion, sehingga pengajar dapat mempertimbangkan tindak lanjut berupa klarifikasi materi, perbaikan instruksi tugas, atau pemberian contoh tambahan."
        )
    elif suggestion_count > problem_count:
        sentences.append(
            "Jumlah Suggestion relatif menonjol, yang menunjukkan bahwa mahasiswa tidak hanya mengidentifikasi isu, tetapi juga memberikan masukan perbaikan yang dapat dimanfaatkan untuk peningkatan pembelajaran."
        )

    if appreciation_count > 0:
        sentences.append(
            "Kemunculan Appreciation menunjukkan adanya aspek pembelajaran atau kinerja teman sebaya yang dipersepsi positif oleh mahasiswa."
        )

    if neutral_count == total:
        sentences.append(
            "Seluruh feedback terklasifikasi sebagai Neutral, sehingga data mungkin berisi komentar umum yang kurang memuat evaluasi spesifik."
        )

    return " ".join(sentences)


def save_outputs_to_folder(
    result_df: pd.DataFrame,
    label_count_df: pd.DataFrame,
    combination_df: pd.DataFrame,
    summary: str,
    metadata: Dict,
    output_base_dir: Path = OUTPUT_DIR,
) -> Dict[str, Path]:
    output_base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(metadata.get("model_choice", "model"))
    upload_slug = slugify(Path(metadata.get("uploaded_filename", "uploaded_data")).stem)
    run_dir = output_base_dir / f"run_{timestamp}_{model_slug}_{upload_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "run_dir": run_dir,
        "classification_csv": run_dir / "classification_results.csv",
        "label_counts_csv": run_dir / "label_counts.csv",
        "label_combinations_csv": run_dir / "label_combinations.csv",
        "summary_txt": run_dir / "summary_learning_insight.txt",
        "insights_json": run_dir / "learning_insights.json",
        "metadata_json": run_dir / "metadata.json",
    }

    result_df.to_csv(paths["classification_csv"], index=False, encoding="utf-8-sig")
    label_count_df.to_csv(paths["label_counts_csv"], index=False, encoding="utf-8-sig")
    combination_df.to_csv(paths["label_combinations_csv"], index=False, encoding="utf-8-sig")

    with open(paths["summary_txt"], "w", encoding="utf-8") as f:
        f.write(summary)

    learning_insights = {
        "summary": summary,
        "label_counts": label_count_df.to_dict(orient="records"),
        "label_combinations": combination_df.to_dict(orient="records"),
    }
    with open(paths["insights_json"], "w", encoding="utf-8") as f:
        json.dump(learning_insights, f, ensure_ascii=False, indent=2)

    with open(paths["metadata_json"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return paths


def zip_run_folder(run_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                zip_file.write(path, arcname=path.relative_to(run_dir))
    buffer.seek(0)
    return buffer.getvalue()


def render_saved_output_section(paths: Dict[str, Path]):
    run_dir = paths.get("run_dir")
    if not run_dir:
        return

    st.subheader("7. Penyimpanan output")
    st.success(f"Hasil klasifikasi dan learning insights sudah disimpan ke folder `{run_dir.as_posix()}`.")

    output_rows = []
    for key, path in paths.items():
        if key == "run_dir":
            continue
        output_rows.append({"Jenis output": key, "Path": path.as_posix()})
    st.dataframe(pd.DataFrame(output_rows), use_container_width=True)

    zip_bytes = zip_run_folder(run_dir)
    st.download_button(
        "⬇️ Download semua output sebagai ZIP",
        data=zip_bytes,
        file_name=f"{run_dir.name}.zip",
        mime="application/zip",
    )


def render_wordcloud(text: str, title: str, stopwords: Optional[Set[str]] = None):
    stopwords = stopwords or set()
    text = clean_for_wordcloud(text, stopwords=stopwords)
    if not text:
        st.info(
            f"Tidak ada teks yang cukup untuk membuat wordcloud label {title} "
            "setelah stopwords dibuang."
        )
        return

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        collocations=False,
        max_words=150,
        stopwords=stopwords,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🧠 Peer Feedback Multi-label Classification")
st.caption(
    "Aplikasi untuk mengklasifikasikan peer feedback/peer comments ke label Problem, Appreciation, Suggestion, dan Neutral."
)

with st.sidebar:
    st.header("⚙️ Pengaturan")

    labels = load_labels()
    st.write("**Label aktif:**", ", ".join(labels))

    model_choice = st.radio(
        "Pilih model terbaik",
        options=["Machine Learning", "Deep Learning"],
        index=0,
        key="model_choice",
    )

    previous_model_choice = st.session_state.get("_previous_model_choice")
    if previous_model_choice is None:
        st.session_state["_previous_model_choice"] = model_choice
    elif previous_model_choice != model_choice:
        clear_classification_results()
        st.session_state["_previous_model_choice"] = model_choice
        st.info("Pilihan model berubah. Hasil klasifikasi sebelumnya disembunyikan sampai klasifikasi dijalankan ulang.")

    threshold = st.slider(
        "Threshold klasifikasi multi-label",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Label akan aktif jika probabilitasnya sama dengan atau lebih besar dari threshold ini.",
    )

    use_neutral_rule = st.checkbox(
        "Gunakan aturan Neutral eksklusif",
        value=True,
        help="Jika Problem/Appreciation/Suggestion muncul, Neutral dibuat 0. Jika tidak ada label lain, Neutral dibuat 1.",
    )

    st.divider()
    st.subheader("Wordcloud")
    use_default_stopwords = st.checkbox(
        "Buang stopwords Indonesia + Inggris",
        value=True,
        help="Jika aktif, kata umum seperti yang, dan, di, the, and, of, serta kata domain umum dibuang dari wordcloud.",
    )
    custom_stopwords_text = st.text_area(
        "Stopwords tambahan",
        value="",
        height=90,
        help="Pisahkan dengan koma, spasi, titik koma, atau baris baru. Contoh: tugas, materi, kelompok",
    )

    st.divider()
    st.subheader("Lokasi model")
    ml_model_path = st.text_input("Path model ML (.joblib/.pkl)", "models/best_ml_model.joblib")
    dl_model_path = st.text_input("Path model DL (.keras/.h5)", "models/best_dl_model.keras")
    tokenizer_path = st.text_input("Path tokenizer DL (.pkl/.joblib)", "models/tokenizer.pkl")

    if model_choice == "Deep Learning":
        with st.expander("Info runtime Deep Learning"):
            st.write(f"TensorFlow: `{_version_of(tf)}`")
            st.write(f"Keras: `{_version_of(standalone_keras)}`")
            if _major_version(_version_of(standalone_keras)) is not None and _major_version(_version_of(standalone_keras)) < 3:
                st.warning(
                    "Runtime saat ini masih Keras 2.x. Model `.keras` yang dibuat dengan Keras 3 "
                    "harus dijalankan dengan `tensorflow==2.16.2` dan `keras==3.4.1`."
                )

    max_len = st.number_input("Max sequence length DL", min_value=16, max_value=1024, value=200, step=8)
    batch_size = st.number_input("Batch size DL", min_value=1, max_value=512, value=32, step=1)

    st.divider()
    st.subheader("Output")
    auto_save_outputs = st.checkbox(
        "Simpan otomatis ke folder outputs/",
        value=True,
        help="Jika aktif, hasil klasifikasi dan learning insights disimpan ke folder outputs/ setiap kali proses klasifikasi selesai.",
    )
    output_base_dir = st.text_input("Folder output", "outputs")

wordcloud_stopwords: Set[str] = set()
if use_default_stopwords:
    wordcloud_stopwords.update(DEFAULT_WORDCLOUD_STOPWORDS)
wordcloud_stopwords.update(parse_custom_stopwords(custom_stopwords_text))

uploaded_file = st.file_uploader("Upload file CSV berisi teks peer feedback", type=["csv"])

if uploaded_file is None:
    st.info("Upload file CSV terlebih dahulu. File minimal memiliki satu kolom teks, misalnya `text`, `comment`, atau `feedback`.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"File CSV tidak dapat dibaca: {e}")
    st.stop()

if df.empty:
    st.warning("File CSV kosong.")
    st.stop()

st.subheader("1. Preview data")
st.dataframe(df.head(20), use_container_width=True)

candidate_cols = safe_columns(df)
default_text_col = next(
    (col for col in candidate_cols if col.lower() in ["text", "comment", "comments", "feedback", "peer_feedback", "peer_comment"]),
    candidate_cols[0],
)

text_col = st.selectbox(
    "Pilih kolom yang berisi teks peer feedback",
    options=candidate_cols,
    index=candidate_cols.index(default_text_col),
)

texts = df[text_col].map(normalize_text).tolist()
valid_mask = [bool(t) for t in texts]
if not any(valid_mask):
    st.error("Kolom teks yang dipilih tidak memiliki isi yang valid.")
    st.stop()

if st.button("🚀 Klasifikasikan peer feedback", type="primary"):
    progress = st.progress(0, text="Menyiapkan data...")

    try:
        progress.progress(15, text="Memuat model...")
        if model_choice == "Machine Learning":
            model = load_ml_model(ml_model_path)
            progress.progress(40, text="Mengklasifikasikan dengan model Machine Learning...")
            probabilities = predict_with_ml(model, texts, labels)
            selected_model_path = ml_model_path
        else:
            model, tokenizer = load_dl_assets(dl_model_path, tokenizer_path)
            progress.progress(40, text="Mengklasifikasikan dengan model Deep Learning...")
            probabilities = predict_with_dl(model, tokenizer, texts, labels, int(max_len), int(batch_size))
            selected_model_path = dl_model_path

        progress.progress(70, text="Mengolah hasil klasifikasi...")
        predictions = probabilities_to_predictions(probabilities, labels, threshold, use_neutral_rule)

        result_df = df.copy()
        result_df[text_col] = texts

        for i, label in enumerate(labels):
            result_df[f"prob_{label}"] = probabilities[:, i]
            result_df[f"pred_{label}"] = predictions[:, i]

        result_df["predicted_labels"] = predictions_to_label_string(predictions, labels)

        label_count_df, combination_df = build_insight_tables(result_df, labels)
        summary = generate_summary(label_count_df, combination_df, len(result_df))

        saved_paths = None
        if auto_save_outputs:
            progress.progress(90, text="Menyimpan output ke folder outputs/...")
            metadata = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "uploaded_filename": uploaded_file.name,
                "text_column": text_col,
                "model_choice": model_choice,
                "model_path": selected_model_path,
                "threshold": float(threshold),
                "use_neutral_rule": bool(use_neutral_rule),
                "use_default_wordcloud_stopwords": bool(use_default_stopwords),
                "custom_wordcloud_stopwords": sorted(parse_custom_stopwords(custom_stopwords_text)),
                "wordcloud_stopwords_count": int(len(wordcloud_stopwords)),
                "labels": labels,
                "total_rows": int(len(result_df)),
            }
            saved_paths = save_outputs_to_folder(
                result_df=result_df,
                label_count_df=label_count_df,
                combination_df=combination_df,
                summary=summary,
                metadata=metadata,
                output_base_dir=Path(output_base_dir),
            )

        progress.progress(100, text="Selesai.")

        st.session_state["classification_result_df"] = result_df
        st.session_state["label_count_df"] = label_count_df
        st.session_state["combination_df"] = combination_df
        st.session_state["summary"] = summary
        st.session_state["saved_paths"] = saved_paths
        st.session_state["active_labels"] = labels
        st.session_state["active_text_col"] = text_col
        st.session_state["active_model_choice"] = model_choice

        st.success("Klasifikasi selesai.")

    except FileNotFoundError as e:
        st.error(
            f"File model tidak ditemukan: {e}. Pastikan file model sudah berada di folder `models/` atau ubah path pada sidebar."
        )
    except RuntimeError as e:
        # RuntimeError from the model loader is already written as a user-friendly message.
        st.error(str(e))
    except Exception as e:
        st.error("Terjadi error saat klasifikasi. Detail teknis ditampilkan di bawah untuk debugging.")
        st.exception(e)

if (
    "classification_result_df" in st.session_state
    and st.session_state.get("active_model_choice", model_choice) == model_choice
):
    result_df = st.session_state["classification_result_df"]
    label_count_df = st.session_state["label_count_df"]
    combination_df = st.session_state["combination_df"]
    summary = st.session_state["summary"]
    saved_paths = st.session_state.get("saved_paths")
    labels = st.session_state.get("active_labels", labels)
    text_col = st.session_state.get("active_text_col", text_col)

    st.subheader("2. Hasil klasifikasi")
    st.dataframe(result_df, use_container_width=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download hasil klasifikasi CSV",
        data=csv_bytes,
        file_name="peer_feedback_classification_results.csv",
        mime="text/csv",
    )

    st.subheader("3. Learning insight: distribusi label")
    metric_cols = st.columns(len(labels))
    for idx, label in enumerate(labels):
        row = label_count_df[label_count_df["Label"] == label].iloc[0]
        metric_cols[idx].metric(label, int(row["Count"]), f"{row['Percentage']}%")

    left, right = st.columns(2)
    with left:
        st.write("**Jumlah feedback per label**")
        st.dataframe(label_count_df, use_container_width=True)
        st.bar_chart(label_count_df.set_index("Label")["Count"])

    with right:
        st.write("**Kombinasi label**")
        st.dataframe(combination_df, use_container_width=True)
        if not combination_df.empty:
            st.bar_chart(combination_df.set_index("Label combination")["Count"])

    st.subheader("4. Wordcloud per label")
    st.caption(
        f"Wordcloud dibuat setelah membuang {len(wordcloud_stopwords)} stopwords "
        "dan token yang terlalu pendek/berupa angka."
    )
    tabs = st.tabs(labels)
    for tab, label in zip(tabs, labels):
        with tab:
            label_texts = result_df.loc[result_df[f"pred_{label}"] == 1, text_col].tolist()
            render_wordcloud(" ".join(label_texts), label, stopwords=wordcloud_stopwords)

    st.subheader("5. Summary learning insight")
    st.write(summary)

    st.download_button(
        "⬇️ Download summary learning insight TXT",
        data=summary.encode("utf-8"),
        file_name="summary_learning_insight.txt",
        mime="text/plain",
    )

    st.subheader("6. Contoh feedback per label")
    for label in labels:
        with st.expander(f"Contoh feedback: {label}"):
            sample_df = result_df.loc[result_df[f"pred_{label}"] == 1, [text_col, "predicted_labels"]].head(10)
            if sample_df.empty:
                st.info(f"Tidak ada feedback yang diklasifikasikan sebagai {label}.")
            else:
                st.dataframe(sample_df, use_container_width=True)

    if saved_paths:
        render_saved_output_section(saved_paths)
    else:
        st.subheader("7. Penyimpanan output")
        st.info("Output belum disimpan ke folder. Aktifkan opsi `Simpan otomatis ke folder outputs/` pada sidebar, lalu jalankan klasifikasi kembali.")
