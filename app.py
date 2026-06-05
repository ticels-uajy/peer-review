import io
import json
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

# Optional imports for deep learning model.
# The app will still run for ML mode even if TensorFlow is not installed,
# as long as DL mode is not selected.
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:  # pragma: no cover
    tf = None
    pad_sequences = None


APP_TITLE = "Peer Feedback Multi-label Classification"
DEFAULT_LABELS = ["Problem", "Appreciation", "Suggestion", "Neutral"]
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Light text cleaning for display, wordcloud, and fallback preprocessing."""
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_wordcloud(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    """
    Optional practical rule:
    - If Problem/Appreciation/Suggestion is predicted, Neutral is set to 0.
    - If no non-neutral label is predicted, Neutral is set to 1.

    This is often suitable when Neutral means "no specific feedback act".
    Disable from the sidebar if your dataset allows Neutral to co-occur with others.
    """
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
    """
    Converts several common sklearn outputs to a 2D probability-like matrix.
    """
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


@st.cache_resource(show_spinner=False)
def load_dl_assets(model_path: str, tokenizer_path: str):
    if tf is None:
        raise ImportError(
            "TensorFlow is not installed. Install it with `pip install tensorflow` to use DL mode."
        )
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer


def predict_with_ml(model, texts: List[str], labels: List[str]) -> np.ndarray:
    """
    Supports common sklearn patterns:
    1. Pipeline/OneVsRestClassifier with predict_proba
    2. Pipeline/model with decision_function
    3. Pipeline/model with predict
    4. Saved dict {'model': ..., 'vectorizer': ...}
    """
    x = texts

    if isinstance(model, dict):
        vectorizer = model.get("vectorizer")
        classifier = model.get("model") or model.get("classifier")
        if vectorizer is not None:
            x = vectorizer.transform(texts)
        if classifier is None:
            raise ValueError("ML model dict must contain key 'model' or 'classifier'.")
        model = classifier

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
        raise ValueError("Unsupported ML model. Provide a sklearn-compatible model or pipeline.")

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
    """Save classification results and learning insights to a timestamped folder."""
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
    """Create an in-memory ZIP from a saved output folder."""
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


def render_wordcloud(text: str, title: str):
    text = clean_for_wordcloud(text)
    if not text:
        st.info(f"Tidak ada teks yang cukup untuk membuat wordcloud label {title}.")
        return

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        collocations=False,
        max_words=150,
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
    )

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
    st.subheader("Lokasi model")
    ml_model_path = st.text_input("Path model ML (.joblib/.pkl)", "models/best_ml_model.joblib")
    dl_model_path = st.text_input("Path model DL (.keras/.h5)", "models/best_dl_model.keras")
    tokenizer_path = st.text_input("Path tokenizer DL (.pkl/.joblib)", "models/tokenizer.pkl")
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

        st.success("Klasifikasi selesai.")

    except FileNotFoundError as e:
        st.error(
            f"File model tidak ditemukan: {e}. Pastikan file model sudah berada di folder `models/` atau ubah path pada sidebar."
        )
    except Exception as e:
        st.exception(e)

if "classification_result_df" in st.session_state:
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
    tabs = st.tabs(labels)
    for tab, label in zip(tabs, labels):
        with tab:
            label_texts = result_df.loc[result_df[f"pred_{label}"] == 1, text_col].tolist()
            render_wordcloud(" ".join(label_texts), label)

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
