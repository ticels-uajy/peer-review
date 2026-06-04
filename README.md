# Peer Feedback Multi-label Classification Streamlit App

Aplikasi ini digunakan untuk mengklasifikasikan teks peer feedback/peer comments ke dalam label:

- Problem
- Appreciation
- Suggestion
- Neutral

## Struktur folder

```text
peer_feedback_streamlit_app/
├── app.py
├── requirements.txt
├── sample_peer_feedback.csv
├── models/
│   ├── best_ml_model.joblib
│   ├── best_dl_model.keras
│   ├── tokenizer.pkl
│   └── labels.json
└── outputs/
```

## Format CSV input

File CSV minimal memiliki satu kolom teks, misalnya:

```csv
text
The explanation is clear but the argument needs stronger evidence.
I appreciate the structure and suggest adding more examples.
```

Nama kolom dapat berupa `text`, `comment`, `feedback`, `peer_feedback`, atau nama lain. Kolom dapat dipilih dari UI Streamlit.

## Kontrak model ML

Simpan model ML terbaik ke:

```text
models/best_ml_model.joblib
```

Aplikasi mendukung beberapa format umum:

### Opsi A: sklearn Pipeline langsung

```python
import joblib

joblib.dump(best_ml_pipeline, "models/best_ml_model.joblib")
```

Pipeline sebaiknya sudah mencakup vectorizer dan classifier multi-label, misalnya `TfidfVectorizer + OneVsRestClassifier`.

### Opsi B: dictionary berisi vectorizer dan classifier

```python
import joblib

joblib.dump(
    {
        "vectorizer": tfidf_vectorizer,
        "model": best_classifier,
    },
    "models/best_ml_model.joblib"
)
```

Classifier sebaiknya mendukung `predict_proba`. Jika tidak tersedia, aplikasi akan mencoba `decision_function` atau `predict`.

## Kontrak model DL

Simpan model DL terbaik dan tokenizer ke:

```text
models/best_dl_model.keras
models/tokenizer.pkl
```

Contoh:

```python
import joblib

best_dl_model.save("models/best_dl_model.keras")
joblib.dump(tokenizer, "models/tokenizer.pkl")
```

Model DL diasumsikan menerima input hasil `Tokenizer.texts_to_sequences` yang diproses dengan `pad_sequences`, lalu menghasilkan probabilitas multi-label berukuran `(n_data, n_labels)`.

## Format labels.json

Agar urutan output model konsisten dengan nama label, buat file:

```json
["Problem", "Appreciation", "Suggestion", "Neutral"]
```

atau:

```json
{
  "labels": ["Problem", "Appreciation", "Suggestion", "Neutral"]
}
```

Urutan label di `labels.json` harus sama dengan urutan output probabilitas model.

## Menjalankan aplikasi

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Catatan tentang Neutral

Di sidebar terdapat opsi **Gunakan aturan Neutral eksklusif**.

Jika aktif:

- Jika `Problem`, `Appreciation`, atau `Suggestion` muncul, maka `Neutral = 0`.
- Jika tidak ada label non-neutral yang muncul, maka `Neutral = 1`.

Matikan opsi ini jika skema anotasi Anda memperbolehkan `Neutral` muncul bersama label lain.
