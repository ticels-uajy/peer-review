# Peer Feedback Multi-Label Classifier Streamlit App

Aplikasi ini digunakan untuk mengklasifikasikan data teks peer feedback/peer comments ke dalam empat label multi-label:

- Appreciation
- Problem
- Suggestion
- Neutral

Aplikasi mendukung dua model terbaik:

1. Model Machine Learning: `best_ml_model.joblib`
2. Model Deep Learning: `best_dl_model.keras` + `best_dl_tokenizer.joblib`

## Struktur Folder yang Disarankan

Letakkan artefak model dari pipeline training di folder berikut:

```text
peer_feedback_streamlit_app/
├── app.py
├── requirements.txt
└── artifacts/
    └── latest_run/
        ├── best_ml_model.joblib
        ├── best_ml_model_metadata.json
        ├── best_dl_model.keras
        ├── best_dl_tokenizer.joblib
        └── best_dl_model_metadata.json
```

Jika nama folder run Anda berbeda, ubah input **Folder artefak model** di sidebar Streamlit.

## Instalasi

```bash
pip install -r requirements.txt
```

Jika hanya menggunakan model ML, dependency TensorFlow dapat dihapus dari `requirements.txt`.

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

## Format CSV Input

CSV minimal memiliki satu kolom teks, misalnya:

```csv
text
"Bagian ini sudah bagus, tetapi perlu ditambahkan referensi."
"Tulisan sudah rapi dan mudah dipahami."
```

Di aplikasi, pilih nama kolom yang berisi teks peer feedback.

## Fitur

1. Upload data teks peer feedback dalam format CSV.
2. Memilih model terbaik yang digunakan untuk klasifikasi: ML atau DL.
3. Mengklasifikasikan teks peer feedback secara multi-label.
4. Menampilkan insight jumlah feedback per label dan kombinasi label.
5. Menampilkan wordcloud untuk setiap label.
6. Menampilkan summary learning insight otomatis.
7. Mengunduh hasil klasifikasi dan ringkasan dalam format CSV.

## Output Prediksi

Aplikasi akan menambahkan kolom berikut:

```text
pred_Appreciation
pred_Problem
pred_Suggestion
pred_Neutral
score_Appreciation
score_Problem
score_Suggestion
score_Neutral
predicted_labels
n_predicted_labels
```

## Catatan Metodologis

Aplikasi ini hanya digunakan untuk inference/deployment. Proses pemilihan model terbaik tetap dilakukan di pipeline training dengan protokol:

```text
Dataset gabungan
→ iterative multilabel train-test split 80:20
→ training set untuk cross-validation dan model selection
→ threshold tuning pada validation fold/split
→ test set hanya untuk final evaluation
```
