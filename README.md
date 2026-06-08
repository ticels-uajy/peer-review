# Peer Feedback Multi-Label Classifier and Learning Insight Generator

Aplikasi Streamlit ini mengklasifikasikan peer feedback / peer comments ke dalam label multi-label:

- Appreciation
- Problem
- Suggestion
- Neutral

Aplikasi mendukung dua artefak model terbaik:

- model machine learning: `best_ml_model.joblib`
- model deep learning: `best_dl_model.keras` + `best_dl_tokenizer.joblib`

## Fitur utama

1. Upload CSV berisi teks peer feedback.
2. Pilih model terbaik yang digunakan untuk prediksi: Machine Learning atau Deep Learning.
3. Klasifikasi multi-label setiap komentar.
4. Ringkasan learning insights global untuk pengajar.
5. Tab analisis untuk setiap label yang berisi:
   - abstractive summary,
   - wordcloud,
   - keyphrases/topik utama,
   - contoh komentar representatif.
6. Tab khusus **Keyphrase Extraction & Topic Modelling**:
   - keyphrase per label menggunakan TF-IDF unigram/bigram,
   - topic modelling global menggunakan NMF,
   - assignment topik untuk setiap dokumen.
7. Download hasil klasifikasi, ringkasan label, kombinasi label, keyphrase, dan topic modelling.

## Struktur artefak yang disarankan

```text
peer_feedback_streamlit_app/
├── app.py
├── requirements.txt
├── sample_peer_feedback.csv
└── artifacts/
    └── latest_run/
        ├── best_ml_model.joblib
        ├── best_ml_model_metadata.json
        ├── best_dl_model.keras
        ├── best_dl_tokenizer.joblib
        └── best_dl_model_metadata.json
```

## Instalasi

```bash
pip install -r requirements.txt
```

Jika memakai model deep learning `.keras` yang disimpan dengan Keras 3, gunakan environment bersih:

```bash
pip uninstall -y keras tensorflow tensorflow-cpu tf-keras
pip install -r requirements.txt
```

## Menjalankan aplikasi

```bash
streamlit run app.py
```

## Abstractive summarization

Aplikasi menyediakan dua mode summary:

1. **Transformer-based abstractive summarization** jika opsi di sidebar diaktifkan dan model summarization tersedia.
   Default model: `cahya/t5-base-indonesian-summarization-cased`.
2. **Topic-guided abstractive synthesis** sebagai fallback otomatis jika model Transformer tidak tersedia atau gagal dimuat.

Fallback tetap bersifat abstraktif dalam arti summary disusun sebagai narasi baru berbasis distribusi label, keyphrase, dan interpretasi pedagogis, bukan sekadar mengambil kalimat asli dari komentar.

## Format CSV

CSV harus memiliki minimal satu kolom teks. Nama kolom bebas, misalnya:

```csv
text
"Penjelasannya sudah baik, tetapi bagian metode perlu dibuat lebih jelas."
"Sebaiknya tambahkan contoh agar pembaca lebih mudah memahami."
```

Setelah upload, pilih kolom teks pada UI aplikasi.
