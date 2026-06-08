# Peer Feedback Insight Generator

Aplikasi Streamlit untuk klasifikasi multi-label peer feedback dan pembuatan learning insights bagi pengajar.

## Fitur

1. Upload data teks peer feedback dalam format CSV.
2. Pilih model terbaik yang digunakan untuk klasifikasi: Machine Learning atau Deep Learning.
3. Klasifikasi multi-label untuk setiap komentar: `Appreciation`, `Problem`, `Suggestion`, dan `Neutral`.
4. Ringkasan learning insights yang lebih readable untuk pengajar.
5. Tab analisis untuk setiap label, berisi:
   - abstractive summary,
   - wordcloud,
   - keyphrases/topik utama,
   - contoh komentar representatif.
6. Bagian **Keyphrase Extraction & Topic Modelling per Label**, berisi:
   - keyphrase per label menggunakan TF-IDF unigram, bigram, dan trigram,
   - topic modelling NMF khusus untuk setiap label,
   - assignment topik untuk setiap dokumen pada masing-masing label,
   - download hasil keyphrase, topik, dan assignment topik.
7. Topic modelling global tersedia sebagai informasi tambahan melalui expander.
8. Download semua hasil analisis dalam satu ZIP, termasuk:
   - hasil klasifikasi multi-label,
   - ringkasan label,
   - kombinasi label,
   - co-occurrence antarlabel,
   - learning insights Markdown,
   - summary per label,
   - wordcloud PNG per label,
   - keyphrases per label,
   - NMF topics per label,
   - assignment topik per label,
   - topic modelling global,
   - metadata model.

## Struktur Artefak Model

Letakkan artefak model dalam folder seperti berikut:

```text
artifacts/latest_run/
├── best_ml_model.joblib
├── best_ml_model_metadata.json
├── best_dl_model.keras
├── best_dl_tokenizer.joblib
└── best_dl_model_metadata.json
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Jika model DL `.keras` disimpan dengan Keras 3, gunakan environment yang bersih:

```bash
pip uninstall -y keras tensorflow tensorflow-cpu tf-keras
pip install -r requirements.txt
streamlit run app.py
```

## Catatan Summary

Aplikasi menyediakan summary per label. Jika opsi Transformer-based abstractive summarization diaktifkan dan model summarization tersedia, aplikasi akan mencoba menggunakan model tersebut. Jika gagal, aplikasi otomatis menggunakan fallback topic-guided abstractive synthesis berbasis jumlah komentar, keyphrase, kombinasi label, dan interpretasi pedagogis.
