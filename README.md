# Peer Feedback Multi-Label Classifier

Aplikasi Streamlit untuk mengklasifikasikan peer feedback/peer comments ke dalam label multi-label: `Appreciation`, `Problem`, `Suggestion`, dan `Neutral`, lalu mengubah hasil klasifikasi menjadi learning insights untuk pengajar.

## Fitur

1. Upload CSV peer feedback.
2. Pilih model terbaik: Machine Learning atau Deep Learning.
3. Klasifikasi multi-label untuk setiap komentar.
4. Ringkasan learning insights yang readable dan actionable.
5. Tab analisis untuk setiap label:
   - abstractive summary,
   - wordcloud,
   - keyphrases/topik utama,
   - contoh komentar representatif.
6. Tab khusus **Keyphrase Extraction & Topic Modelling**:
   - keyphrase per label menggunakan TF-IDF unigram/bigram/trigram,
   - topic modelling global menggunakan NMF,
   - assignment topik untuk setiap dokumen.
7. Download hasil klasifikasi, ringkasan label, kombinasi label, keyphrases, dan assignment topik.

## Struktur artefak model

Letakkan artefak model di folder, misalnya:

```text
artifacts/latest_run/
├── best_ml_model.joblib
├── best_ml_model_metadata.json
├── best_dl_model.keras
├── best_dl_tokenizer.joblib
└── best_dl_model_metadata.json
```

## Menjalankan aplikasi

```bash
pip install -r requirements.txt
streamlit run app.py
```

Jika environment lama masih bermasalah dengan TensorFlow/Keras:

```bash
pip uninstall -y keras tensorflow tensorflow-cpu tf-keras
pip install -r requirements.txt
streamlit run app.py
```

## Catatan abstractive summarization

Aplikasi memiliki dua mode summary:

1. Transformer-based abstractive summarization, jika diaktifkan dan model tersedia.
2. Fallback topic-guided abstractive synthesis, yang tetap berjalan tanpa model Transformer.

Model default untuk summarization adalah:

```text
cahya/t5-base-indonesian-summarization-cased
```

Jika aplikasi dijalankan di server tanpa internet, pastikan model sudah tersedia di cache lokal atau gunakan fallback summary.
