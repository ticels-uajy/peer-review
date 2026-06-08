# Peer Feedback Multi-Label Classifier

Aplikasi Streamlit untuk mengklasifikasikan peer feedback/peer comments ke dalam label multi-label: Appreciation, Problem, Suggestion, dan Neutral. Aplikasi ini mendukung model terbaik berbasis Machine Learning dan Deep Learning, serta menghasilkan learning insights untuk pengajar.

## Fitur Utama

1. Upload data peer feedback dalam format CSV.
2. Pilih model terbaik: Machine Learning atau Deep Learning.
3. Klasifikasi multi-label untuk setiap komentar.
4. Ringkasan learning insights yang lebih readable untuk pengajar.
5. Tab analisis untuk setiap label yang berisi:
   - abstractive summary,
   - wordcloud,
   - keyphrases/topik utama,
   - contoh komentar representatif.
6. Bagian **Keyphrase Extraction & Topic Modelling per Label**:
   - keyphrase per label menggunakan TF-IDF unigram, bigram, dan trigram,
   - topic modelling NMF khusus untuk setiap label,
   - assignment topik untuk setiap dokumen pada masing-masing label,
   - download hasil keyphrase, topik, dan assignment topik.
7. Topic modelling global tersedia sebagai informasi tambahan melalui expander.
8. Download hasil klasifikasi, ringkasan label, dan kombinasi label.

## Struktur Artefak Model

Letakkan artefak model pada folder, misalnya:

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

Jika menggunakan model DL dengan format Keras 3 dan environment lama bermasalah, jalankan:

```bash
pip uninstall -y keras tensorflow tensorflow-cpu tf-keras
pip install -r requirements.txt
streamlit run app.py
```
