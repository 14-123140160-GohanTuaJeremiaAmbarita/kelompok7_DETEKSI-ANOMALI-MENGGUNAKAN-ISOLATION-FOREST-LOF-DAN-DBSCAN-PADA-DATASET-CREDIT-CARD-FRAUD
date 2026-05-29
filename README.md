# kelompok7_DETEKSI-ANOMALI-MENGGUNAKAN-ISOLATION-FOREST-LOF-DAN-DBSCAN-PADA-DATASET-CREDIT-CARD-FRAUD

# Deteksi Anomali Menggunakan Isolation Forest, LOF, dan DBSCAN pada Dataset Credit Card Fraud

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Data Mining](https://img.shields.io/badge/Domain-Data%20Mining-orange)
![Machine Learning](https://img.shields.io/badge/Framework-Scikit--Learn-blue)
![Institution](https://img.shields.io/badge/Institution-ITERA-gold)

Repositori ini berisi proyek penelitian eksperimental mengenai analisis komparatif tiga metode *unsupervised anomaly detection*—**Isolation Forest**, **Local Outlier Factor (LOF)**, dan **DBSCAN**—sebagai tahap *preprocessing* untuk meningkatkan performa klasifikasi penipuan (*fraud detection*) pada transaksi kartu kredit. Proyek ini disusun untuk memenuhi tugas mata kuliah **IF25-35025 Data Mining**, Program Studi Teknik Informatika, Institut Teknologi Sumatera (ITERA).

---

## 📌 1. Latar Belakang & Ringkasan Penelitian
Peningkatan volume transaksi digital berdampak langsung pada melonjaknya risiko *credit card fraud* yang merugikan secara finansial. Dalam konteks data mining, kasus penipuan ini memunculkan tantangan besar berupa **Extreme Class Imbalance** (ketidakseimbangan kelas ekstrem), di mana transaksi *fraud* hanya mencakup **0,172%** dari keseluruhan dataset (rasio 578:1). 

Keberadaan pencilan (*outlier*) atau data yang menyimpang berpotensi mengaburkan pola penting dan menurunkan performa model klasifikasi konvensional. Penelitian ini menerapkan pendekatan *unsupervised anomaly detection* pada tahap *preprocessing* data latih untuk membersihkan data dari gangguan (*noise*), mempertegas batas keputusan (*decision boundary*), dan membandingkan dampaknya terhadap model klasifikasi akhir berbasis **Random Forest Classifier**.

---

## 🚀 2. Alur Pipeline Sistematis
Untuk menjamin validitas hasil eksperimen dan mencegah terjadinya kebocoran data (**data leakage**), proyek ini mengimplementasikan alur kerja yang ketat:

1. **Pengumpulan Data**: Mengambil dataset *Credit Card Fraud* yang berisi 284.807 transaksi dari Kaggle.
2. **Eksplorasi Data (EDA)**: Menganalisis karakteristik statistik, struktur data, dan mengecek ketersediaan data (0% *missing values*).
3. **Stratified Train-Test Split**: Memisahkan data menjadi **80% Data Latih** dan **20% Data Uji**. Penggunaan *stratified sampling* menjamin proporsi kelas target tetap konsisten di kedua set.
4. **Feature Scaling**: Menyamakan skala fitur numerik menggunakan `StandardScaler` yang dipasang (*fitted*) hanya pada data latih.
5. **Deteksi Anomali (Preprocessing Data Latih)**: Mengidentifikasi dan menghapus data yang dianggap anomali/outlier menggunakan tiga algoritma unsupervised secara terpisah.
6. **Pelatihan Model Klasifikasi**: Melatih model klasifikasi *Random Forest* menggunakan data latih asli (Baseline) serta masing-masing data latih hasil pembersihan anomali.
7. **Evaluasi Objektif**: Menguji kinerja akhir model pada *data uji asli yang belum pernah dimanipulasi* guna mencerminkan performa pada kondisi riil.

---

## ⚙️ 3. Detail Teknis & Parameter Model

### A. Dependensi Environment
* Python 3.8+
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib

### B. Konfigurasi Parameter Algoritma Preprocessing
1. **Isolation Forest (I-Forest)**: 
   * `n_estimators=100`, `contamination=0.0017` (disesuaikan dengan rasio fraud riil), `random_state=42`.
   * *Alasan*: Sangat efisien untuk data skala besar dan berdimensi tinggi karena memiliki kompleksitas linear.
2. **Local Outlier Factor (LOF)**: 
   * `n_neighbors=20`, `contamination=0.0017`, `metric='minkowski'`.
   * *Alasan*: Bagus untuk menangkap anomali lokal berdasarkan kerapatan lingkungan sekitarnya.
3. **DBSCAN**: 
   * `eps=2.5`, `min_samples=5`, `metric='euclidean'`.
   * *Alasan*: Mampu mendeteksi klaster dengan bentuk tidak beraturan dan mengisolasi data di luar klaster sebagai *noise*.
4. **Random Forest Classifier (Evaluator)**:
   * `n_estimators=100`, `random_state=42`.
   * *Alasan*: Tangguh terhadap overfitting dan pencilan tersisa, serta mampu menangani fitur non-linear hasil PCA secara efektif.

---

## 📊 4. Hasil Eksperimen Utama

### A. Deteksi Outlier pada Data Latih (Total Data Latih Awal: 227.845 Sampel)
Masing-masing algoritma menunjukkan karakteristik penyaringan data yang sangat kontras:

| Metode Preprocessing | Data Latih Awal | Data Setelah Pembersihan | Outlier Terdeteksi | Persentase Outlier |
| :--- | :---: | :---: | :---: | :---: |
| **Isolation Forest** | 227.845 | 227.457 | 388 | 0,17% |
| **LOF (Local Outlier Factor)** | 227.845 | 227.457 | 388 | 0,17% |
| **DBSCAN** | 227.845 | 201.140 | 26.705 | 11,72% |

### B. Performa Akhir Klasifikasi Random Forest (Data Uji Asli)
Evaluasi performa klasifikasi diuji langsung pada data uji asli untuk mendapatkan metrik yang valid:

| Metode | Precision | Recall | F1-Score | AUC-ROC | Keterangan / Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Tanpa Preprocessing)** | **0.94** | 0.82 | **0.87** | **0.963** | Standar performa awal model. |
| **Isolation Forest** | 0.87 | **0.85** | 0.86 | 0.953 | **Metode Terbaik (Recall ↑ 3,66%)**. |
| **Local Outlier Factor (LOF)** | **0.94** | 0.82 | **0.87** | **0.963** | Performa identik dengan Baseline. |
| **DBSCAN** | 0.78 | 0.33 | 0.46 | 0.939 | Performa anjlok (*Over-cleaning*). |

---

## 💡 5. Analisis Hasil & Inovasi Proyek

### Analisis Temuan Eksperimen
* **Trade-off Metrik Terkendali**: Penerapan **Isolation Forest** berhasil menaikkan tingkat sensitivitas model dalam mengenali transaksi ilegal, dibuktikan dengan **skor Recall yang naik dari 0.82 menjadi 0.85**. Konsekuensinya, terjadi sedikit penurunan *precision* karena model menjadi lebih agresif mendeteksi fraud.
* **Kegagalan Deteksi Berbasis Kepadatan (DBSCAN)**: Tanpa optimasi parameter (*hyperparameter tuning*) yang tepat, DBSCAN mengelompokkan data secara salah pada dimensi tinggi hasil PCA. Penghapusan data sebesar **11,72%** menghilangkan pola krusial transaksi *fraud*, sehingga performa *Recall* jatuh bebas ke angka **0.33**.
* **Inersia Model LOF**: LOF mendeteksi jumlah anomali yang terkontrol, tetapi tidak berdampak pada performa akhir. Ini mengindikasikan bahwa data pencilan lokal yang dibuang tidak tumpang tindih dengan area keputusan penting pada model *Random Forest*.

### Inovasi & Nilai Tambah Proyek ini
1. **Pipeline Anti-Bocor (Zero Data Leakage)**: Proyek ini secara ketat menerapkan *Train-Test split sebelum scaling dan deteksi anomali*. Banyak implementasi awam melakukan penyaringan outlier di awal seluruh dataset, yang menyebabkan bias evaluasi (*data leakage*).
2. **Konteks Komersial Perbankan**: Alih-alih hanya mengejar nilai akurasi umum (*accuracy*), proyek ini memprioritaskan peningkatan **Recall**. Dalam industri perbankan, meloloskan satu transaksi kejahatan (*false negative*) berdampak kerugian finansial yang jauh lebih besar dibanding melakukan verifikasi tambahan akibat salah sangka (*false positive*).
3. **Penyelarasan Kontaminasi Empiris**: Penentuan nilai parameter `contamination=0.0017` pada *Isolation Forest* dan *LOF* merupakan penyesuaian berbasis pengetahuan domain (*domain knowledge*) terhadap karakteristik distribusi asli dataset, bukan sekadar tebakan acak.

---

## 📝 6. Rangkuman Lengkap Penelitian
* **Karakteristik Data**: Dataset sangat tidak seimbang dengan 284.807 baris, di mana kelas *fraud* hanya berjumlah 492 transaksi.
* **Efektivitas Metode Preprocessing**: *Isolation Forest* keluar sebagai skema *preprocessing* terbaik karena sukses menyaring data pengganggu (*noise*) secara presisi tanpa membuang informasi penting.
* **Kesimpulan Utama**: Teknik *unsupervised anomaly detection* tidak selalu meningkatkan performa klasifikasi. Penerapannya harus dilakukan secara hati-hati; pendekatan yang terlalu agresif (seperti DBSCAN pada kasus ini) justru akan merusak pemahaman model terhadap data. Keselarasan antara mekanisme algoritma preprocessing dengan struktur distribusi data adalah kunci sukses perbaikan performa.

---

## 👥 Anggota Kelompok Proyek (Kelompok 07)

Proyek data mining ini berhasil diselesaikan berkat kolaborasi dari tim Kelompok 07:

| Nama Anggota | NIM | Peran Utama dalam Tim |
| :--- | :---: | :--- |
| **Aditya Hot Martua Sihite** | 123140114 | Perancangan Eksperimen & Analisis Teoretis |
| **Gohan Tua Ambarita** | 123140160 | Pengembangan Kode Python & Integrasi Pipeline |
| **Jana Rohman Wasiso** | 123140046 | Eksplorasi Data (EDA) & Visualisasi Matplotlib |
| **Hezkiel Rajani Aritonang** | 123140118 | Implementasi & Tuning Pemodelan Klasifikasi |
| **Adelia Ramadani** | 123140183 | Evaluasi Metrik & Tabel Perbandingan Performa |
| **Rian Rafael Sangap Tamba** | 123140190 | Penyusunan Laporan Utama & Dokumentasi BAB IV |
| **Nadine Aura Rahmadhani** | 123140195 | Studi Literatur & Validasi Metrik Imbalanced Data |

---
**PROGRAM STUDI TEKNIK INFORMATIKA** **FAKULTAS TEKNOLOGI INDUSTRI** **INSTITUT TEKNOLOGI SUMATERA (ITERA)** **2026**
