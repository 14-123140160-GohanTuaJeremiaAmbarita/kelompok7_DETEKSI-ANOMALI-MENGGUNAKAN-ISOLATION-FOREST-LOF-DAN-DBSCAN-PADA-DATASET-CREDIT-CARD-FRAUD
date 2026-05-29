# kelompok7_DETEKSI-ANOMALI-MENGGUNAKAN-ISOLATION-FOREST-LOF-DAN-DBSCAN-PADA-DATASET-CREDIT-CARD-FRAUD

# Deteksi Anomali Menggunakan Isolation Forest, LOF, dan DBSCAN pada Dataset Credit Card Fraud

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Data Mining](https://img.shields.io/badge/Domain-Data%20Mining-orange)
![Machine Learning](https://img.shields.io/badge/Framework-Scikit--Learn-blue)
![Institution](https://img.shields.io/badge/Institution-ITERA-gold)

[cite_start]Repositori ini berisi proyek penelitian eksperimental mengenai analisis komparatif tiga metode *unsupervised anomaly detection*—**Isolation Forest**, **Local Outlier Factor (LOF)**, dan **DBSCAN**—sebagai tahap *preprocessing* untuk meningkatkan performa klasifikasi penipuan (*fraud detection*) pada transaksi kartu kredit[cite: 17]. [cite_start]Proyek ini disusun untuk memenuhi tugas mata kuliah **IF25-35025 Data Mining**, Program Studi Teknik Informatika, Institut Teknologi Sumatera (ITERA)[cite: 2, 9, 11].

---

## 📌 1. Latar Belakang & Ringkasan Penelitian
[cite_start]Peningkatan volume transaksi digital berdampak langsung pada melonjaknya risiko *credit card fraud* yang merugikan secara finansial[cite: 27, 34, 35]. [cite_start]Dalam konteks data mining, kasus penipuan ini memunculkan tantangan besar berupa **Extreme Class Imbalance** (ketidakseimbangan kelas ekstrem), di mana transaksi *fraud* hanya mencakup **0,172%** dari keseluruhan dataset (rasio 578:1)[cite: 15, 18, 185]. 

[cite_start]Keberadaan pencilan (*outlier*) atau data yang menyimpang berpotensi mengaburkan pola penting dan menurunkan performa model klasifikasi konvensional[cite: 16, 31]. [cite_start]Penelitian ini menerapkan pendekatan *unsupervised anomaly detection* pada tahap *preprocessing* data latih untuk membersihkan data dari gangguan (*noise*), mempertegas batas keputusan (*decision boundary*), dan membandingkan dampaknya terhadap model klasifikasi akhir berbasis **Random Forest Classifier**[cite: 20, 23, 219].

---

## 🚀 2. Alur Pipeline Sistematis
[cite_start]Untuk menjamin validitas hasil eksperimen dan mencegah terjadinya kebocoran data (**data leakage**), proyek ini mengimplementasikan alur kerja yang ketat[cite: 19, 166]:

1. [cite_start]**Pengumpulan Data**: Mengambil dataset *Credit Card Fraud* yang berisi 284.807 transaksi dari Kaggle[cite: 65, 168].
2. [cite_start]**Eksplorasi Data (EDA)**: Menganalisis karakteristik statistik, struktur data, dan mengecek ketersediaan data (0% *missing values*)[cite: 170, 186].
3. [cite_start]**Stratified Train-Test Split**: Memisahkan data menjadi **80% Data Latih** dan **20% Data Uji**[cite: 171, 193]. [cite_start]Penggunaan *stratified sampling* menjamin proporsi kelas target tetap konsisten di kedua set[cite: 193, 194].
4. [cite_start]**Feature Scaling**: Menyamakan skala fitur numerik menggunakan `StandardScaler` yang dipasang (*fitted*) hanya pada data latih[cite: 174].
5. [cite_start]**Deteksi Anomali (Preprocessing Data Latih)**: Mengidentifikasi dan menghapus data yang dianggap anomali/outlier menggunakan tiga algoritma unsupervised secara terpisah[cite: 174, 209].
6. [cite_start]**Pelatihan Model Klasifikasi**: Melatih model klasifikasi *Random Forest* menggunakan data latih asli (Baseline) serta masing-masing data latih hasil pembersihan anomali[cite: 175, 176].
7. [cite_start]**Evaluasi Objektif**: Menguji kinerja akhir model pada *data uji asli yang belum pernah dimanipulasi* guna mencerminkan performa pada kondisi riil[cite: 177, 178].

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
   * [cite_start]`n_estimators=100`, `contamination=0.0017` (disesuaikan dengan rasio fraud riil), `random_state=42`[cite: 211].
   * [cite_start]*Alasan*: Sangat efisien untuk data skala besar dan berdimensi tinggi karena memiliki kompleksitas linear[cite: 212].
2. **Local Outlier Factor (LOF)**: 
   * [cite_start]`n_neighbors=20`, `contamination=0.0017`, `metric='minkowski'`[cite: 215].
   * [cite_start]*Alasan*: Bagus untuk menangkap anomali lokal berdasarkan kerapatan lingkungan sekitarnya[cite: 213, 217].
3. **DBSCAN**: 
   * `eps=2.5`, `min_samples=5`, `metric='euclidean'`.
   * [cite_start]*Alasan*: Mampu mendeteksi klaster dengan bentuk tidak beraturan dan mengisolasi data di luar klaster sebagai *noise*[cite: 115, 223].
4. **Random Forest Classifier (Evaluator)**:
   * [cite_start]`n_estimators=100`, `random_state=42`[cite: 230].
   * [cite_start]*Alasan*: Tangguh terhadap overfitting dan pencilan tersisa, serta mampu menangani fitur non-linear hasil PCA secara efektif[cite: 233, 235].

---

## 📊 4. Hasil Eksperimen Utama

### A. Deteksi Outlier pada Data Latih (Total Data Latih Awal: 227.845 Sampel)
[cite_start]Masing-masing algoritma menunjukkan karakteristik penyaringan data yang sangat kontras[cite: 257]:

| Metode Preprocessing | Data Latih Awal | Data Setelah Pembersihan | Outlier Terdeteksi | Persentase Outlier |
| :--- | :---: | :---: | :---: | :---: |
| **Isolation Forest** | 227.845 | 227.457 | 388 | 0,17% |
| **LOF (Local Outlier Factor)** | 227.845 | 227.457 | 388 | 0,17% |
| **DBSCAN** | 227.845 | 201.140 | 26.705 | 11,72% |

### B. Performa Akhir Klasifikasi Random Forest (Data Uji Asli)
[cite_start]Evaluasi performa klasifikasi diuji langsung pada data uji asli untuk mendapatkan metrik yang valid[cite: 252]:

| Metode | Precision | Recall | F1-Score | AUC-ROC | Keterangan / Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Tanpa Preprocessing)** | **0.94** | 0.82 | **0.87** | **0.963** | [cite_start]Standar performa awal model[cite: 313]. |
| **Isolation Forest** | 0.87 | **0.85** | 0.86 | 0.953 | [cite_start]**Metode Terbaik (Recall ↑ 3,66%)**[cite: 543, 573]. |
| **Local Outlier Factor (LOF)** | **0.94** | 0.82 | **0.87** | **0.963** | [cite_start]Performa identik dengan Baseline[cite: 325]. |
| **DBSCAN** | 0.78 | 0.33 | 0.46 | 0.939 | [cite_start]Performa anjlok (*Over-cleaning*)[cite: 329, 331]. |

---

## 💡 5. Analisis Hasil & Inovasi Proyek

### Analisis Temuan Eksperimen
* [cite_start]**Trade-off Metrik Terkendali**: Penerapan **Isolation Forest** berhasil menaikkan tingkat sensitivitas model dalam mengenali transaksi ilegal, dibuktikan dengan **skor Recall yang naik dari 0.82 menjadi 0.85**[cite: 318]. [cite_start]Konsekuensinya, terjadi sedikit penurunan *precision* karena model menjadi lebih agresif mendeteksi fraud[cite: 320, 323].
* [cite_start]**Kegagalan Deteksi Berbasis Kepadatan (DBSCAN)**: Tanpa optimasi parameter (*hyperparameter tuning*) yang tepat, DBSCAN mengelompokkan data secara salah pada dimensi tinggi hasil PCA[cite: 263, 595, 596]. [cite_start]Penghapusan data sebesar **11,72%** menghilangkan pola krusial transaksi *fraud*, sehingga performa *Recall* jatuh bebas ke angka **0.33**[cite: 329, 331].
* [cite_start]**Inersia Model LOF**: LOF mendeteksi jumlah anomali yang terkontrol, tetapi tidak berdampak pada performa akhir[cite: 260, 325]. [cite_start]Ini mengindikasikan bahwa data pencilan lokal yang dibuang tidak tumpang tindih dengan area keputusan penting pada model *Random Forest*[cite: 326].

### Inovasi & Nilai Tambah Proyek ini
1. [cite_start]**Pipeline Anti-Bocor (Zero Data Leakage)**: Proyek ini secara ketat menerapkan *Train-Test split sebelum scaling dan deteksi anomali*[cite: 172, 195]. [cite_start]Banyak implementasi awam melakukan penyaringan outlier di awal seluruh dataset, yang menyebabkan bias evaluasi (*data leakage*)[cite: 19].
2. [cite_start]**Konteks Komersial Perbankan**: Alih-alih hanya mengejar nilai akurasi umum (*accuracy*), proyek ini memprioritaskan peningkatan **Recall**[cite: 239, 242]. [cite_start]Dalam industri perbankan, meloloskan satu transaksi kejahatan (*false negative*) berdampak kerugian finansial yang jauh lebih besar dibanding melakukan verifikasi tambahan akibat salah sangka (*false positive*)[cite: 242, 535, 536].
3. [cite_start]**Penyelarasan Kontaminasi Empiris**: Penentuan nilai parameter `contamination=0.0017` pada *Isolation Forest* dan *LOF* merupakan penyesuaian berbasis pengetahuan domain (*domain knowledge*) terhadap karakteristik distribusi asli dataset, bukan sekadar tebakan acak[cite: 211, 259].

---

## 📝 6. Rangkuman Lengkap Penelitian
* [cite_start]**Karakteristik Data**: Dataset sangat tidak seimbang dengan 284.807 baris, di mana kelas *fraud* hanya berjumlah 492 transaksi[cite: 18, 182, 185].
* [cite_start]**Efektivitas Metode Preprocessing**: *Isolation Forest* keluar sebagai skema *preprocessing* terbaik karena sukses menyaring data pengganggu (*noise*) secara presisi tanpa membuang informasi penting[cite: 543, 571].
* [cite_start]**Kesimpulan Utama**: Teknik *unsupervised anomaly detection* tidak selalu meningkatkan performa klasifikasi[cite: 501]. [cite_start]Penerapannya harus dilakukan secara hati-hati; pendekatan yang terlalu agresif (seperti DBSCAN pada kasus ini) justru akan merusak pemahaman model terhadap data[cite: 560]. [cite_start]Keselarasan antara mekanisme algoritma preprocessing dengan struktur distribusi data adalah kunci sukses perbaikan performa[cite: 510].

---

## 👥 Anggota Kelompok Proyek (Kelompok 07)

[cite_start]Proyek data mining ini berhasil diselesaikan berkat kolaborasi dari tim Kelompok 07[cite: 4]:

| Nama Anggota | NIM | Peran Utama dalam Tim |
| :--- | :---: | :--- |
| **Aditya Hot Martua Sihite** | 123140114 | [cite_start]Perancangan Eksperimen & Analisis Teoretis [cite: 7] |
| **Gohan Tua Ambarita** | 123140160 | [cite_start]Pengembangan Kode Python & Integrasi Pipeline [cite: 7] |
| **Jana Rohman Wasiso** | 123140046 | [cite_start]Eksplorasi Data (EDA) & Visualisasi Matplotlib [cite: 8] |
| **Hezkiel Rajani Aritonang** | 123140118 | [cite_start]Implementasi & Tuning Pemodelan Klasifikasi [cite: 8] |
| **Adelia Ramadani** | 123140183 | [cite_start]Evaluasi Metrik & Tabel Perbandingan Performa [cite: 8] |
| **Rian Rafael Sangap Tamba** | 123140190 | [cite_start]Penyusunan Laporan Utama & Dokumentasi BAB IV [cite: 8] |
| **Nadine Aura Rahmadhani** | 123140195 | [cite_start]Studi Literatur & Validasi Metrik Imbalanced Data [cite: 8] |

---
[cite_start]**PROGRAM STUDI TEKNIK INFORMATIKA** **FAKULTAS TEKNOLOGI INDUSTRI** **INSTITUT TEKNOLOGI SUMATERA (ITERA)** **2026** [cite: 9, 10, 11, 12]
