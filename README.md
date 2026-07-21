# Prediksi Genre Musik Menggunakan Algoritma Support Vector Machine (SVM)

Tugas Besar Mata Kuliah Rekayasa Fitur - IF 10
Pramesa Alyusufi - 10123392
Aldira Fawwaz Rahadian - 10123402
Annangga Triwanalingga Heriawan - 101223418
Universitas Komputer Indonesia (UNIKOM)

## Deskripsi

Proyek ini membangun sistem prediksi genre musik otomatis berbasis fitur audio menggunakan algoritma **Support Vector Machine (SVM)**. Seluruh tahapan rekayasa fitur diterapkan secara menyeluruh, mulai dari pra-pemrosesan sinyal audio, ekstraksi fitur akustik, hingga pemodelan dan evaluasi.

Dataset yang digunakan adalah subset **fma_small** dari [Free Music Archive (FMA)](https://github.com/mdeff/fma), terdiri dari 8.000 klip audio berdurasi 30 detik yang terbagi rata ke dalam 8 genre: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, dan Rock.

## Struktur Repository

```
├── 01_data_prep.ipynb
├── 02_data_preprocessing.ipynb
├── 03_ekstraksi_fitur.ipynb
├── 04_training_model.ipynb
├── 05_prediksi_lagu.ipynb
├── data/
│   ├── best_model.pkl        # model SVM terlatih (kernel RBF)
│   ├── scaler.pkl            # StandardScaler yang digunakan saat training
│   └── model_info.pkl        # metadata model (tipe model, kebutuhan scaling)
└── README.md
```

> **Catatan:** File `features.csv` (hasil ekstraksi fitur, 75.296 baris x 39 kolom fitur) tidak disertakan di repository ini karena ukurannya melebihi batas upload GitHub (100MB). File ini dapat dihasilkan ulang dengan menjalankan notebook `03_ekstraksi_fitur.ipynb` secara lengkap (estimasi waktu proses: ±3-4 jam tergantung spesifikasi komputer), setelah dataset FMA diunduh sesuai instruksi di bawah.

## Deskripsi Tiap Notebook

| Notebook | Deskripsi |
|---|---|
| `01_data_prep.ipynb` | Memuat metadata (`tracks.csv`), memfilter subset `small`, dan memetakan `track_id` ke path audio serta label genre. |
| `02_data_preprocessing.ipynb` | Validasi berkas audio, penyeragaman sample rate, normalisasi amplitudo, dan segmentasi klip 3 detik. |
| `03_ekstraksi_fitur.ipynb` | Ekstraksi 39 fitur akustik (MFCC, chroma, fitur spektral, ZCR, RMS, tempo) dari setiap klip menggunakan `librosa`. |
| `04_training_model.ipynb` | Pembagian data latih/uji, standardisasi fitur, pelatihan dan penyetelan hyperparameter model SVM (GridSearchCV), serta evaluasi. |
| `05_prediksi_lagu.ipynb` | Pipeline prediksi end-to-end untuk memprediksi genre dari berkas MP3 baru. |

## Cara Menjalankan

### 1. Unduh dataset FMA

Dataset tidak disertakan di repository ini karena ukurannya besar (±7,5 GB). Unduh dari sumber resmi:

```bash
mkdir -p data
curl -o data/fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -o data/fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip
```

Ekstrak kedua berkas tersebut ke dalam folder `data/` (bisa menggunakan Python `zipfile`, 7-Zip, atau `unzip`).

Referensi dataset: [github.com/mdeff/fma](https://github.com/mdeff/fma)

### 2. Install dependencies

```bash
pip install librosa scikit-learn pandas numpy matplotlib seaborn joblib
```

### 3. Jalankan notebook secara berurutan

Jalankan notebook `01` hingga `05` sesuai urutan nomornya. Notebook `03_ekstraksi_fitur.ipynb` memerlukan waktu proses paling lama (±3-4 jam) karena mengekstraksi fitur dari 75.296 klip audio.

Jika hanya ingin mencoba pipeline prediksi tanpa mengulang seluruh proses training, cukup jalankan `05_prediksi_lagu.ipynb` langsung menggunakan model yang sudah tersedia di folder `data/` (`best_model.pkl` dan `scaler.pkl`).

### 4. (Alternatif) Jalankan di Google Colab

Notebook ini juga dapat dijalankan di Google Colab. Upload notebook beserta isi folder `data/` ke sesi Colab atau Google Drive yang di-mount, lalu sesuaikan path folder pada tiap notebook.

## Metodologi Singkat

1. **Pra-pemrosesan**: penyeragaman sample rate (22.050 Hz), normalisasi amplitudo, segmentasi klip 30 detik menjadi klip 3 detik (7.994 track valid → 75.296 klip).
2. **Ekstraksi fitur**: 39 fitur akustik per klip (13 MFCC × mean/std, chroma, spectral centroid, spectral bandwidth, spectral rolloff, zero-crossing rate, RMS energy, tempo).
3. **Pembagian data**: 80:20 (train:test) pada level track (bukan level klip) dan distratifikasi berdasarkan genre, untuk mencegah data leakage.
4. **Pemodelan**: SVM dengan kernel RBF, fitur distandardisasi menggunakan `StandardScaler`, hyperparameter (`C`) disetel menggunakan `GridSearchCV`.
5. **Evaluasi**: akurasi dihitung pada dua level — level klip (prediksi langsung per klip 3 detik) dan level lagu (voting mayoritas seluruh klip dari track yang sama).


## Lisensi Dataset

Dataset FMA disediakan oleh Defferrard dkk. (2017) di bawah lisensi Creative Commons. Lihat [repository resmi FMA](https://github.com/mdeff/fma) untuk detail lisensi tiap track.
