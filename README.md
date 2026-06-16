# Analisis BERT untuk Word Sense Disambiguation (WSD) Menggunakan Dataset SemCor-13

## Deskripsi Proyek

Proyek ini merupakan pengembangan dari repository **BERT Disambiguation** milik Daniel Loureiro yang digunakan untuk melakukan penelitian pada tugas **Word Sense Disambiguation (WSD)**.

Pada proyek ini dilakukan adaptasi sistem agar dapat berjalan pada dataset **SemCor-13**, kemudian dilakukan ekstraksi representasi makna kata menggunakan model **BERT Base Uncased** dan evaluasi menggunakan metode **1-Nearest Neighbor (1NN)**.

Word Sense Disambiguation (WSD) adalah tugas dalam Natural Language Processing (NLP) yang bertujuan menentukan makna yang benar dari sebuah kata ambigu berdasarkan konteks kalimat.

Contoh:

> "He sat on the bank of the river"

Kata **bank** berarti tepi sungai.

> "She deposited money in the bank"

Kata **bank** berarti institusi keuangan.

Sistem WSD bertugas menentukan makna yang tepat berdasarkan konteks kalimat.

---

# Referensi Paper

Proyek ini mengacu pada paper:

**Analysis and Evaluation of Language Models for Word Sense Disambiguation**

Penulis:

* Daniel Loureiro
* Kiamehr Rezaee
* Mohammad Taher Pilehvar
* Jose Camacho-Collados

Link Paper:

[Analysis and Evaluation of Language Models for Word Sense Disambiguation (MIT Press)](https://direct.mit.edu/coli/article/47/2/387/98520/Analysis-and-Evaluation-of-Language-Models-for?utm_source=chatgpt.com)

Versi ArXiv:

[ArXiv Paper (2008.11608)](https://arxiv.org/abs/2008.11608?utm_source=chatgpt.com)

Paper ini menunjukkan bahwa model bahasa seperti BERT mampu merepresentasikan perbedaan makna kata dengan baik dan metode feature extraction menggunakan embedding BERT cukup efektif untuk tugas Word Sense Disambiguation. ([MIT Press Direct][1])

---

# Tujuan Proyek

Tujuan proyek ini adalah:

1. Memahami konsep Word Sense Disambiguation (WSD)
2. Menggunakan embedding BERT untuk representasi makna kata
3. Membuat sense vector dari dataset SemCor-13
4. Melakukan klasifikasi makna kata menggunakan metode Nearest Neighbor
5. Mengevaluasi performa model pada data uji

---

# Dataset

Dataset yang digunakan adalah:

## SemCor-13

Dataset berisi kalimat-kalimat yang telah dianotasi dengan sense WordNet.

Kata ambigu yang digunakan:

* case
* face
* form
* head
* interest
* life
* light
* matter
* point
* state
* time
* way
* work

Struktur dataset:

```text
data/
└── SemCor-13/
    ├── case/
    ├── face/
    ├── form/
    ├── head/
    ├── ...
```

Setiap folder berisi:

```text
train.data.txt
train.gold.txt

test.data.txt
test.gold.txt

classes_map.txt
```

---

# Arsitektur Sistem

Sistem terdiri dari dua tahap utama.

## Tahap 1 – Pembuatan Sense Vector

File:

```bash
create_1nn_vecs.py
```

Proses:

1. Membaca data training
2. Mengambil embedding BERT untuk kata target
3. Menghitung rata-rata embedding setiap sense
4. Menyimpan hasil sebagai sense vector

Ilustrasi:

```text
Kalimat Training
        ↓
     BERT
        ↓
 Contextual Embedding
        ↓
 Rata-rata per Sense
        ↓
 Sense Vector
```

Output:

```text
vectors/semcor13_test.txt
```

---

## Tahap 2 – Evaluasi 1NN

File:

```bash
eval_1nn.py
```

Proses:

1. Membaca data test
2. Mengambil embedding kata target
3. Membandingkan embedding dengan seluruh sense vector
4. Memilih sense dengan cosine similarity tertinggi
5. Menghitung akurasi

Ilustrasi:

```text
Kalimat Test
      ↓
    BERT
      ↓
Embedding Kata
      ↓
Cosine Similarity
      ↓
Sense Vector Terdekat
      ↓
Prediksi Sense
```

---

# Modifikasi yang Dilakukan

Repository asli hanya mendukung:

```text
CoarseWSD-20
```

Pada proyek ini dilakukan beberapa modifikasi:

## 1. Menambahkan Reader SemCor-13

File:

```python
semcor13_reader.py
```

Fungsi:

* membaca data SemCor-13
* membaca label sense
* membentuk instance train dan test

---

## 2. Memodifikasi create_1nn_vecs.py

Perubahan:

* menambahkan dukungan dataset SemCor-13
* menyesuaikan format pembacaan data
* menangani variasi bentuk kata

Contoh:

```text
work
works
worked
working
```

---

## 3. Memodifikasi eval_1nn.py

Perubahan:

* mengganti reader CoarseWSD menjadi SemCor-13
* memperbaiki proses evaluasi
* menangani prediksi kosong
* membuat output summary otomatis

---

## 4. Memodifikasi nlm_encoder.py

Perubahan:

* kompatibilitas dengan Transformers terbaru
* penanganan tokenisasi subword
* perbaikan loading model BERT

---

# Cara Menjalankan Proyek

## 1. Aktifkan Environment

```bash
source .venv/bin/activate
```

---

## 2. Membuat Sense Vector

```bash
python create_1nn_vecs.py \
-dataset_id SemCor-13 \
-out_path vectors/semcor13_test.txt
```

Output:

```text
vectors/semcor13_test.txt
```

---

## 3. Menjalankan Evaluasi

```bash
python eval_1nn.py \
-dataset_id SemCor-13 \
-sv_path vectors/semcor13_test.txt
```

Output:

```text
results/SemCor-13/1nn/bert-base-uncased/
```

---

# Hasil Evaluasi

File:

```text
results/SemCor-13/1nn/bert-base-uncased/summary.csv
```

Contoh hasil:

| Kata  | Akurasi |
| ----- | ------- |
| head  | 96.67%  |
| point | 100.00% |
| state | 86.27%  |
| light | 83.33%  |
| life  | 80.56%  |
| time  | 72.86%  |
| work  | 69.33%  |
| face  | 38.10%  |
| way   | 9.76%   |

Hasil menunjukkan bahwa beberapa kata memiliki sense yang sangat mudah dibedakan (misalnya *point* dan *head*), sedangkan kata seperti *way* memiliki tingkat ambiguitas yang tinggi sehingga lebih sulit didisambiguasi.

---

# Struktur Folder

```text
bert-disambiguation-semcor/
│
├── data/
│   └── SemCor-13/
│
├── vectors/
│   └── semcor13_test.txt
│
├── results/
│   └── SemCor-13/
│
├── create_1nn_vecs.py
├── eval_1nn.py
├── semcor13_reader.py
├── nlm_encoder.py
│
└── README.md
```

---

# Kesimpulan

Pada proyek ini berhasil dilakukan:

1. Adaptasi repository BERT Disambiguation untuk dataset SemCor-13

2. Pembuatan sense vector menggunakan embedding BERT

3. Implementasi klasifikasi Word Sense Disambiguation menggunakan metode 1NN

4. Evaluasi pada 13 kata ambigu dalam dataset SemCor-13

5. Penyimpanan hasil evaluasi dalam format CSV dan JSONL

Hasil menunjukkan bahwa representasi kontekstual dari BERT mampu menangkap perbedaan makna kata dengan baik pada sebagian besar kasus Word Sense Disambiguation. 
