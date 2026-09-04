\# MAMAFIT



MAMAFIT adalah sistem AI untuk mengenali 5 jenis gerakan secara real-time menggunakan kamera.



\## Teknologi

\- Python

\- MediaPipe Pose

\- Random Forest

\- OpenCV



\## Alur AI

Kamera → MediaPipe Pose → Ekstraksi fitur → Random Forest → Prediksi gerakan

## Cara Kerja AI

1. Kamera menangkap gambar tubuh.
2. MediaPipe Pose mendeteksi 33 landmark tubuh.
3. Landmark diubah menjadi data fitur.
4. Random Forest memproses fitur tersebut.
5. Sistem menghasilkan prediksi dari 5 gerakan.

## Model

Model yang digunakan adalah Random Forest Classifier.

File model:
`model/mama_fit_mirror.pkl`
