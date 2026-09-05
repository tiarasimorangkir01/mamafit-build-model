import pandas as pd
import os


FILES = {
    "gerakan_1": "gerakan_1.csv",
    "gerakan_2": "gerakan_2.csv",
    "gerakan_3": "gerakan_3.csv",
    "gerakan_4": "gerakan_4.csv",
    "gerakan_5": "gerakan_5.csv"
}


def mirror_pose(df):

    mirrored = df.copy()

    # MediaPipe:
    # x dibalik
    # visibility tetap
    # y dan z tetap

    for i in range(33):

        x_col = f"x_{i}"

        if x_col in mirrored.columns:
            mirrored[x_col] = 1.0 - mirrored[x_col]

    # Tukar landmark kiri-kanan
    # pasangan landmark MediaPipe Pose

    swap_pairs = [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
        (17, 18),
        (19, 20),
        (21, 22),
        (23, 24),
        (25, 26),
        (27, 28),
        (29, 30),
        (31, 32)
    ]

    for left, right in swap_pairs:

        for prefix in ["x", "y", "z", "visibility"]:

            left_col = f"{prefix}_{left}"
            right_col = f"{prefix}_{right}"

            if left_col in mirrored.columns and right_col in mirrored.columns:

                temp = mirrored[left_col].copy()

                mirrored[left_col] = mirrored[right_col]
                mirrored[right_col] = temp

    return mirrored


print("==============================")
print("MIRROR AUGMENTATION")
print("==============================")


for label, filename in FILES.items():

    print()
    print("Memproses:", filename)

    if not os.path.exists(filename):

        print("❌ File tidak ditemukan")
        continue

    df = pd.read_csv(filename)

    mirrored = mirror_pose(df)

    output = f"{label}_mirror.csv"

    mirrored.to_csv(
        output,
        index=False
    )

    print("Data asli :", len(df))
    print("Data mirror:", len(mirrored))
    print("Disimpan  :", output)


print()
print("==============================")
print("SEMUA MIRROR SELESAI")
print("==============================")