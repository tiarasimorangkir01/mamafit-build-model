import pandas as pd
import joblib
import glob
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# CONFIG
# ==========================================

MODEL_OUTPUT = "mama_fit_mirror.pkl"

FILES = [
    "gerakan_1.csv",
    "gerakan_1_mirror.csv",

    "gerakan_2.csv",
    "gerakan_2_mirror.csv",

    # PENTING:
    # kembali ke data asli, BUKAN updated
    "gerakan_3.csv",
    "gerakan_3_mirror.csv",

    "gerakan_4.csv",
    "gerakan_4_mirror.csv",

    "gerakan_5.csv",
    "gerakan_5_mirror.csv"
]


# ==========================================
# LOAD DATA
# ==========================================

print("==============================")
print("MEMUAT DATA TRAINING")
print("==============================")


dataframes = []

for filename in FILES:

    if not os.path.exists(filename):

        print("❌ File tidak ditemukan:", filename)
        continue

    print("Membaca:", filename)

    df = pd.read_csv(filename)

    # Ambil label dari nama file
    if "gerakan_1" in filename:
        label = "gerakan_1"

    elif "gerakan_2" in filename:
        label = "gerakan_2"

    elif "gerakan_3" in filename:
        label = "gerakan_3"

    elif "gerakan_4" in filename:
        label = "gerakan_4"

    elif "gerakan_5" in filename:
        label = "gerakan_5"

    else:
        print("❌ Label tidak dikenal:", filename)
        continue

    df = df.copy()
    df["label"] = label

    dataframes.append(df)

    print(
        f"{filename:<30} {len(df)} frame"
    )


# ==========================================
# GABUNGKAN
# ==========================================

if not dataframes:

    print("❌ Tidak ada dataset.")
    exit()


dataset = pd.concat(
    dataframes,
    ignore_index=True
)


# ==========================================
# DATASET SEBELUM BALANCE
# ==========================================

print()
print("==============================")
print("DATASET SEBELUM BALANCE")
print("==============================")

print(
    dataset["label"].value_counts()
)


# ==========================================
# BALANCE DATASET
# ==========================================

# Ambil jumlah frame paling sedikit
# dari semua kelas

class_counts = dataset["label"].value_counts()

target_count = class_counts.min()

print()
print("==============================")
print("BALANCING DATASET")
print("==============================")

print(
    "Target frame per kelas:",
    target_count
)


balanced_parts = []

for label in sorted(
    dataset["label"].unique()
):

    class_df = dataset[
        dataset["label"] == label
    ]

    # Sampling tanpa replacement
    # supaya setiap kelas memiliki
    # jumlah frame yang sama

    class_df = class_df.sample(
        n=target_count,
        random_state=42
    )

    balanced_parts.append(
        class_df
    )


dataset = pd.concat(
    balanced_parts,
    ignore_index=True
)


# Acak seluruh dataset
dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# HASIL BALANCE
# ==========================================

print()
print("==============================")
print("DATASET SETELAH BALANCE")
print("==============================")

print(
    dataset["label"].value_counts()
)


# ==========================================
# FEATURES
# ==========================================

X = dataset.drop(
    columns=["label"]
)

y = dataset["label"]


print()
print("Total frame :", len(X))
print("Total fitur :", len(X.columns))


# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print()
print("==============================")
print("TRAIN / VALIDATION")
print("==============================")

print("Training   :", len(X_train))
print("Validation :", len(X_test))


# ==========================================
# RANDOM FOREST
# ==========================================

print()
print("==============================")
print("SEDANG TRAINING...")
print("==============================")


model = RandomForestClassifier(
    n_estimators=400,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    max_features="sqrt",

    class_weight="balanced_subsample",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ==========================================
# VALIDATION
# ==========================================

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print()
print("==============================")
print("HASIL VALIDATION")
print("==============================")

print(
    f"Akurasi: {accuracy * 100:.2f}%"
)

print()

print(
    classification_report(
        y_test,
        y_pred,
        digits=2
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

from sklearn.metrics import confusion_matrix

labels = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


print("==============================")
print("CONFUSION MATRIX")
print("==============================")

print(
    "              ",
    " ".join(
        f"{x:>12}"
        for x in labels
    )
)

for label, row in zip(
    labels,
    cm
):

    print(
        f"{label:<12}",
        " ".join(
            f"{value:>12}"
            for value in row
        )
    )


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_OUTPUT
)


print()
print("==============================")
print("MODEL FINAL SELESAI")
print("==============================")

print(
    "File:",
    MODEL_OUTPUT
)

print(
    "Jumlah fitur:",
    len(model.feature_names_in_)
)

print(
    "Kelas:",
    list(model.classes_)
)

print("==============================")