"""
pip install -r requirements-convert.txt
python convert_to_onnx.py

Run it from the repository root.

HASIL DARI FILE:
-> mamafit.onnx -> Model file for Android app
-> model_meta.json -> Metadata file for the ONNX model (Ini penting untuk android app agar bisa tahu urutan output modelnya)


"""

import json
import os

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# --------------------------------------------------------------------------
# Paths. Note these are relative to the repository root, not to the location of
# this file - run the script from the repository root.
# --------------------------------------------------------------------------
PKL_PATH = os.path.join("model", "mama_fit_mirror.pkl")
ONNX_DIR = os.path.join("android", "pose", "src", "main", "assets")
ONNX_PATH = os.path.join(ONNX_DIR, "mamafit.onnx")
META_PATH = os.path.join(ONNX_DIR, "model_meta.json")

EXPECTED_FEATURES = 132
EXPECTED_CLASSES = 5


def load_model():
    """Load the trained scikit-learn model and sanity-check its shape."""
    print(f"Loading {PKL_PATH} ...")
    model = joblib.load(PKL_PATH)

    n_features = int(model.n_features_in_)
    classes = [str(c) for c in model.classes_]

    print(f"  estimator     : {type(model).__name__}")
    print(f"  n_estimators  : {getattr(model, 'n_estimators', 'n/a')}")
    print(f"  features      : {n_features}")
    print(f"  classes       : {classes}")

    # Fail loudly rather than silently producing a model the app will misuse.
    if n_features != EXPECTED_FEATURES:
        raise SystemExit(
            f"Expected {EXPECTED_FEATURES} features but the model has {n_features}. "
            "If you retrained with a different feature set, update EXPECTED_FEATURES "
        )
    if len(classes) != EXPECTED_CLASSES:
        raise SystemExit(
            f"Expected {EXPECTED_CLASSES} classes but the model has {len(classes)}: {classes}. "
            "If you added or removed a movement, update EXPECTED_CLASSES here."
        )

    return model


def convert(model):
    """Convert the scikit-learn model to ONNX."""
    initial_types = [("float_input", FloatTensorType([None, EXPECTED_FEATURES]))]
    options = {id(model): {"zipmap": False}}

    print("Converting to ONNX ...")
    onx = convert_sklearn(model, initial_types=initial_types, options=options)

    os.makedirs(ONNX_DIR, exist_ok=True)
    with open(ONNX_PATH, "wb") as f:
        f.write(onx.SerializeToString())

    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"  wrote {ONNX_PATH} ({size_mb:.2f} MB)")
    return onx


def write_meta(model):
    """Record the input contract so the Android side can be checked against it."""
    feature_names = [str(name) for name in model.feature_names_in_]
    meta = {
        "n_features": int(model.n_features_in_),
        "feature_names": feature_names,
        "classes": [str(c) for c in model.classes_],
        "input_name": "float_input",
        "notes": (
            "Features are 33 MediaPipe pose landmarks as x, y, z, visibility, "
            "interleaved per landmark. No scaling or normalisation is applied."
        ),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {META_PATH}")
    print(f"  first 8 features: {feature_names[:8]}")


def main():
    model = load_model()
    convert(model)
    write_meta(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
