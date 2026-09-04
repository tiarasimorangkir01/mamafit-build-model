import pandas as pd

import warnings

warnings.filterwarnings(
    "ignore",
    message = "`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`"
    )
    
import cv2
import joblib
import mediapipe as mp
from collections import Counter, deque

MODEL = "mama_fit_mirror.pkl"
VIDEO = "video-lama/gerakan1_benar.mp4"

WINDOW_SIZE = 30
MIN_VISIBILITY = 0.20


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(MODEL)

print("==============================")
print("MODEL :", MODEL)
print("VIDEO :", VIDEO)
print("==============================")


# ==========================================
# MEDIAPIPE
# ==========================================

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = mp.tasks.vision.PoseLandmarker.create_from_options(
    options
)


# ==========================================
# VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("❌ Video gagal dibuka")
    detector.close()
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


# ==========================================
# VARIABLES
# ==========================================

frame_number = 0

predictions = []

window = deque(
    maxlen=WINDOW_SIZE
)


# ==========================================
# PROCESS
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = int(
        frame_number / fps * 1000
    )

    result = detector.detect_for_video(
        image,
        timestamp_ms
    )

    current = "NO POSE"

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        visibility = [
            lm.visibility
            for lm in landmarks
        ]

        avg_visibility = (
            sum(visibility) /
            len(visibility)
        )

        if avg_visibility >= MIN_VISIBILITY:

            features = []

            for lm in landmarks:

                features.extend([
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility
                ])

            features_df = pd.DataFrame(
                [features],
                columns=model.feature_names_in_
            )

            current = model.predict(
                features_df
            )[0]

            window.append(current)
            predictions.append(current)

    # ======================================
    # MAJORITY VOTING
    # ======================================

    if window:

        counter = Counter(window)

        stable = counter.most_common(1)[0][0]

        stable_percent = (
            counter[stable] /
            len(window)
        ) * 100

    else:

        stable = "NO POSE"
        stable_percent = 0


    # ======================================
    # DISPLAY
    # ======================================

    cv2.putText(
        frame,
        f"Prediction: {current}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"STABLE: {stable}",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Voting: {stable_percent:.1f}%",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "MamaFit Mirror Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
detector.close()
cv2.destroyAllWindows()


# ==========================================
# FINAL RESULT
# ==========================================

print()
print("==============================")
print("HASIL TEST MIRROR")
print("==============================")

if predictions:

    counter = Counter(predictions)

    total = len(predictions)

    for label, count in counter.most_common():

        percent = (
            count / total
        ) * 100

        print(
            f"{label}: {percent:.2f}%"
        )

    final_label = counter.most_common(1)[0][0]

    final_percent = (
        counter[final_label] /
        total
    ) * 100

    print()
    print("==============================")
    print("PREDIKSI AKHIR")
    print("==============================")
    print(
        f"{final_label}: {final_percent:.2f}%"
    )

else:

    print("Tidak ada pose terdeteksi.")

print("==============================")