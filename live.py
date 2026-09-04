import cv2
import joblib
import mediapipe as mp
import pandas as pd
import warnings

from collections import Counter, deque


# ==========================================
# HILANGKAN WARNING YANG TIDAK PENTING
# ==========================================

warnings.filterwarnings(
    "ignore",
    message=".*X does not have valid feature names.*"
)

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*"
)

warnings.filterwarnings(
    "ignore",
    message=".*Feedback manager requires.*"
)

warnings.filterwarnings(
    "ignore",
    message=".*Using NORM_RECT without IMAGE_DIMENSIONS.*"
)


# ==========================================
# CONFIG
# ==========================================

MODEL = "mama_fit_mirror.pkl"

WINDOW_SIZE = 30

MIN_VISIBILITY = 0.20


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(MODEL)

# Penting:
# Live tidak perlu menggunakan banyak CPU thread
model.n_jobs = 1

feature_names = model.feature_names_in_


print("==============================")
print("MAMA FIT LIVE")
print("==============================")
print("Model:", MODEL)
print("Tekan Q untuk keluar")
print("==============================")


# ==========================================
# MEDIAPIPE
# ==========================================

options = mp.tasks.vision.PoseLandmarkerOptions(

    base_options=mp.tasks.BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    ),

    running_mode=mp.tasks.vision.RunningMode.VIDEO,

    min_pose_detection_confidence=0.3,

    min_pose_presence_confidence=0.3,

    min_tracking_confidence=0.3
)


detector = (
    mp.tasks.vision.PoseLandmarker
    .create_from_options(options)
)


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Webcam tidak dapat dibuka")

    detector.close()

    exit()


# ==========================================
# VARIABLES
# ==========================================

frame_number = 0

prediction_window = deque(
    maxlen=WINDOW_SIZE
)


# ==========================================
# LIVE LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame_number += 1


    # ======================================
    # MIRROR WEBCAM
    # ======================================

    frame = cv2.flip(
        frame,
        1
    )


    # ======================================
    # CONVERT IMAGE
    # ======================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # ======================================
    # TIMESTAMP
    # ======================================

    timestamp_ms = int(
        frame_number * 33.333
    )


    # ======================================
    # MEDIAPIPE DETECTION
    # ======================================

    result = detector.detect_for_video(
        image,
        timestamp_ms
    )


    current_prediction = "NO POSE"

    pose_status = "NO POSE"


    # ======================================
    # POSE TERDETEKSI
    # ======================================

    if result.pose_landmarks:

        pose_status = "POSE TERDETEKSI"


        landmarks = result.pose_landmarks[0]


        # ==================================
        # VISIBILITY
        # ==================================

        visibility_values = [
            lm.visibility
            for lm in landmarks
        ]


        if visibility_values:

            avg_visibility = (
                sum(visibility_values)
                /
                len(visibility_values)
            )

        else:

            avg_visibility = 0


        # ==================================
        # FEATURE EXTRACTION
        # ==================================

        if avg_visibility >= MIN_VISIBILITY:

            features = []


            for lm in landmarks:

                features.extend([
                    lm.x,
                    lm.y,
                    lm.z,
                    lm.visibility
                ])


            # ==================================
            # PASTIKAN 132 FITUR
            # ==================================

            if len(features) == len(feature_names):


                # ==================================
                # DATAFRAME
                # ==================================

                features_df = pd.DataFrame(
                    [features],
                    columns=feature_names
                )


                # ==================================
                # PREDICT
                # ==================================

                current_prediction = model.predict(
                    features_df
                )[0]


                prediction_window.append(
                    current_prediction
                )


    # ==========================================
    # MAJORITY VOTING
    # ==========================================

    if prediction_window:

        counts = Counter(
            prediction_window
        )


        stable_prediction = (
            counts.most_common(1)[0][0]
        )


        stable_percentage = (
            counts.most_common(1)[0][1]
            /
            len(prediction_window)
        ) * 100

    else:

        stable_prediction = "MENUNGGU..."

        stable_percentage = 0


    # ==========================================
    # DISPLAY
    # ==========================================

    cv2.putText(
        frame,
        "MAMA FIT",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Frame: {current_prediction}",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"GERAKAN: {stable_prediction}",
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {stable_percentage:.1f}%",
        (30, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        pose_status,
        (30, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ==========================================
    # SHOW WINDOW
    # ==========================================

    cv2.imshow(
        "Mama Fit - Live",
        frame
    )


    # ==========================================
    # EXIT
    # ==========================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

detector.close()

cv2.destroyAllWindows()


print()

print("==============================")
print("LIVE SELESAI")
print("==============================")