import cv2
import csv
import os
import mediapipe as mp

VIDEOS = [
    ("videos/gerakan1.mp4", "gerakan_1.csv"),
    ("videos/gerakan2.mp4", "gerakan_2.csv"),
    ("videos/gerakan3.mp4", "gerakan_3.csv"),
    ("videos/gerakan4.mp4", "gerakan_4.csv"),
    ("videos/gerakan5.mp4", "gerakan_5.csv"),
]

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

for video_path, output_csv in VIDEOS:

    print()
    print("==============================")
    print("Memproses:", video_path)
    print("Output   :", output_csv)
    print("==============================")

    if not os.path.exists(video_path):
        print("❌ Video tidak ditemukan")
        continue

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Video gagal dibuka")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    detector = mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    )

    # Header CSV
    header = []

    for i in range(33):
        header.extend([
            f"x_{i}",
            f"y_{i}",
            f"z_{i}",
            f"visibility_{i}"
        ])

    with open(output_csv, "w", newline="") as file:

        writer = csv.writer(file)
        writer.writerow(header)

        frame_number = 0
        berhasil = 0

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

            if not result.pose_landmarks:
                continue

            landmarks = result.pose_landmarks[0]

            row = []

            for landmark in landmarks:

                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])

            writer.writerow(row)

            berhasil += 1

    cap.release()
    detector.close()

    print("Frame berhasil:", berhasil)

print()
print("==============================")
print("SEMUA CSV SELESAI")
print("==============================")