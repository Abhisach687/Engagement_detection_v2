import cv2
from pathlib import Path
from typing import List


def select_top_frames(video_folder: Path, num_frames: int = 30) -> List[Path]:
    """
    Rank frames by face-region sharpness and return the best `num_frames`.
    Falls back to full frame if no face is detected.
    """
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    frames = sorted(video_folder.glob("frame_*.jpg"))
    quality = []

    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)

        region = gray
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            region = gray[y : y + h, x : x + w]

        sharpness = cv2.Laplacian(region, cv2.CV_64F).var()
        quality.append((fp, sharpness))

    quality.sort(key=lambda x: x[1], reverse=True)
    return [fp for fp, _ in quality[:num_frames]]
