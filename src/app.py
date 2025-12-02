import cv2
import time
from typing import Tuple
import numpy as np

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def now_s() -> float:
    return time.perf_counter()

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def draw_text(img, text: str, org: Tuple[int, int], scale: float = 0.7, thickness: int = 2) -> None:
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (org[0] + 2, org[1] + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

CAMERA_INDEX = 0
WINDOW_NAME = "Fruit Arcade"
FLIP_CAMERA = True


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Não foi possível abrir a câmara (ajusta CAMERA_INDEX).")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        draw_text(frame, "Fruit Arcade (TESTE) ç º ã ó ", (20, 40), 0.9, 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
