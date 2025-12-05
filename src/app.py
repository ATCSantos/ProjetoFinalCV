from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from dataclasses import dataclass
import mediapipe as mp

# ----------------------------
# Configuração
# ----------------------------

CAMERA_INDEX = 0
WINDOW_NAME = "Fruit Arcade"
FLIP_CAMERA = True

# caminho robusto (independente de onde corres o python)
ASSETS_DIR = (Path(__file__).resolve().parent.parent / "assets")

# ----------------------------
# Utilitários
# ----------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def now_s() -> float:
    return time.perf_counter()

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def draw_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.7, thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (org[0] + 2, org[1] + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ----------------------------
# PNG BGRA + overlay alpha
# ----------------------------

def load_png_rgba(path: str) -> Optional[np.ndarray]:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return im

def overlay_bgra(dst_bgr: np.ndarray, src_bgra: np.ndarray, center_xy: Tuple[int, int], size: Tuple[int, int]) -> None:
    if src_bgra is None:
        return

    w, h = size
    if w <= 1 or h <= 1:
        return

    src = cv2.resize(src_bgra, (w, h), interpolation=cv2.INTER_AREA)

    cx, cy = center_xy
    x1 = int(cx - w // 2)
    y1 = int(cy - h // 2)
    x2 = x1 + w
    y2 = y1 + h

    H, W = dst_bgr.shape[:2]

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = w - max(0, x2 - W)
    sy2 = h - max(0, y2 - H)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    if x1 >= x2 or y1 >= y2 or sx1 >= sx2 or sy1 >= sy2:
        return

    roi = dst_bgr[y1:y2, x1:x2]
    src_roi = src[sy1:sy2, sx1:sx2]

    alpha = (src_roi[:, :, 3:4].astype(np.float32) / 255.0)
    src_rgb = src_roi[:, :, :3].astype(np.float32)
    roi_f = roi.astype(np.float32)

    out = src_rgb * alpha + roi_f * (1.0 - alpha)
    roi[:, :] = out.astype(np.uint8)


EMA_ALPHA = 0.25

@dataclass
class FaceData:
    nose: Tuple[float, float]
    mouth_open: float
    has_face: bool

class FaceTracker:
    NOSE_IDX = 1
    UPPER_LIP_IDX = 13
    LOWER_LIP_IDX = 14
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    def __init__(self) -> None:
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._nose_ema: Optional[Tuple[float, float]] = None
        self._mouth_ema: Optional[float] = None

    def process(self, frame_bgr: np.ndarray) -> FaceData:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.mesh.process(rgb)
        rgb.flags.writeable = True

        if not res.multi_face_landmarks:
            self._nose_ema = None
            self._mouth_ema = None
            return FaceData((0.5, 0.5), 0.0, False)

        lm = res.multi_face_landmarks[0].landmark
        nose = (lm[self.NOSE_IDX].x, lm[self.NOSE_IDX].y)

        eyeL = np.array([lm[self.LEFT_EYE_OUTER].x, lm[self.LEFT_EYE_OUTER].y], dtype=np.float32)
        eyeR = np.array([lm[self.RIGHT_EYE_OUTER].x, lm[self.RIGHT_EYE_OUTER].y], dtype=np.float32)
        eye_dist = float(np.linalg.norm(eyeL - eyeR)) + 1e-6

        upper = np.array([lm[self.UPPER_LIP_IDX].x, lm[self.UPPER_LIP_IDX].y], dtype=np.float32)
        lower = np.array([lm[self.LOWER_LIP_IDX].x, lm[self.LOWER_LIP_IDX].y], dtype=np.float32)
        mouth_open = float(np.linalg.norm(upper - lower)) / eye_dist

        if self._nose_ema is None:
            self._nose_ema = nose
            self._mouth_ema = mouth_open
        else:
            self._nose_ema = (
                lerp(self._nose_ema[0], nose[0], EMA_ALPHA),
                lerp(self._nose_ema[1], nose[1], EMA_ALPHA),
            )
            self._mouth_ema = lerp(self._mouth_ema, mouth_open, EMA_ALPHA)

        return FaceData(self._nose_ema, float(self._mouth_ema), True)



class SpriteBank:
    def __init__(self) -> None:
        self.logo = load_png_rgba(str(ASSETS_DIR / "logo.png"))
        self.fruits = {
            "orange": load_png_rgba(str(ASSETS_DIR / "orange.png")),
            "apple": load_png_rgba(str(ASSETS_DIR / "apple.png")),
            "watermelon": load_png_rgba(str(ASSETS_DIR / "watermelon.png")),
            "banana": load_png_rgba(str(ASSETS_DIR / "banana.png")),
        }
        self.fruit_keys = list(self.fruits.keys())

    def draw_logo(self, frame: np.ndarray) -> None:
        if self.logo is None:
            return
        h, w = frame.shape[:2]
        target_w = int(w * 0.45)
        target_h = int(target_w * (self.logo.shape[0] / self.logo.shape[1]))
        overlay_bgra(frame, self.logo, (w // 2, int(h * 0.12)), (target_w, target_h))

    def draw_fruit(self, frame: np.ndarray, kind: str, center: Tuple[int, int], size_px: int) -> None:
        sprite = self.fruits.get(kind)
        if sprite is None:
            cv2.circle(frame, center, size_px // 2, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, center, size_px // 2, (255, 255, 255), 2, cv2.LINE_AA)
            return
        overlay_bgra(frame, sprite, center, (size_px, size_px))

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Não foi possível abrir a câmara (ajusta CAMERA_INDEX).")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # logo = load_png_rgba(str(ASSETS_DIR / "logo.png"))
    bank = SpriteBank()
    logo = bank.logo

    face_tracker = FaceTracker()

    t0 = now_s()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        face = face_tracker.process(frame)

        h, w = frame.shape[:2]

        if logo is not None:
            target_w = int(w * 0.42)
            target_h = int(target_w * (logo.shape[0] / logo.shape[1]))
            overlay_bgra(frame, logo, (w // 2, int(h * 0.16)), (target_w, target_h))
        else:
            # PLACEHOLDER
            x1, y1 = w // 2 - 180, int(h * 0.06)
            x2, y2 = w // 2 + 180, int(h * 0.22)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            draw_text(frame, "LOGO EM FALTA", (w // 2 - 135, int(h * 0.16)), 0.8, 2)

        if face.has_face:
            cx = int(clamp(face.nose[0], 0.0, 1.0) * w)
            cy = int(clamp(face.nose[1], 0.0, 1.0) * h)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 2)

        #texto por cima
        draw_text(frame, "Fruit Arcade", (20, 40), 0.8, 2)
        draw_text(frame, f"Uptime: {now_s() - t0:0.1f}s", (20, 75), 0.7, 2)
        draw_text(frame, f"mouth: {face.mouth_open:0.3f}", (20, 110), 0.65, 2)
        draw_text(frame, "ESC: sair", (20, h - 20), 0.65, 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
