from __future__ import annotations
import time
import math
import random
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import mediapipe as mp

# ----------------------------
# Configuração
# ----------------------------

CAMERA_INDEX = 0
NOME_JANELA = "Fruit Arcade"
ESPELHAR_CAMARA = True

LARGURA_CAMARA = 640
ALTURA_CAMARA = 480
ESCALA_TRACKING = 0.5
TRACK_CADA_N_FRAMES = 2

PASTA_ASSETS = (Path(__file__).resolve().parent.parent / "assets")

MENU_DX = 0.07
MENU_DY = 0.07

MENU_DY_CONFIRM = 0.11
MENU_DY_EXIT = 0.13
MENU_DX_STABLE = 0.04
MENU_VERTICAL_RATIO = 1.8
MENU_REARM_DY = 0.03

HOLD_CONFIRM_S = 0.35
HOLD_EXIT_S = 0.60
CALIBRATION_S = 0.9

MENU_CONFIRM_BY_MOUTH = True
MENU_MOUTH_THRESHOLD = 0.055
MENU_MOUTH_HOLD_S = 0.22

CATCHER_ROUND_S = 45
FRUIT_FALL_SPEED_PX = 4.2
FRUIT_FALL_SPEED_INC = 0.02
FRUIT_SPAWN_EVERY_S = (0.45, 0.90)
BASKET_W = 160
BASKET_H = 30
MAX_FRUTAS_ECRA = 22

ENABLE_MOUTH_TO_CATCH = False
MOUTH_OPEN_THRESHOLD = 0.055

GUN_ROUND_TIMEOUT_S = 2.5
SIGNAL_DELAY_S = (1.0, 2.6)
TARGET_RADIUS = 60
SHOT_COOLDOWN_S = 0.7

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

    if src_bgra.shape[1] == w and src_bgra.shape[0] == h:
        src = src_bgra
    else:
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

    a = src_roi[:, :, 3].astype(np.uint16)
    if a.size == 0:
        return

    if a.min() == 0 and a.max() == 0:
        return

    src_rgb = src_roi[:, :, :3].astype(np.uint16)
    roi_u = roi.astype(np.uint16)

    if a.min() == 255:
        roi[:, :] = src_roi[:, :, :3]
        return

    inv = (255 - a).astype(np.uint16)
    out = (src_rgb * a[:, :, None] + roi_u * inv[:, :, None]) // 255
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
            refine_landmarks=False,
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


@dataclass
class HandData:
    tip: Tuple[float, float]
    label: str
    gesture: str

class HandTracker:
    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18
    THUMB_TIP = 4
    THUMB_IP = 3

    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    def _finger_extended(self, lm, tip_i: int, pip_i: int) -> bool:
        return lm[tip_i].y < lm[pip_i].y

    def _thumb_extended(self, lm, label: str) -> bool:
        if label == "Right":
            return lm[self.THUMB_TIP].x > lm[self.THUMB_IP].x
        return lm[self.THUMB_TIP].x < lm[self.THUMB_IP].x

    def _classify(self, lm, label: str) -> str:
        thumb = self._thumb_extended(lm, label)
        index_ = self._finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP)
        middle = self._finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring = self._finger_extended(lm, self.RING_TIP, self.RING_PIP)
        pinky = self._finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP)

        extended = [thumb, index_, middle, ring, pinky]
        cnt = sum(1 for v in extended if v)

        if cnt == 0:
            return "FIST"
        if cnt == 5:
            return "OPEN"
        if index_ and (not middle) and (not ring) and (not pinky) and (not thumb):
            return "POINT"
        if index_ and middle and (not ring) and (not pinky):
            return "PEACE"
        if thumb and (not index_) and (not middle) and (not ring) and (not pinky):
            return "THUMBS"
        return "OTHER"

    def process(self, frame_bgr: np.ndarray) -> Optional[HandData]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not res.multi_hand_landmarks or not res.multi_handedness:
            return None

        lm = res.multi_hand_landmarks[0].landmark
        label = res.multi_handedness[0].classification[0].label
        tip = (lm[self.INDEX_TIP].x, lm[self.INDEX_TIP].y)
        gesto = self._classify(lm, label)
        return HandData(tip=tip, label=label, gesture=gesto)



class SpriteBank:
    def __init__(self) -> None:
        self.logo = load_png_rgba(str(PASTA_ASSETS / "logo.png"))
        self.fruits = {
            "orange": load_png_rgba(str(PASTA_ASSETS / "orange.png")),
            "apple": load_png_rgba(str(PASTA_ASSETS / "apple.png")),
            "watermelon": load_png_rgba(str(PASTA_ASSETS / "watermelon.png")),
            "banana": load_png_rgba(str(PASTA_ASSETS / "banana.png")),
        }
        self.fruit_keys = list(self.fruits.keys())
        self._cache_fruit: dict[Tuple[str, int], np.ndarray] = {}
        self._cache_logo: dict[Tuple[int, int], np.ndarray] = {}

    def logo_scaled(self, size: Tuple[int, int]) -> Optional[np.ndarray]:
        if self.logo is None:
            return None
        w, h = size
        key = (w, h)
        if key not in self._cache_logo:
            self._cache_logo[key] = cv2.resize(self.logo, (w, h), interpolation=cv2.INTER_AREA)
        return self._cache_logo[key]

    def fruit_scaled(self, kind: str, size_px: int) -> Optional[np.ndarray]:
        sprite = self.fruits.get(kind)
        if sprite is None:
            return None
        key = (kind, size_px)
        if key not in self._cache_fruit:
            self._cache_fruit[key] = cv2.resize(sprite, (size_px, size_px), interpolation=cv2.INTER_AREA)
        return self._cache_fruit[key]

    def draw_fruit(self, frame: np.ndarray, kind: str, center: Tuple[int, int], size_px: int) -> None:
        sprite = self.fruit_scaled(kind, size_px)
        if sprite is None:
            cv2.circle(frame, center, size_px // 2, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, center, size_px // 2, (255, 255, 255), 2, cv2.LINE_AA)
            return
        overlay_bgra(frame, sprite, center, (size_px, size_px))


class Mode(Enum):
    MENU = auto()
    FRUIT = auto()
    GUN = auto()


@dataclass
class MenuState:
    baseline: Optional[Tuple[float, float]] = None
    calib_start: float = 0.0
    is_calibrating: bool = True
    sample_sum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    sample_count: int = 0

    selected: int = 0
    t_hold_confirm: Optional[float] = None
    t_hold_exit: Optional[float] = None
    t_mouth_confirm: Optional[float] = None
    armed: bool = True


def menu_reset(ms: MenuState) -> None:
    ms.baseline = None
    ms.calib_start = now_s()
    ms.is_calibrating = True
    ms.sample_sum[:] = 0.0
    ms.sample_count = 0
    ms.t_hold_confirm = None
    ms.t_hold_exit = None
    ms.t_mouth_confirm = None
    ms.armed = True


def menu_update(ms: MenuState, face: FaceData) -> None:
    if not face.has_face:
        return

    n = np.array(face.nose, dtype=np.float64)

    if ms.is_calibrating:
        ms.sample_sum += n
        ms.sample_count += 1
        if now_s() - ms.calib_start >= CALIBRATION_S and ms.sample_count > 0:
            ms.baseline = tuple((ms.sample_sum / ms.sample_count).tolist())
            ms.is_calibrating = False
            ms.armed = True
        return

    if ms.baseline is None:
        return

    bx, by = ms.baseline
    dx = n[0] - bx
    dy = n[1] - by
    t = now_s()

    if abs(dy) < MENU_REARM_DY:
        ms.armed = True

    if dx < -MENU_DX:
        ms.selected = 0
    elif dx > MENU_DX:
        ms.selected = 1

    vertical_ok = (abs(dx) <= MENU_DX_STABLE) and (abs(dy) > abs(dx) * MENU_VERTICAL_RATIO)

    want_confirm = vertical_ok and (dy < -MENU_DY_CONFIRM)
    if want_confirm:
        if ms.t_hold_confirm is None:
            if ms.armed:
                ms.t_hold_confirm = t
                ms.armed = False
    else:
        ms.t_hold_confirm = None

    want_exit = vertical_ok and (dy > MENU_DY_EXIT)
    if want_exit:
        if ms.t_hold_exit is None:
            if ms.armed:
                ms.t_hold_exit = t
                ms.armed = False
    else:
        ms.t_hold_exit = None

    if MENU_CONFIRM_BY_MOUTH and face.mouth_open >= MENU_MOUTH_THRESHOLD:
        if ms.t_mouth_confirm is None:
            if ms.armed:
                ms.t_mouth_confirm = t
                ms.armed = False
    else:
        ms.t_mouth_confirm = None


def menu_confirmed(ms: MenuState) -> bool:
    by_head = ms.t_hold_confirm is not None and (now_s() - ms.t_hold_confirm) >= HOLD_CONFIRM_S
    by_mouth = ms.t_mouth_confirm is not None and (now_s() - ms.t_mouth_confirm) >= MENU_MOUTH_HOLD_S
    return by_head or by_mouth


def menu_exit(ms: MenuState) -> bool:
    return ms.t_hold_exit is not None and (now_s() - ms.t_hold_exit) >= HOLD_EXIT_S


@dataclass
class Fruit:
    x: float
    y: float
    r: int
    kind: str
    vy: float


@dataclass
class CatcherState:
    start_t: float
    score: int = 0
    fruits: List[Fruit] = field(default_factory=list)
    next_spawn_t: float = 0.0
    speed: float = FRUIT_FALL_SPEED_PX
    mouth_required: bool = ENABLE_MOUTH_TO_CATCH


def spawn_fruit(cs: CatcherState, w: int, bank: SpriteBank) -> None:
    if len(cs.fruits) >= MAX_FRUTAS_ECRA:
        return
    r = random.randint(20, 32)
    x = random.randint(r + 10, w - r - 10)
    kind = random.choice(bank.fruit_keys) if bank.fruit_keys else "orange"
    cs.fruits.append(Fruit(x=float(x), y=float(-r), r=r, kind=kind, vy=cs.speed))
    cs.speed += FRUIT_FALL_SPEED_INC


def run_fruit_catcher(frame: np.ndarray, face: FaceData, cs: CatcherState, bank: SpriteBank) -> Tuple[np.ndarray, bool]:
    h, w = frame.shape[:2]
    t = now_s()
    remaining = max(0.0, CATCHER_ROUND_S - (t - cs.start_t))

    bx = int(clamp(face.nose[0], 0.0, 1.0) * w) if face.has_face else w // 2
    basket_x1 = int(clamp(bx - BASKET_W // 2, 0, w - BASKET_W))
    basket_y1 = h - 70
    basket_x2 = basket_x1 + BASKET_W
    basket_y2 = basket_y1 + BASKET_H

    basket_active = (not cs.mouth_required) or (face.mouth_open >= MOUTH_OPEN_THRESHOLD)

    if t >= cs.next_spawn_t:
        spawn_fruit(cs, w, bank)
        cs.next_spawn_t = t + random.uniform(*FRUIT_SPAWN_EVERY_S)

    new_fruits: List[Fruit] = []
    for f in cs.fruits:
        f.y += f.vy

        if basket_active:
            cx = clamp(f.x, basket_x1, basket_x2)
            cy = clamp(f.y, basket_y1, basket_y2)
            if (f.x - cx) ** 2 + (f.y - cy) ** 2 <= f.r ** 2:
                cs.score += 1
                continue

        if f.y - f.r > h + 10:
            continue

        new_fruits.append(f)
    cs.fruits = new_fruits

    for f in cs.fruits:
        bank.draw_fruit(frame, f.kind, (int(f.x), int(f.y)), size_px=f.r * 2)

    basket_col = (80, 255, 80) if basket_active else (80, 80, 255)
    cv2.rectangle(frame, (basket_x1, basket_y1), (basket_x2, basket_y2), basket_col, -1)
    cv2.rectangle(frame, (basket_x1, basket_y1), (basket_x2, basket_y2), (255, 255, 255), 2)

    draw_text(frame, f"FRUIT CATCHER  |  Score: {cs.score}", (20, 140), 0.9, 2)
    draw_text(frame, f"Tempo: {remaining:0.1f}s", (20, 175), 0.8, 2)

    done = remaining <= 0.0
    if done:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0)
        draw_text(frame, f"Fim! Pontuacao: {cs.score}", (w // 2 - 190, h // 2), 0.9, 2)

    return frame, done


@dataclass
class GunslingerState:
    phase: str
    score: int
    best_time: Optional[float]
    signal_t: float
    next_signal_t: float
    target: Tuple[int, int]
    target_kind: str
    last_shot_t: float
    last_reaction: Optional[float] = None


def gunslinger_new_target(w: int, h: int, bank: SpriteBank) -> Tuple[Tuple[int, int], float, str]:
    tx = random.randint(w // 3, int(w * 0.66))
    ty = random.randint(h // 3, int(h * 0.66))
    next_signal = now_s() + random.uniform(*SIGNAL_DELAY_S)
    kind = random.choice(bank.fruit_keys) if bank.fruit_keys else "orange"
    return (tx, ty), next_signal, kind


def run_gunslinger(frame: np.ndarray, mao: Optional[HandData], gs: GunslingerState, bank: SpriteBank) -> np.ndarray:
    h, w = frame.shape[:2]
    t = now_s()
    tx, ty = gs.target

    if gs.phase == "WAIT" and t >= gs.next_signal_t:
        gs.phase = "SIGNAL"
        gs.signal_t = t
        gs.last_reaction = None

    if gs.phase == "SIGNAL":
        if t - gs.signal_t >= GUN_ROUND_TIMEOUT_S:
            gs.phase = "RESULT"
            gs.last_shot_t = t
            gs.last_reaction = None

        if mao is not None and (t - gs.last_shot_t) >= SHOT_COOLDOWN_S:
            hx = int(clamp(mao.tip[0], 0.0, 1.0) * w)
            hy = int(clamp(mao.tip[1], 0.0, 1.0) * h)

            if mao.gesture in ("POINT", "PEACE", "OPEN") and (hx - tx) ** 2 + (hy - ty) ** 2 <= TARGET_RADIUS ** 2:
                rt = t - gs.signal_t
                gs.last_reaction = rt
                gs.best_time = rt if gs.best_time is None else min(gs.best_time, rt)
                gs.score += 1
                gs.phase = "RESULT"
                gs.last_shot_t = t

    if gs.phase == "RESULT" and (t - gs.last_shot_t) >= 1.2:
        gs.target, gs.next_signal_t, gs.target_kind = gunslinger_new_target(w, h, bank)
        gs.phase = "WAIT"

    if gs.phase == "WAIT":
        cv2.circle(frame, (tx, ty), TARGET_RADIUS, (140, 140, 140), -1, cv2.LINE_AA)
        cv2.circle(frame, (tx, ty), TARGET_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
        bank.draw_fruit(frame, gs.target_kind, (tx, ty), size_px=TARGET_RADIUS * 2 - 6)
        draw_text(frame, "GUNSLINGER  |  Espera pelo sinal...", (20, 140), 0.9, 2)

    elif gs.phase == "SIGNAL":
        pulse = 0.5 + 0.5 * math.sin((t - gs.signal_t) * 10.0)
        col = (int(80 + 175 * pulse), int(80 + 175 * pulse), 0)
        cv2.circle(frame, (tx, ty), TARGET_RADIUS, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (tx, ty), TARGET_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
        bank.draw_fruit(frame, gs.target_kind, (tx, ty), size_px=TARGET_RADIUS * 2 - 6)
        draw_text(frame, "AGORA! Toca na fruta com o indicador", (20, 140), 0.75, 2)

    else:
        cv2.circle(frame, (tx, ty), TARGET_RADIUS, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (tx, ty), TARGET_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
        bank.draw_fruit(frame, gs.target_kind, (tx, ty), size_px=TARGET_RADIUS * 2 - 6)
        if gs.last_reaction is not None:
            draw_text(frame, f"OK! {gs.last_reaction:0.3f}s", (20, 175), 0.85, 2)
        else:
            draw_text(frame, "Falhou (timeout)", (20, 175), 0.85, 2)

    if mao is not None:
        hx = int(clamp(mao.tip[0], 0.0, 1.0) * w)
        hy = int(clamp(mao.tip[1], 0.0, 1.0) * h)
        cv2.circle(frame, (hx, hy), 10, (255, 255, 255), -1)
        cv2.circle(frame, (hx, hy), 10, (0, 0, 0), 2)
        draw_text(frame, f"gesto: {mao.gesture}", (20, 210), 0.7, 2)

    draw_text(frame, f"Score: {gs.score}", (20, 245), 0.8, 2)
    if gs.best_time is not None:
        draw_text(frame, f"Melhor: {gs.best_time:0.3f}s", (20, 280), 0.8, 2)

    return frame


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Não foi possível abrir a câmara (ajusta CAMERA_INDEX).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA_CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_CAMARA)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)

    bank = SpriteBank()
    logo = bank.logo

    face_tracker = FaceTracker()
    hand_tracker = HandTracker()

    mode = Mode.MENU
    menu = MenuState()
    menu_reset(menu)

    catcher: Optional[CatcherState] = None
    guns: Optional[GunslingerState] = None

    t0 = now_s()
    contador_frames = 0
    ultimo_face = FaceData((0.5, 0.5), 0.0, False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if ESPELHAR_CAMARA:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        if ESCALA_TRACKING != 1.0:
            frame_track = cv2.resize(
                frame,
                (int(w * ESCALA_TRACKING), int(h * ESCALA_TRACKING)),
                interpolation=cv2.INTER_AREA
            )
        else:
            frame_track = frame

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if mode != Mode.GUN:
            contador_frames += 1
            if TRACK_CADA_N_FRAMES <= 1 or (contador_frames % TRACK_CADA_N_FRAMES) == 0 or (not ultimo_face.has_face):
                ultimo_face = face_tracker.process(frame_track)
        face = ultimo_face if mode != Mode.GUN else FaceData((0.5, 0.5), 0.0, False)

        if mode == Mode.MENU:
            menu_update(menu, face)

            if logo is not None:
                target_w = int(w * 0.42)
                target_h = int(target_w * (logo.shape[0] / logo.shape[1]))
                logo_ok = bank.logo_scaled((target_w, target_h))
                overlay_bgra(frame, logo_ok, (w // 2, int(h * 0.16)), (target_w, target_h))
            else:
                x1, y1 = w // 2 - 180, int(h * 0.06)
                x2, y2 = w // 2 + 180, int(h * 0.22)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                draw_text(frame, "LOGO EM FALTA", (w // 2 - 135, int(h * 0.16)), 0.8, 2)

            tile_w = w // 3
            tile_h = int(h * 0.35)
            y0 = int(h * 0.35)

            x1 = w // 10
            x2 = x1 + tile_w
            x3 = w - w // 10 - tile_w
            x4 = x3 + tile_w

            cv2.rectangle(frame, (x1, y0), (x2, y0 + tile_h), (40, 160, 40), -1)
            cv2.rectangle(frame, (x3, y0), (x4, y0 + tile_h), (40, 40, 160), -1)

            bank.draw_fruit(frame, "banana", (x1 + tile_w // 2, y0 + tile_h // 2 - 10), size_px=min(tile_w, tile_h) // 2)
            bank.draw_fruit(frame, "watermelon", (x3 + tile_w // 2, y0 + tile_h // 2 - 10), size_px=min(tile_w, tile_h) // 2)

            if menu.selected == 0:
                cv2.rectangle(frame, (x1 - 6, y0 - 6), (x2 + 6, y0 + tile_h + 6), (255, 255, 255), 3)
            else:
                cv2.rectangle(frame, (x3 - 6, y0 - 6), (x4 + 6, y0 + tile_h + 6), (255, 255, 255), 3)

            draw_text(frame, "1) Fruit Catcher", (x1 + 22, y0 + tile_h - 45), 0.85, 2)
            draw_text(frame, "Nariz move", (x1 + 22, y0 + tile_h - 18), 0.60, 2)
            draw_text(frame, "2) Gunslinger", (x3 + 22, y0 + tile_h - 45), 0.85, 2)
            draw_text(frame, "Indicador toca", (x3 + 22, y0 + tile_h - 18), 0.60, 2)

            if menu.is_calibrating:
                draw_text(frame, "A calibrar... olha em frente", (20, h - 55), 0.65, 2)

            if key in (ord("r"), ord("R")):
                menu_reset(menu)

            do_confirm = (not menu.is_calibrating) and (menu_confirmed(menu) or key in (13, 10))
            if menu_exit(menu):
                break

            if do_confirm:
                if menu.selected == 0:
                    mode = Mode.FRUIT
                    catcher = CatcherState(start_t=now_s())
                else:
                    mode = Mode.GUN
                    alvo, proximo, kind = gunslinger_new_target(w, h, bank)
                    guns = GunslingerState(
                        phase="WAIT",
                        score=0,
                        best_time=None,
                        signal_t=0.0,
                        next_signal_t=proximo,
                        target=alvo,
                        target_kind=kind,
                        last_shot_t=0.0,
                    )

        elif mode == Mode.FRUIT:
            assert catcher is not None
            frame, done = run_fruit_catcher(frame, face, catcher, bank)
            if key in (ord("q"), ord("Q")) or (done and key in (13, 10)):
                mode = Mode.MENU
                menu_reset(menu)
                catcher = None

        elif mode == Mode.GUN:
            assert guns is not None
            mao = hand_tracker.process(frame_track)
            frame = run_gunslinger(frame, mao, guns, bank)
            if key in (ord("q"), ord("Q")):
                mode = Mode.MENU
                menu_reset(menu)
                guns = None

        if face.has_face and mode != Mode.GUN:
            cx = int(clamp(face.nose[0], 0.0, 1.0) * w)
            cy = int(clamp(face.nose[1], 0.0, 1.0) * h)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 2, cv2.LINE_AA)

        draw_text(frame, "Fruit Arcade", (20, 40), 0.8, 2)
        draw_text(frame, f"Uptime: {now_s() - t0:0.1f}s", (20, 75), 0.7, 2)

        if mode == Mode.MENU:
            draw_text(frame, "ESC: sair", (20, h - 20), 0.65, 2)
        else:
            draw_text(frame, "ESC: sair  |  Q: menu", (20, h - 20), 0.65, 2)

        cv2.imshow(NOME_JANELA, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
