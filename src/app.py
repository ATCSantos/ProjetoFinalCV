
from __future__ import annotations

import time
import math
import random
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np
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
TRACK_FACE_CADA_N_FRAMES = 2
TRACK_HAND_MENU_CADA_N_FRAMES = 1
TRACK_HAND_GUN_CADA_N_FRAMES = 1

PASTA_ASSETS = (Path(__file__).resolve().parent.parent / "assets")

# Menu (nariz)
MENU_DX = 0.07
MENU_DY = 0.07
HOLD_CONFIRM_S = 0.22
HOLD_EXIT_S = 0.35
CALIBRATION_S = 0.9

MENU_CONFIRM_BY_MOUTH = True
MENU_MOUTH_THRESHOLD = 0.055
MENU_MOUTH_HOLD_S = 0.22

# Hard Mode (toggle no menu)
MENU_HARD_GESTO = "THUMBS"
MENU_HARD_HOLD_S = 1.00     
MENU_HARD_COOLDOWN_S = 0.70
MENU_HARD_FLASH_S = 0.85
MENU_HARD_GRACE_S = 0.22    

# Fruit Catcher
CATCHER_ROUND_S = 45.0
FRUIT_FALL_SPEED_PX = 4.2
FRUIT_FALL_SPEED_INC = 0.02
FRUIT_SPAWN_EVERY_S = (0.45, 0.90)
BASKET_W = 160
BASKET_H = 30

ENABLE_MOUTH_TO_CATCH = False
MOUTH_OPEN_THRESHOLD = 0.055

# Olhos fechados (Hard Mode)
EYES_CLOSED_RATIO = 0.55    
EYES_CLOSED_HOLD_S = 0.18

# Gunslinger
GUN_TOTAL_ROUND_S = 45.0
GUN_ROUND_TIMEOUT_S = 2.5
SIGNAL_DELAY_S = (1.0, 2.6)
TARGET_RADIUS = 60
SHOT_COOLDOWN_S = 0.70

# Gunslinger Hard Mode
GUN_ROUND_TIMEOUT_HARD_S = 2.1
TARGET_RADIUS_HARD = 48
GUN_GESTO_HARD = "POINT"

# Hard perks
HARD_DUAL_TARGET_CHANCE = 0.35
HARD_BLINK_TARGET_CHANCE = 0.25
HARD_BLINK_MOVE_INTERVAL_S = 0.55    
HARD_BLINK_BORDER_MARGIN = 55
HARD_DUAL_SHOT_COOLDOWN_S = 0.35     
HARD_SINGLE_SHOT_COOLDOWN_S = 0.55

# Anti early bird (mão visível antes do "AGORA")
ANTI_PRESHOT_RESET_S = 0.22

# UI
UI_PANEL_ALPHA = 0.48

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

def draw_multiline(img: np.ndarray, lines: List[str], x: int, y: int, scale: float, thickness: int, line_h: int) -> None:
    yy = y
    for s in lines:
        draw_text(img, s, (x, yy), scale, thickness)
        yy += line_h

def overlay_tint(frame: np.ndarray, bgr: Tuple[int, int, int], alpha: float) -> np.ndarray:
    overlay = frame.copy()
    overlay[:, :] = bgr
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)

def draw_panel(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = UI_PANEL_ALPHA, color: Tuple[int,int,int]=(0,0,0)) -> None:
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:, :] = color
    frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0)

def draw_end_screen(frame: np.ndarray, titulo: str, subtitulo: str) -> np.ndarray:
    h, w = frame.shape[:2]
    frame = overlay_tint(frame, (0, 0, 0), 0.60)
    draw_text(frame, titulo, (w // 2 - 110, h // 2 - 20), 1.10, 3)
    draw_text(frame, subtitulo, (w // 2 - 145, h // 2 + 25), 0.90, 2)
    draw_text(frame, "ENTER ou Q: voltar ao menu", (w // 2 - 195, h // 2 + 70), 0.72, 2)
    return frame

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

    alpha = (src_roi[:, :, 3:4].astype(np.float32) / 255.0)
    src_rgb = src_roi[:, :, :3].astype(np.float32)
    roi_f = roi.astype(np.float32)

    out = src_rgb * alpha + roi_f * (1.0 - alpha)
    roi[:, :] = out.astype(np.uint8)

# ----------------------------
# Trackers
# ----------------------------

EMA_ALPHA = 0.25

@dataclass
class FaceData:
    nose: Tuple[float, float]
    mouth_open: float
    eyes_open: float
    has_face: bool

class FaceTracker:
    NOSE_IDX = 1
    UPPER_LIP_IDX = 13
    LOWER_LIP_IDX = 14

    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    L_EYE_OUT = 33
    L_EYE_IN = 133
    L_EYE_UP = 159
    L_EYE_DN = 145

    R_EYE_OUT = 263
    R_EYE_IN = 362
    R_EYE_UP = 386
    R_EYE_DN = 374

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
        self._eyes_ema: Optional[float] = None

    def process(self, frame_bgr: np.ndarray) -> FaceData:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.mesh.process(rgb)
        rgb.flags.writeable = True

        if not res.multi_face_landmarks:
            self._nose_ema = None
            self._mouth_ema = None
            self._eyes_ema = None
            return FaceData((0.5, 0.5), 0.0, 1.0, False)

        lm = res.multi_face_landmarks[0].landmark
        nose = (lm[self.NOSE_IDX].x, lm[self.NOSE_IDX].y)

        eyeL = np.array([lm[self.LEFT_EYE_OUTER].x, lm[self.LEFT_EYE_OUTER].y], dtype=np.float32)
        eyeR = np.array([lm[self.RIGHT_EYE_OUTER].x, lm[self.RIGHT_EYE_OUTER].y], dtype=np.float32)
        eye_dist = float(np.linalg.norm(eyeL - eyeR)) + 1e-6

        upper = np.array([lm[self.UPPER_LIP_IDX].x, lm[self.UPPER_LIP_IDX].y], dtype=np.float32)
        lower = np.array([lm[self.LOWER_LIP_IDX].x, lm[self.LOWER_LIP_IDX].y], dtype=np.float32)
        mouth_open = float(np.linalg.norm(upper - lower)) / eye_dist

        # eye open ratio
        l_up = np.array([lm[self.L_EYE_UP].x, lm[self.L_EYE_UP].y], dtype=np.float32)
        l_dn = np.array([lm[self.L_EYE_DN].x, lm[self.L_EYE_DN].y], dtype=np.float32)
        l_out = np.array([lm[self.L_EYE_OUT].x, lm[self.L_EYE_OUT].y], dtype=np.float32)
        l_in = np.array([lm[self.L_EYE_IN].x, lm[self.L_EYE_IN].y], dtype=np.float32)
        l_h = float(np.linalg.norm(l_out - l_in)) + 1e-6
        l_v = float(np.linalg.norm(l_up - l_dn))

        r_up = np.array([lm[self.R_EYE_UP].x, lm[self.R_EYE_UP].y], dtype=np.float32)
        r_dn = np.array([lm[self.R_EYE_DN].x, lm[self.R_EYE_DN].y], dtype=np.float32)
        r_out = np.array([lm[self.R_EYE_OUT].x, lm[self.R_EYE_OUT].y], dtype=np.float32)
        r_in = np.array([lm[self.R_EYE_IN].x, lm[self.R_EYE_IN].y], dtype=np.float32)
        r_h = float(np.linalg.norm(r_out - r_in)) + 1e-6
        r_v = float(np.linalg.norm(r_up - r_dn))

        eyes_open = 0.5 * ((l_v / l_h) + (r_v / r_h))

        if self._nose_ema is None:
            self._nose_ema = nose
            self._mouth_ema = mouth_open
            self._eyes_ema = eyes_open
        else:
            self._nose_ema = (lerp(self._nose_ema[0], nose[0], EMA_ALPHA), lerp(self._nose_ema[1], nose[1], EMA_ALPHA))
            self._mouth_ema = lerp(self._mouth_ema, mouth_open, EMA_ALPHA)
            self._eyes_ema = lerp(self._eyes_ema, eyes_open, EMA_ALPHA)

        return FaceData(self._nose_ema, float(self._mouth_ema), float(self._eyes_ema), True)

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
            model_complexity=0,
            min_detection_confidence=0.50,  # mais permissivo no menu
            min_tracking_confidence=0.50,
        )

    def _finger_extended(self, lm, tip_i: int, pip_i: int) -> bool:
        return lm[tip_i].y < lm[pip_i].y

    def _thumbs_up(self, lm) -> bool:
        return lm[self.THUMB_TIP].y < (lm[self.THUMB_IP].y - 0.015)

    def _thumb_side(self, lm) -> bool:
        return abs(lm[self.THUMB_TIP].x - lm[self.THUMB_IP].x) > 0.040

    def _classify(self, lm) -> str:
        index_ = self._finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP)
        middle = self._finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring = self._finger_extended(lm, self.RING_TIP, self.RING_PIP)
        pinky = self._finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP)

        thumb_up = self._thumbs_up(lm)
        thumb_side = self._thumb_side(lm)
        thumb = thumb_up or thumb_side

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
        if thumb_up and (not index_) and (not middle) and (not ring) and (not pinky):
            return "THUMBS"
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
        gesto = self._classify(lm)
        return HandData(tip=tip, label=label, gesture=gesto)

# ----------------------------
# Sprites
# ----------------------------

class SpriteBank:
    def __init__(self) -> None:
        self.logo = load_png_rgba(str(PASTA_ASSETS / "logo.png"))
        self.fruits = {
            "orange": load_png_rgba(str(PASTA_ASSETS / "orange.png")),
            "apple": load_png_rgba(str(PASTA_ASSETS / "apple.png")),
            "watermelon": load_png_rgba(str(PASTA_ASSETS / "watermelon.png")),
            "banana": load_png_rgba(str(PASTA_ASSETS / "banana.png")),
        }
        self.fruit_keys = [k for k,v in self.fruits.items() if v is not None] or list(self.fruits.keys())
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

# ----------------------------
# Estados
# ----------------------------

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
    sample_eyes_sum: float = 0.0
    sample_count: int = 0

    baseline_eyes_open: float = 0.30

    selected: int = 0
    t_hold_confirm: Optional[float] = None
    t_hold_exit: Optional[float] = None
    t_mouth_confirm: Optional[float] = None

    hard_mode: bool = False
    t_hard_hold: Optional[float] = None
    t_hard_last_seen: float = 0.0
    t_hard_cooldown: float = 0.0
    hard_flash_until: float = 0.0
    hard_flash_txt: str = ""

def menu_reset(ms: MenuState) -> None:
    ms.baseline = None
    ms.calib_start = now_s()
    ms.is_calibrating = True
    ms.sample_sum[:] = 0.0
    ms.sample_eyes_sum = 0.0
    ms.sample_count = 0
    ms.t_hold_confirm = None
    ms.t_hold_exit = None
    ms.t_mouth_confirm = None
    ms.t_hard_hold = None
    ms.t_hard_last_seen = 0.0

def menu_update(ms: MenuState, face: FaceData, mao: Optional[HandData]) -> None:
    if not face.has_face:
        ms.t_hard_hold = None
        return

    n = np.array(face.nose, dtype=np.float64)

    if ms.is_calibrating:
        ms.sample_sum += n
        ms.sample_eyes_sum += float(face.eyes_open)
        ms.sample_count += 1
        if now_s() - ms.calib_start >= CALIBRATION_S and ms.sample_count > 0:
            ms.baseline = tuple((ms.sample_sum / ms.sample_count).tolist())
            ms.baseline_eyes_open = float(ms.sample_eyes_sum / ms.sample_count)
            ms.is_calibrating = False
        ms.t_hard_hold = None
        return

    if ms.baseline is None:
        ms.t_hard_hold = None
        return

    bx, by = ms.baseline
    dx = n[0] - bx
    dy = n[1] - by
    t = now_s()

    if dx < -MENU_DX:
        ms.selected = 0
    elif dx > MENU_DX:
        ms.selected = 1

    centro_ok = abs(dx) < (MENU_DX * 0.60)
    dy_confirm = -(MENU_DY * 1.35)
    dy_exit = (MENU_DY * 1.35)

    if centro_ok and dy < dy_confirm:
        ms.t_hold_confirm = ms.t_hold_confirm or t
    else:
        ms.t_hold_confirm = None

    if centro_ok and dy > dy_exit:
        ms.t_hold_exit = ms.t_hold_exit or t
    else:
        ms.t_hold_exit = None

    if MENU_CONFIRM_BY_MOUTH and face.mouth_open >= MENU_MOUTH_THRESHOLD:
        ms.t_mouth_confirm = ms.t_mouth_confirm or t
    else:
        ms.t_mouth_confirm = None

    # toggle hard mode (thumbs up hold)
    if mao is not None and mao.gesture == MENU_HARD_GESTO:
        ms.t_hard_last_seen = t
        if (t - ms.t_hard_cooldown) >= MENU_HARD_COOLDOWN_S:
            ms.t_hard_hold = ms.t_hard_hold or t
            if ms.t_hard_hold is not None and (t - ms.t_hard_hold) >= MENU_HARD_HOLD_S:
                ms.hard_mode = not ms.hard_mode
                ms.t_hard_cooldown = t
                ms.t_hard_hold = None
                ms.hard_flash_until = t + MENU_HARD_FLASH_S
                ms.hard_flash_txt = "HARD MODE: ON" if ms.hard_mode else "HARD MODE: OFF"
    else:
        if ms.t_hard_hold is not None and (t - ms.t_hard_last_seen) <= MENU_HARD_GRACE_S:
            pass
        else:
            ms.t_hard_hold = None

def menu_confirmed(ms: MenuState) -> bool:
    by_head = ms.t_hold_confirm is not None and (now_s() - ms.t_hold_confirm) >= HOLD_CONFIRM_S
    by_mouth = ms.t_mouth_confirm is not None and (now_s() - ms.t_mouth_confirm) >= MENU_MOUTH_HOLD_S
    return by_head or by_mouth

def menu_exit(ms: MenuState) -> bool:
    return ms.t_hold_exit is not None and (now_s() - ms.t_hold_exit) >= HOLD_EXIT_S

# ----------------------------
# Fruit Catcher
# ----------------------------

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
    t_olhos_fechados: Optional[float] = None
    ended: bool = False

def spawn_fruit(cs: CatcherState, w: int, bank: SpriteBank) -> None:
    r = random.randint(20, 32)
    x = random.randint(r + 10, w - r - 10)
    kind = random.choice(bank.fruit_keys) if bank.fruit_keys else "orange"
    cs.fruits.append(Fruit(x=float(x), y=float(-r), r=r, kind=kind, vy=cs.speed))
    cs.speed += FRUIT_FALL_SPEED_INC

def run_fruit_catcher(frame: np.ndarray, face: FaceData, cs: CatcherState, bank: SpriteBank, hard_mode: bool, eyes_base: float) -> Tuple[np.ndarray, bool]:
    h, w = frame.shape[:2]
    t = now_s()

    remaining = max(0.0, CATCHER_ROUND_S - (t - cs.start_t))
    if cs.ended or remaining <= 0.0:
        cs.ended = True
        cs.fruits.clear()
        frame = draw_end_screen(frame, "FIM!", f"Pontuacao: {cs.score}")
        return frame, True

    # basket position by nose
    bx = int(clamp(face.nose[0], 0.0, 1.0) * w) if face.has_face else w // 2
    basket_x1 = int(clamp(bx - BASKET_W // 2, 0, w - BASKET_W))
    basket_y1 = h - 70
    basket_x2 = basket_x1 + BASKET_W
    basket_y2 = basket_y1 + BASKET_H

    basket_active = True

    if cs.mouth_required:
        basket_active = (face.mouth_open >= MOUTH_OPEN_THRESHOLD)

    # eyes closed gating in hard mode
    eyes_thr = max(0.07, float(eyes_base) * EYES_CLOSED_RATIO)
    olhos_fechados_raw = face.has_face and (face.eyes_open <= eyes_thr)

    if hard_mode:
        if olhos_fechados_raw:
            cs.t_olhos_fechados = cs.t_olhos_fechados or t
        else:
            cs.t_olhos_fechados = None

        olhos_fechados_ok = (cs.t_olhos_fechados is not None) and ((t - cs.t_olhos_fechados) >= EYES_CLOSED_HOLD_S)
        basket_active = basket_active and olhos_fechados_ok
    else:
        cs.t_olhos_fechados = None

    # spawn
    if t >= cs.next_spawn_t:
        spawn_fruit(cs, w, bank)
        cs.next_spawn_t = t + random.uniform(*FRUIT_SPAWN_EVERY_S)

    # update as fruits
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

    # draw fruits
    for f in cs.fruits:
        bank.draw_fruit(frame, f.kind, (int(f.x), int(f.y)), size_px=f.r * 2)

    # basket
    basket_col = (80, 255, 80) if basket_active else (80, 80, 255)
    cv2.rectangle(frame, (basket_x1, basket_y1), (basket_x2, basket_y2), basket_col, -1)
    cv2.rectangle(frame, (basket_x1, basket_y1), (basket_x2, basket_y2), (255, 255, 255), 2)

    # HUD
    titulo = "FRUIT CATCHER" if not hard_mode else "FRUIT CATCHER: HARD MODE"
    draw_panel(frame, 14, 120, 330, 255, alpha=0.40)
    draw_text(frame, f"{titulo}", (22, 150), 0.78, 2)
    draw_text(frame, f"Score: {cs.score}", (22, 178), 0.75, 2)
    draw_text(frame, f"Tempo: {remaining:0.1f}s", (22, 206), 0.72, 2)

    if hard_mode:
        estado_olhos = "FECHADOS" if olhos_fechados_raw else "ABERTOS"
        estado_cesto = "ATIVO" if basket_active else "BLOQUEADO"
        draw_text(frame, f"Olhos: {estado_olhos}", (22, 232), 0.62, 2)
        draw_text(frame, f"Cesto: {estado_cesto}", (22, 255), 0.62, 2)

    draw_text(frame, "Q: menu | ESC: sair", (20, h - 45), 0.6, 2)
    return frame, False

# ----------------------------
# Gunslinger
# ----------------------------

@dataclass
class Target:
    x: int
    y: int
    kind: str
    hit: bool = False
    blink: bool = False
    last_move_t: float = 0.0

@dataclass
class GunslingerState:
    start_t: float
    phase: str
    score: int
    best_time: Optional[float]
    signal_t: float
    next_signal_t: float
    targets: List[Target]
    last_shot_t: float
    last_reaction: Optional[float] = None
    ended: bool = False

    bloqueado_mao: bool = False
    t_mao_sumiu: Optional[float] = None
    hard_case: str = "single"  # single | dual | blink

def _rand_kind(bank: SpriteBank) -> str:
    return random.choice(bank.fruit_keys) if bank.fruit_keys else "orange"

def _pick_center_target(w: int, h: int) -> Tuple[int,int]:
    tx = random.randint(w // 3, int(w * 0.66))
    ty = random.randint(h // 3, int(h * 0.66))
    return tx, ty

def _pick_border_target(w: int, h: int) -> Tuple[int,int]:
    m = HARD_BLINK_BORDER_MARGIN
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return random.randint(m, w - m), random.randint(m, m + 25)
    if side == "bottom":
        return random.randint(m, w - m), random.randint(h - m - 25, h - m)
    if side == "left":
        return random.randint(m, m + 25), random.randint(m, h - m)
    return random.randint(w - m - 25, w - m), random.randint(m, h - m)

def gunslinger_new_round(w: int, h: int, bank: SpriteBank, hard_mode: bool) -> Tuple[List[Target], float, str]:
    next_signal = now_s() + random.uniform(*SIGNAL_DELAY_S)
    if not hard_mode:
        tx, ty = _pick_center_target(w, h)
        return [Target(tx, ty, _rand_kind(bank))], next_signal, "single"

    r = random.random()
    if r < HARD_DUAL_TARGET_CHANCE:
        # dual targets (ensure some spacing)
        raio = TARGET_RADIUS_HARD
        for _ in range(25):
            a = _pick_center_target(w, h)
            b = _pick_center_target(w, h)
            if (a[0]-b[0])**2 + (a[1]-b[1])**2 >= (2*raio + 40)**2:
                t1 = Target(a[0], a[1], _rand_kind(bank))
                t2 = Target(b[0], b[1], _rand_kind(bank))
                return [t1, t2], next_signal, "dual"
        # fallback single
        tx, ty = _pick_center_target(w, h)
        return [Target(tx, ty, _rand_kind(bank))], next_signal, "single"

    if r < HARD_DUAL_TARGET_CHANCE + HARD_BLINK_TARGET_CHANCE:
        tx, ty = _pick_border_target(w, h)
        t1 = Target(tx, ty, _rand_kind(bank), blink=True, last_move_t=0.0)
        return [t1], next_signal, "blink"

    tx, ty = _pick_center_target(w, h)
    return [Target(tx, ty, _rand_kind(bank))], next_signal, "single"

def _shot_cooldown(hard_mode: bool, hard_case: str) -> float:
    if not hard_mode:
        return SHOT_COOLDOWN_S
    if hard_case == "dual":
        return HARD_DUAL_SHOT_COOLDOWN_S
    return HARD_SINGLE_SHOT_COOLDOWN_S

def run_gunslinger(frame: np.ndarray, mao: Optional[HandData], gs: GunslingerState, bank: SpriteBank, hard_mode: bool) -> Tuple[np.ndarray, bool]:
    h, w = frame.shape[:2]
    t = now_s()

    remaining = max(0.0, GUN_TOTAL_ROUND_S - (t - gs.start_t))
    if gs.ended or remaining <= 0.0:
        gs.ended = True
        frame = draw_end_screen(frame, "FIM!", f"Pontuacao: {gs.score}")
        return frame, True

    raio = TARGET_RADIUS_HARD if hard_mode else TARGET_RADIUS
    timeout = GUN_ROUND_TIMEOUT_HARD_S if hard_mode else GUN_ROUND_TIMEOUT_S

    if hard_mode and gs.hard_case == "blink" and gs.phase == "SIGNAL":
        if gs.targets and gs.targets[0].blink:
            tgt = gs.targets[0]
            if tgt.last_move_t == 0.0:
                tgt.last_move_t = t
            if (t - tgt.last_move_t) >= HARD_BLINK_MOVE_INTERVAL_S:
                tgt.x, tgt.y = _pick_border_target(w, h)
                tgt.last_move_t = t

    # phase transitions
    if gs.phase == "WAIT" and t >= gs.next_signal_t:
        gs.phase = "SIGNAL"
        gs.signal_t = t
        gs.last_reaction = None
        # anti pre-shot: se a mão já está visível quando aparece o "AGORA", bloqueia
        gs.bloqueado_mao = (mao is not None)
        gs.t_mao_sumiu = None

        # reset hits for safety
        for tg in gs.targets:
            tg.hit = False

    # handle signal phase
    if gs.phase == "SIGNAL":
        # desbloqueio: mão tem de desaparecer por um curto período
        if gs.bloqueado_mao:
            if mao is None:
                gs.t_mao_sumiu = gs.t_mao_sumiu or t
                if gs.t_mao_sumiu is not None and (t - gs.t_mao_sumiu) >= ANTI_PRESHOT_RESET_S:
                    gs.bloqueado_mao = False
            else:
                gs.t_mao_sumiu = None

        # timeout
        if (t - gs.signal_t) >= timeout:
            gs.phase = "RESULT"
            gs.last_shot_t = t
            gs.last_reaction = None

        # input
        cd = _shot_cooldown(hard_mode, gs.hard_case)
        if (not gs.bloqueado_mao) and mao is not None and (t - gs.last_shot_t) >= cd:
            hx = int(clamp(mao.tip[0], 0.0, 1.0) * w)
            hy = int(clamp(mao.tip[1], 0.0, 1.0) * h)

            gesto_ok = mao.gesture in ("POINT", "PEACE", "OPEN")
            if hard_mode:
                gesto_ok = (mao.gesture == GUN_GESTO_HARD)

            if gesto_ok:
                hit_any = False
                for tg in gs.targets:
                    if tg.hit:
                        continue
                    if (hx - tg.x) ** 2 + (hy - tg.y) ** 2 <= raio ** 2:
                        tg.hit = True
                        gs.last_shot_t = t
                        hit_any = True
                        break

                if hit_any and all(tg.hit for tg in gs.targets):
                    rt = t - gs.signal_t
                    gs.last_reaction = rt
                    gs.best_time = rt if gs.best_time is None else min(gs.best_time, rt)
                    gs.score += 1
                    gs.phase = "RESULT"
                    gs.last_shot_t = t

    # next round after result
    if gs.phase == "RESULT" and (t - gs.last_shot_t) >= 1.2:
        gs.targets, gs.next_signal_t, gs.hard_case = gunslinger_new_round(w, h, bank, hard_mode)
        gs.phase = "WAIT"
        gs.bloqueado_mao = False
        gs.t_mao_sumiu = None

    # --------- DRAW ---------

    titulo = "GUNSLINGER" if not hard_mode else "GUNSLINGER: HARD MODE"

    # top panel
    draw_panel(frame, 14, 120, 360, 260, alpha=0.40)
    draw_text(frame, titulo, (22, 150), 0.78, 2)
    draw_text(frame, f"Score: {gs.score}", (22, 178), 0.75, 2)
    draw_text(frame, f"Tempo: {remaining:0.1f}s", (22, 206), 0.72, 2)
    if gs.best_time is not None:
        draw_text(frame, f"Melhor: {gs.best_time:0.3f}s", (22, 234), 0.68, 2)

    # instructions
    if hard_mode:
        if gs.hard_case == "dual":
            draw_text(frame, "Hard: 2 alvos (acerta ambos)", (22, 258), 0.62, 2)
        elif gs.hard_case == "blink":
            draw_text(frame, "Hard: alvo movel na borda", (22, 258), 0.62, 2)
        else:
            draw_text(frame, "Hard: so POINT", (22, 258), 0.62, 2)

    # target visuals
    def _draw_target(tg: Target, active: bool) -> None:
        # hit targets appear dimmer
        if tg.hit:
            base = (70, 70, 70)
        elif active:
            # signal glow
            pulse = 0.55 + 0.45 * math.sin((t - gs.signal_t) * 7.0)
            base = (int(70 + 185 * pulse), int(70 + 185 * pulse), 0)
        else:
            base = (140, 140, 140)

        cv2.circle(frame, (tg.x, tg.y), raio, base, -1, cv2.LINE_AA)
        cv2.circle(frame, (tg.x, tg.y), raio, (255, 255, 255), 2, cv2.LINE_AA)
        bank.draw_fruit(frame, tg.kind, (tg.x, tg.y), size_px=raio * 2 - 6)

        if tg.hit:
            draw_text(frame, "OK", (tg.x - 16, tg.y + 6), 0.65, 2)

    if gs.phase == "WAIT":
        for tg in gs.targets:
            _draw_target(tg, active=False)
        draw_text(frame, "Espera...", (22, 290), 0.70, 2)

    elif gs.phase == "SIGNAL":
        for tg in gs.targets:
            _draw_target(tg, active=True)

        if gs.bloqueado_mao:
            frame[:] = overlay_tint(frame, (0, 0, 255), 0.22)
            draw_text(frame, "RETIRA A MAO DO ECRA!", (w // 2 - 230, h // 2 - 10), 1.0, 3)
            draw_text(frame, "So depois podes disparar", (w // 2 - 215, h // 2 + 28), 0.8, 2)
        else:
            if hard_mode:
                if gs.hard_case == "dual":
                    msg = "AGORA! (POINT) acerta os 2 alvos"
                elif gs.hard_case == "blink":
                    msg = "AGORA! (POINT) apanha o alvo movel"
                else:
                    msg = "AGORA! (POINT) toca na fruta"
            else:
                msg = "AGORA! toca na fruta"
            draw_text(frame, msg, (22, 290), 0.70, 2)

    else:
        # RESULT
        for tg in gs.targets:
            _draw_target(tg, active=False)
        if gs.last_reaction is not None:
            draw_text(frame, f"OK! {gs.last_reaction:0.3f}s", (22, 290), 0.80, 2)
        else:
            draw_text(frame, "Falhou (timeout)", (22, 290), 0.80, 2)

    # finger debug
    if mao is not None:
        hx = int(clamp(mao.tip[0], 0.0, 1.0) * w)
        hy = int(clamp(mao.tip[1], 0.0, 1.0) * h)
        cv2.circle(frame, (hx, hy), 10, (255, 255, 255), -1)
        cv2.circle(frame, (hx, hy), 10, (0, 0, 0), 2)
        draw_text(frame, f"gesto: {mao.gesture}", (22, 320), 0.65, 2)

    draw_text(frame, "Q: menu | ESC: sair", (20, h - 45), 0.6, 2)
    return frame, False

# ----------------------------
# Menu UI
# ----------------------------

def draw_menu(frame: np.ndarray, w: int, h: int, bank: SpriteBank, menu: MenuState, mao_menu: Optional[HandData]) -> None:
    # Layout constants
    pad = 18
    top_h = int(h * 0.22)
    cards_y = int(h * 0.28)
    cards_h = int(h * 0.42)
    bottom_h = int(h * 0.20)

    # Logo
    if bank.logo is not None:
        target_w = int(w * 0.34)
        target_h = int(target_w * (bank.logo.shape[0] / bank.logo.shape[1]))
        logo_ok = bank.logo_scaled((target_w, target_h))
        overlay_bgra(frame, logo_ok, (w // 2, int(top_h * 0.55)), (target_w, target_h))

    # Left info panel (Hard Mode + perks)
    panel_w = int(w * 0.46)
    draw_panel(frame, pad, pad, pad + panel_w, cards_y - 10, alpha=0.42)

    hard_txt = "ON" if menu.hard_mode else "OFF"
    draw_text(frame, f"HARD MODE: {hard_txt}", (pad + 10, pad + 30), 0.95, 3)
    draw_text(frame, f"Toggle: thumbs up (segura {MENU_HARD_HOLD_S:0.1f}s)", (pad + 10, pad + 62), 0.60, 2)

    # Progress bar if holding
    if menu.t_hard_hold is not None:
        prog = clamp((now_s() - menu.t_hard_hold) / MENU_HARD_HOLD_S, 0.0, 1.0)
        bar_x1 = pad + 10
        bar_y1 = pad + 78
        bar_w = panel_w - 20
        bar_h = 10
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (255, 255, 255), 1)
        cv2.rectangle(frame, (bar_x1 + 1, bar_y1 + 1), (bar_x1 + 1 + int((bar_w - 2) * prog), bar_y1 + bar_h - 1), (255, 255, 255), -1)

    lines = [
        "Hard perks:",
        "- Catcher: so conta com olhos fechados",
        "- Gunslinger: so POINT + 2 alvos OU alvo movel",
    ]
    draw_multiline(frame, lines, pad + 10, pad + 110, 0.56, 2, 22)

    # Cards
    gap = 20
    card_w = int((w - pad * 2 - gap) / 2)
    x1 = pad
    x2 = pad + card_w
    x3 = pad + card_w + gap
    x4 = x3 + card_w

    # Card 1 (Catcher)
    cv2.rectangle(frame, (x1, cards_y), (x2, cards_y + cards_h), (40, 160, 40), -1)
    cv2.rectangle(frame, (x1, cards_y), (x2, cards_y + cards_h), (255, 255, 255), 2 if menu.selected == 0 else 1)

    # Card 2 (Gunslinger)
    cv2.rectangle(frame, (x3, cards_y), (x4, cards_y + cards_h), (40, 40, 160), -1)
    cv2.rectangle(frame, (x3, cards_y), (x4, cards_y + cards_h), (255, 255, 255), 2 if menu.selected == 1 else 1)

    # Fruit icons
    size_px = int(min(card_w, cards_h) * 0.52)
    bank.draw_fruit(frame, "banana", (x1 + card_w // 2, cards_y + int(cards_h * 0.45)), size_px=size_px)
    bank.draw_fruit(frame, "watermelon", (x3 + card_w // 2, cards_y + int(cards_h * 0.45)), size_px=size_px)

    # Titles
    t1 = "Fruit Catcher" + (" (Hard)" if menu.hard_mode else "")
    t2 = "Gunslinger" + (" (Hard)" if menu.hard_mode else "")
    draw_text(frame, t1, (x1 + 18, cards_y + cards_h - 38), 0.80, 2)
    draw_text(frame, "Nariz move o cesto", (x1 + 18, cards_y + cards_h - 14), 0.56, 2)
    draw_text(frame, t2, (x3 + 18, cards_y + cards_h - 38), 0.80, 2)
    draw_text(frame, "Indicador toca o alvo", (x3 + 18, cards_y + cards_h - 14), 0.56, 2)

    # Controls panel bottom
    yb1 = h - bottom_h + 10
    draw_panel(frame, pad, h - bottom_h, w - pad, h - 10, alpha=0.40)
    ctrl_lines = [
        "Controlos menu:",
        "Nariz Esq/Dir: escolher  |  Nariz Cima: confirmar  |  Nariz Baixo: sair",
        "Boca aberta: confirmar  |  R: recalibrar  |  ESC: sair",
    ]
    draw_multiline(frame, ctrl_lines, pad + 10, yb1 + 10, 0.58, 2, 22)

    if menu.is_calibrating:
        draw_text(frame, "A calibrar... olha em frente", (pad + 10, h - 22), 0.56, 2)

    if mao_menu is not None:
        draw_text(frame, f"Gesto: {mao_menu.gesture}", (w - 210, pad + 32), 0.58, 2)

    # Flash feedback
    if now_s() < menu.hard_flash_until:
        frame[:] = overlay_tint(frame, (0, 0, 0), 0.18)
        draw_text(frame, menu.hard_flash_txt, (w // 2 - 180, 95), 1.0, 3)

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
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()

    mode = Mode.MENU
    menu = MenuState()
    menu_reset(menu)

    catcher: Optional[CatcherState] = None
    guns: Optional[GunslingerState] = None

    frame_count_face = 0
    frame_count_hand_menu = 0
    frame_count_hand_gun = 0

    ultimo_face = FaceData((0.5, 0.5), 0.0, 1.0, False)
    mao_menu: Optional[HandData] = None
    mao_gun: Optional[HandData] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if ESPELHAR_CAMARA:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        if ESCALA_TRACKING != 1.0:
            frame_track = cv2.resize(frame, (int(w * ESCALA_TRACKING), int(h * ESCALA_TRACKING)), interpolation=cv2.INTER_AREA)
        else:
            frame_track = frame

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        # --- FACE ---
        if mode != Mode.GUN:
            frame_count_face += 1
            if TRACK_FACE_CADA_N_FRAMES <= 1 or (frame_count_face % TRACK_FACE_CADA_N_FRAMES) == 0 or (not ultimo_face.has_face):
                ultimo_face = face_tracker.process(frame_track)
        face = ultimo_face if mode != Mode.GUN else FaceData((0.5, 0.5), 0.0, 1.0, False)

        # --- HAND (menu) ---
        if mode == Mode.MENU:
            frame_count_hand_menu += 1
            if TRACK_HAND_MENU_CADA_N_FRAMES <= 1 or (frame_count_hand_menu % TRACK_HAND_MENU_CADA_N_FRAMES) == 0:
                mao_menu = hand_tracker.process(frame_track)
        else:
            mao_menu = None

        # --- HAND (gun) ---
        if mode == Mode.GUN:
            frame_count_hand_gun += 1
            if TRACK_HAND_GUN_CADA_N_FRAMES <= 1 or (frame_count_hand_gun % TRACK_HAND_GUN_CADA_N_FRAMES) == 0 or (mao_gun is None):
                mao_gun = hand_tracker.process(frame_track)
        else:
            mao_gun = None

        # ---------------- MENU ----------------
        if mode == Mode.MENU:
            menu_update(menu, face, mao_menu)
            draw_menu(frame, w, h, bank, menu, mao_menu)

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
                    targets, next_sig, hard_case = gunslinger_new_round(w, h, bank, hard_mode=menu.hard_mode)
                    guns = GunslingerState(
                        start_t=now_s(),
                        phase="WAIT",
                        score=0,
                        best_time=None,
                        signal_t=0.0,
                        next_signal_t=next_sig,
                        targets=targets,
                        last_shot_t=0.0,
                        hard_case=hard_case,
                    )
                    mao_gun = None
                    frame_count_hand_gun = 0

        # ---------------- FRUIT ----------------
        elif mode == Mode.FRUIT:
            assert catcher is not None
            frame, done = run_fruit_catcher(frame, face, catcher, bank, hard_mode=menu.hard_mode, eyes_base=menu.baseline_eyes_open)

            if done:
                if key in (13, 10, ord("q"), ord("Q")):
                    mode = Mode.MENU
                    menu_reset(menu)
                    catcher = None
            else:
                if key in (ord("q"), ord("Q")):
                    mode = Mode.MENU
                    menu_reset(menu)
                    catcher = None

        # ---------------- GUN ----------------
        elif mode == Mode.GUN:
            assert guns is not None
            frame, done = run_gunslinger(frame, mao_gun, guns, bank, hard_mode=menu.hard_mode)

            if done:
                if key in (13, 10, ord("q"), ord("Q")):
                    mode = Mode.MENU
                    menu_reset(menu)
                    guns = None
                    mao_gun = None
            else:
                if key in (ord("q"), ord("Q")):
                    mode = Mode.MENU
                    menu_reset(menu)
                    guns = None
                    mao_gun = None

        # debug nose dot (only outside gun)
        if face.has_face and mode != Mode.GUN:
            cx = int(clamp(face.nose[0], 0.0, 1.0) * w)
            cy = int(clamp(face.nose[1], 0.0, 1.0) * h)
            cv2.circle(frame, (cx, cy), 9, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 9, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(NOME_JANELA, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
