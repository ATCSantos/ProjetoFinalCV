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
# Configuraçãos
# ----------------------------

CAMERA_INDEX = 0
NOME_JANELA = "Fruit Arcade"
ESPELHAR_CAMARA = True

LARGURA_CAMARA = 640
ALTURA_CAMARA = 480
ESCALA_TRACKING = 0.5
TRACK_CADA_N_FRAMES = 2

PASTA_ASSETS = (Path(__file__).resolve().parent.parent / "assets")

# Menu controls
MENU_DX = 0.07
MENU_DY = 0.07
HOLD_CONFIRM_S = 0.22
HOLD_EXIT_S = 0.35
CALIBRATION_S = 0.9

MENU_CONFIRM_BY_MOUTH = True
MENU_MOUTH_THRESHOLD = 0.055
MENU_MOUTH_HOLD_S = 0.22

# Menu UI layout
MENU_MARGIN_PX = 18
MENU_TILE_GAP_PX = 20
MENU_HEADER_H_RATIO = 0.20
MENU_FOOTER_H_RATIO = 0.16

# Hard Mode
MENU_HARD_GESTO = "THUMBS"
MENU_HARD_HOLD_S = 1.00
MENU_HARD_COOLDOWN_S = 0.70
MENU_HARD_FLASH_S = 0.75
MENU_HARD_GRACE_S = 0.16

# Fruit Catcher
CATCHER_ROUND_S = 45
FRUIT_FALL_SPEED_PX = 4.2
FRUIT_FALL_SPEED_INC = 0.02
FRUIT_SPAWN_EVERY_S = (0.45, 0.90)
BASKET_W = 160
BASKET_H = 30

ENABLE_MOUTH_TO_CATCH = False
MOUTH_OPEN_THRESHOLD = 0.055

# Olhos fechados
EYES_CLOSED_RATIO = 0.75
EYES_CLOSED_HOLD_S = 0.15

# Gunslinger
GUN_TOTAL_ROUND_S = 45.0
GUN_ROUND_TIMEOUT_S = 2.5
SIGNAL_DELAY_S = (1.0, 2.6)
TARGET_RADIUS = 60
SHOT_COOLDOWN_S = 0.7

# Gunslinger Hard Mode
GUN_ROUND_TIMEOUT_HARD_S = 1.9
TARGET_RADIUS_HARD = 45
GUN_GESTO_HARD = "POINT"

# Hard mode patterns
HARD_CHANCE_DOUBLE = 0.35          # 2 alvos ao mesmo tempo
HARD_CHANCE_TELEPORT = 0.25        # 1 alvo que teleporta na borda
HARD_TELEPORT_STEP_S = 0.30        # teleporte
HARD_BORDER_MARGIN_PX = 55         # distância mínima da borda

# Score rules
HARD_DOUBLE_PARTIAL_SCORE = 0.5    # apanha 1/2 => meio ponto; 2/2 => 1 ponto

# Anti early bird
ANTI_PRESHOT_RESET_S = 0.20

# Debug visuals
SHOW_NOSE_CURSOR = False

# ----------------------------
# Power-ups Fruit Catcher
# ----------------------------

# Tipos base usados como fruta "normal" no Fruit Catcher
NORMAL_FRUITS = ["orange", "apple", "watermelon", "banana"]

# Power-ups do Fruit Catcher
STRAWBERRY_BASKET_DURATION_S = 6.0      # morango -> cesto gigante durante N segundos
CHERRY_DOUBLE_DURATION_S = 5.0          # 2 cerejas -> pontos a dobrar durante N segundos
CHERRY_PAIR_WINDOW_S = 8.0              # tempo máximo entre a 1.ª e a 2.ª cereja

STRAWBERRY_SPEED_MULT = 1.6             # morango cai mais rápido
BOMB_SPEED_MULT = 1.2                   # bomba cai um pouco mais rápido

# Probabilidades aproximadas de spawn por fruta especial (por spawn de fruta)
# Tornadas ~2x mais raras do que antes
STRAWBERRY_CHANCE = 0.09
BOMB_CHANCE = 0.06
CHERRY_PAIR_CHANCE = 0.07               # só usada se não houver cerejas no ecrã

# Targets especiais para Gunslinger
GUN_BAD_TARGETS = ["broccoli", "squash"]           # legumes que tiram pontos
GUN_NORMAL_TARGETS = NORMAL_FRUITS                # frutas normais boas
ALL_GUN_TARGETS = GUN_NORMAL_TARGETS + GUN_BAD_TARGETS
GUN_VEG_PENALTY = 1.0                             # penalização por legume acertado

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

def overlay_tint(frame: np.ndarray, bgr: Tuple[int, int, int], alpha: float) -> np.ndarray:
    overlay = frame.copy()
    overlay[:, :] = bgr
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)

def blend_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.45, bgr: Tuple[int,int,int] = (0,0,0)) -> None:
    h, w = frame.shape[:2]
    x1 = int(clamp(x1, 0, w))
    x2 = int(clamp(x2, 0, w))
    y1 = int(clamp(y1, 0, h))
    y2 = int(clamp(y2, 0, h))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:, :] = bgr
    frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0)

def draw_end_screen(frame: np.ndarray, titulo: str, subtitulo: str) -> np.ndarray:
    h, w = frame.shape[:2]
    frame = overlay_tint(frame, (0, 0, 0), 0.58)
    draw_text(frame, titulo, (w // 2 - 160, h // 2 - 10), 1.05, 3)
    draw_text(frame, subtitulo, (w // 2 - 200, h // 2 + 35), 0.85, 2)
    draw_text(frame, "ENTER ou Q: voltar ao menu", (w // 2 - 215, h // 2 + 80), 0.75, 2)
    return frame

def fmt_score(v: float) -> str:
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.1f}".rstrip("0").rstrip(".")

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

EMA_ALPHA = 0.25

# ----------------------------
# Face tracking
# ----------------------------

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
            self._nose_ema = (
                lerp(self._nose_ema[0], nose[0], EMA_ALPHA),
                lerp(self._nose_ema[1], nose[1], EMA_ALPHA),
            )
            self._mouth_ema = lerp(self._mouth_ema, mouth_open, EMA_ALPHA)
            self._eyes_ema = lerp(self._eyes_ema, eyes_open, EMA_ALPHA)

        return FaceData(self._nose_ema, float(self._mouth_ema), float(self._eyes_ema), True)

# ----------------------------
# Hand tracking
# ----------------------------

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
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
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
        self.menu_catcher = load_png_rgba(str(PASTA_ASSETS / "menu_catcher.png"))
        self.menu_gunslinger = load_png_rgba(str(PASTA_ASSETS / "menu_gunslinger.png"))

        # Todos os objetos gráficos que o jogo reconhece
        all_names = [
            "orange", "apple", "watermelon", "banana",   # frutas base
            "strawberry", "bomb", "cherry",              # power-ups Fruit Catcher
            "broccoli", "squash",                        # "legumes" extra para o Gunslinger
        ]

        self.fruits: dict[str, Optional[np.ndarray]] = {}
        for name in all_names:
            self.fruits[name] = load_png_rgba(str(PASTA_ASSETS / f"{name}.png"))

        # Lista geral
        self.fruit_keys = [k for k, v in self.fruits.items() if v is not None]

        self._cache_fruit: dict[Tuple[str, int], np.ndarray] = {}
        self._cache_logo: dict[Tuple[int, int], np.ndarray] = {}
        self._cache_menu_icon: dict[Tuple[str, int, int], np.ndarray] = {}

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

    def _menu_icon_scaled(self, which: str, max_w: int, max_h: int) -> Optional[np.ndarray]:
        src = self.menu_catcher if which == "catcher" else self.menu_gunslinger
        if src is None:
            return None
        max_w = max(2, int(max_w))
        max_h = max(2, int(max_h))
        key = (which, max_w, max_h)
        if key in self._cache_menu_icon:
            return self._cache_menu_icon[key]

        ih, iw = src.shape[:2]
        if iw <= 0 or ih <= 0:
            return None

        scale = min(max_w / float(iw), max_h / float(ih))
        nw = max(2, int(iw * scale))
        nh = max(2, int(ih * scale))
        self._cache_menu_icon[key] = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        return self._cache_menu_icon[key]

    def draw_menu_icon(self, frame: np.ndarray, which: str, center: Tuple[int, int], box: Tuple[int, int], fallback_fruit: str) -> None:
        icon = self._menu_icon_scaled(which, box[0], box[1])
        if icon is None:
            self.draw_fruit(frame, fallback_fruit, center, size_px=min(box[0], box[1]))
            return
        overlay_bgra(frame, icon, center, (icon.shape[1], icon.shape[0]))

# ----------------------------
# Estados & Menu
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

    # toggle hard mode
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

    # --- estado dos power-ups ---
    basket_boost_until: float = 0.0      # morango: cesto maior até este instante
    double_score_until: float = 0.0      # cerejas: pontos a dobrar até este instante
    cherry_pending: bool = False         # já apanhou a 1.ª cereja, falta a 2.ª
    cherry_deadline: float = 0.0         # limite temporal para completar o par

    # Mensagem rápida (desativada agora, mas mantida caso queiras voltar)
    ui_message: str = ""
    ui_message_until: float = 0.0

    # Ecrã de ajuda inicial
    showing_help: bool = True

def spawn_fruit(cs: CatcherState, w: int, bank: SpriteBank) -> None:
    """Spawna fruta normal ou um power-up no Fruit Catcher."""
    r = random.randint(20, 32)
    x_min = r + 10
    x_max = w - r - 10
    if x_max <= x_min:
        x_min, x_max = 10, max(20, w - 10)

    # Não spawnar novo power-up se já houver algum power-up em queda
    upgrade_on_screen = any(f.kind in ("strawberry", "bomb", "cherry") for f in cs.fruits)

    # 1) Tentar spawnar um PAR de cerejas (no máximo 1 par ativo, e sem outros power-ups)
    tem_cerejas_no_ecra = any(f.kind == "cherry" for f in cs.fruits)
    pode_spawnar_cherry = (not tem_cerejas_no_ecra) and (not cs.cherry_pending) and (not upgrade_on_screen)

    if pode_spawnar_cherry and random.random() < CHERRY_PAIR_CHANCE:
        # escolhe um centro e coloca duas cerejas bem afastadas mas não coladas à borda
        largura_util = x_max - x_min
        if largura_util < 100:
            mid = (x_min + x_max) // 2
            offset = 50
        else:
            mid = random.randint(x_min + 50, x_max - 50)
            offset = random.randint(45, 95)

        x1 = int(clamp(mid - offset, x_min, x_max))
        x2 = int(clamp(mid + offset, x_min, x_max))

        cs.fruits.append(Fruit(x=float(x1), y=float(-r), r=r, kind="cherry", vy=cs.speed))
        cs.fruits.append(Fruit(x=float(x2), y=float(-r), r=r, kind="cherry", vy=cs.speed))
        cs.speed += 2 * FRUIT_FALL_SPEED_INC
        return

    # 2) Decidir entre morango / bomba / fruta normal
    if not upgrade_on_screen:
        roll = random.random()
        if roll < STRAWBERRY_CHANCE:
            kind = "strawberry"
            vy = cs.speed * STRAWBERRY_SPEED_MULT
            x = random.randint(x_min, x_max)
        elif roll < STRAWBERRY_CHANCE + BOMB_CHANCE:
            kind = "bomb"
            vy = cs.speed * BOMB_SPEED_MULT
            # bomba cai sempre perto de uma das margens
            margem = min(80, max(40, (x_max - x_min) // 4))
            if random.random() < 0.5:
                x = random.randint(x_min, x_min + margem)
            else:
                x = random.randint(x_max - margem, x_max)
        else:
            kind = random.choice(NORMAL_FRUITS)
            vy = cs.speed
            x = random.randint(x_min, x_max)
    else:
        # já há um power-up em queda -> só fruta normal
        kind = random.choice(NORMAL_FRUITS)
        vy = cs.speed
        x = random.randint(x_min, x_max)

    cs.fruits.append(Fruit(x=float(x), y=float(-r), r=r, kind=kind, vy=vy))
    cs.speed += FRUIT_FALL_SPEED_INC

def handle_catcher_hit(cs: CatcherState, f: Fruit, t: float) -> Tuple[int, bool]:
    # multiplicador global de pontos (ativado pelas 2 cerejas)
    mult = 2 if t < cs.double_score_until else 1

    # --- MORANGO: cesto gigante por alguns segundos ---
    if f.kind == "strawberry":
        cs.basket_boost_until = max(cs.basket_boost_until, t + STRAWBERRY_BASKET_DURATION_S)
        return 3 * mult, False  # morango dá mais pontos

    # --- BOMBA: apanha tudo o que está no ecrã ---
    if f.kind == "bomb":
        total_outros = max(0, len(cs.fruits) - 1)
        ganhos = (1 + total_outros) * mult
        return ganhos, True

    # --- CEREJAS: 2 cerejas => pontos a dobrar ---
    if f.kind == "cherry":
        if not cs.cherry_pending:
            # primeira cereja do par
            cs.cherry_pending = True
            cs.cherry_deadline = t + CHERRY_PAIR_WINDOW_S
        else:
            # segunda cereja
            if t <= cs.cherry_deadline:
                cs.double_score_until = t + CHERRY_DOUBLE_DURATION_S
            cs.cherry_pending = False
            cs.cherry_deadline = 0.0
        return 1 * mult, False  # cada cereja também vale 1 ponto

    # fruta normal
    return 1 * mult, False

def run_fruit_catcher(
    frame: np.ndarray,
    face: FaceData,
    cs: CatcherState,
    bank: SpriteBank,
    hard_mode: bool,
    eyes_base: float
) -> Tuple[np.ndarray, bool]:
    h, w = frame.shape[:2]
    t = now_s()
    remaining = max(0.0, CATCHER_ROUND_S - (t - cs.start_t))
    done = remaining <= 0.0

    if done:
        frame = draw_end_screen(frame, "FIM!", f"Pontuacao: {cs.score}")
        return frame, True

    # largura efetiva do cesto (normal ou "gigante" por efeito do morango)
    if t < cs.basket_boost_until:
        basket_w = int(BASKET_W * 1.6)
    else:
        basket_w = BASKET_W

    bx = int(clamp(face.nose[0], 0.0, 1.0) * w) if face.has_face else w // 2
    basket_x1 = int(clamp(bx - basket_w // 2, 0, w - basket_w))
    basket_y1 = h - 70
    basket_x2 = basket_x1 + basket_w
    basket_y2 = basket_y1 + BASKET_H

    basket_active = (not cs.mouth_required) or (face.mouth_open >= MOUTH_OPEN_THRESHOLD)

    eyes_thr = max(0.08, float(eyes_base) * EYES_CLOSED_RATIO)
    olhos_fechados_raw = face.has_face and (face.eyes_open <= eyes_thr)

    if hard_mode:
        if olhos_fechados_raw:
            cs.t_olhos_fechados = cs.t_olhos_fechados or t
        else:
            cs.t_olhos_fechados = None

        olhos_fechados_ok = cs.t_olhos_fechados is not None and (t - cs.t_olhos_fechados) >= EYES_CLOSED_HOLD_S
        basket_active = basket_active and olhos_fechados_ok

    # spawn normal/power-up
    if t >= cs.next_spawn_t:
        spawn_fruit(cs, w, bank)
        cs.next_spawn_t = t + random.uniform(*FRUIT_SPAWN_EVERY_S)

    new_fruits: List[Fruit] = []
    clear_all = False

    for f in cs.fruits:
        f.y += f.vy

        if basket_active:
            cx = clamp(f.x, basket_x1, basket_x2)
            cy = clamp(f.y, basket_y1, basket_y2)
            if (f.x - cx) ** 2 + (f.y - cy) ** 2 <= f.r ** 2:
                delta, do_clear = handle_catcher_hit(cs, f, t)
                cs.score += delta
                if do_clear:
                    clear_all = True
                continue  # este fruto foi apanhado

        if clear_all:
            # uma bomba já limpou o ecrã neste frame
            continue

        if f.y - f.r > h + 10:
            continue

        new_fruits.append(f)

    cs.fruits = [] if clear_all else new_fruits

    # desenhar frutas
    for f in cs.fruits:
        bank.draw_fruit(frame, f.kind, (int(f.x), int(f.y)), size_px=f.r * 2)

    # desenhar cesto
    basket_col = (80, 255, 80) if basket_active else (80, 80, 255)
    cv2.rectangle(frame, (basket_x1, basket_y1), (basket_x2, basket_y2), basket_col, -1)
    cv2.rectangle(frame, (basket_x1, basket_y1), (basket_x2, basket_y2), (255, 255, 255), 2)

    titulo = "FRUIT CATCHER" if not hard_mode else "FRUIT CATCHER (HARD MODE)"
    draw_text(frame, f"{titulo}  |  Score: {cs.score}", (20, 140), 0.9, 2)
    draw_text(frame, f"Tempo: {remaining:0.1f}s", (20, 175), 0.8, 2)

    # HUD dos power-ups (estado, não explicação chata)
    hud_y = 210
    if t < cs.basket_boost_until:
        draw_text(frame, "Morango ativo: cesto gigante", (20, hud_y), 0.70, 2)
        hud_y += 28
    if t < cs.double_score_until:
        draw_text(frame, "Cerejas ativas: pontos x2", (20, hud_y), 0.70, 2)
        hud_y += 28

    if hard_mode:
        estado_olhos = "FECHADOS" if olhos_fechados_raw else "ABERTOS"
        estado_cesto = "ATIVO" if basket_active else "BLOQUEADO"
        draw_text(frame, f"Olhos: {estado_olhos}", (20, hud_y), 0.70, 2)
        hud_y += 28
        draw_text(frame, f"Cesto: {estado_cesto} (fecha {EYES_CLOSED_HOLD_S:0.2f}s)", (20, hud_y), 0.70, 2)
        hud_y += 28

    draw_text(frame, "Q: menu | ESC: sair", (20, h - 45), 0.6, 2)
    return frame, False

def draw_catcher_help_screen(frame: np.ndarray, bank: SpriteBank, hard_mode: bool) -> None:
    """Ecrã inicial de ajuda para o Fruit Catcher (power-ups)."""
    h, w = frame.shape[:2]
    tinted = overlay_tint(frame, (0, 0, 0), 0.65)
    frame[:] = tinted

    cx = w // 2
    draw_text(frame, "FRUIT CATCHER", (cx - 160, 70), 1.1, 3)
    draw_text(frame, "Power-ups e regras básicas", (cx - 200, 110), 0.8, 2)

    base_y = 160
    linha_gap = 90
    icon_size = 60
    icon_cx = int(w * 0.14)
    text_x = icon_cx + icon_size

    # Morango
    y = base_y
    bank.draw_fruit(frame, "strawberry", (icon_cx, y), icon_size)
    draw_text(frame, "Morango: cesto gigante", (text_x, y - 10), 0.7, 2)
    draw_text(frame, f"Durante ~{int(STRAWBERRY_BASKET_DURATION_S)}s o cesto fica maior.", (text_x, y + 18), 0.6, 2)

    # Cerejas
    y += linha_gap
    # desenhar 2 cerejas lado a lado
    bank.draw_fruit(frame, "cherry", (icon_cx - 18, y), icon_size)
    bank.draw_fruit(frame, "cherry", (icon_cx + 18, y), icon_size)
    draw_text(frame, "Cerejas: 2 seguidas ativam pontos x2", (text_x, y - 10), 0.7, 2)
    draw_text(frame, "Apanha 2 cerejas em pouco tempo", (text_x, y + 18), 0.6, 2)
    draw_text(frame, f"para ativar pontos a dobrar ~{int(CHERRY_DOUBLE_DURATION_S)}s.", (text_x, y + 38), 0.6, 2)

    # Bomba
    y += linha_gap
    bank.draw_fruit(frame, "bomb", (icon_cx, y), icon_size)
    draw_text(frame, "Bomba: limpa o ecrã", (text_x, y - 10), 0.7, 2)
    draw_text(frame, "Se apanhares a bomba, todas as frutas", (text_x, y + 18), 0.6, 2)
    draw_text(frame, "no ecrã contam como apanhadas.", (text_x, y + 38), 0.6, 2)

    footer_y = h - 60
    draw_text(frame, "Espaço: começar  |  Q: voltar ao menu", (cx - 230, footer_y), 0.75, 2)

# ----------------------------
# Gunslinger
# ----------------------------

@dataclass
class Target:
    x: int
    y: int
    kind: str
    hit: bool = False
    is_teleport: bool = False
    next_move_t: float = 0.0

@dataclass
class GunslingerState:
    start_t: float
    phase: str
    score: float
    best_time: Optional[float]
    signal_t: float
    next_signal_t: float
    targets: List[Target]
    last_shot_t: float
    last_reaction: Optional[float] = None
    last_hits: int = 0
    last_total: int = 0
    last_award: float = 0.0

    bloqueado_mao: bool = False
    t_mao_sumiu: Optional[float] = None

    # info extra para legumes + ajuda
    last_bad_hits: int = 0
    showing_help: bool = True

def border_pos(w: int, h: int) -> Tuple[int, int]:
    m = HARD_BORDER_MARGIN_PX
    side = random.randint(0, 3)
    if side == 0:  # top
        return (random.randint(m, w - m), m)
    if side == 1:  # bottom
        return (random.randint(m, w - m), h - m)
    if side == 2:  # left
        return (m, random.randint(m, h - m))
    return (w - m, random.randint(m, h - m))

def gunslinger_new_targets(w: int, h: int, bank: SpriteBank, hard_mode: bool) -> Tuple[List[Target], float]:
    # Pool apenas de frutas normais + legumes (sem power-ups visuais)
    gun_pool = [k for k in bank.fruit_keys if k in ALL_GUN_TARGETS]
    if not gun_pool:
        gun_pool = ["orange"]

    kind1 = random.choice(gun_pool)
    kind2 = random.choice(gun_pool)

    def center_pos() -> Tuple[int, int]:
        tx = random.randint(w // 3, int(w * 0.66))
        ty = random.randint(h // 3, int(h * 0.66))
        return tx, ty

    targets: List[Target] = []
    if hard_mode:
        r = random.random()
        if r < HARD_CHANCE_TELEPORT:
            x, y = border_pos(w, h)
            targets = [Target(x=x, y=y, kind=kind1, hit=False, is_teleport=True, next_move_t=now_s() + HARD_TELEPORT_STEP_S)]
        elif r < HARD_CHANCE_TELEPORT + HARD_CHANCE_DOUBLE:
            x1, y1 = center_pos()
            x2, y2 = center_pos()
            for _ in range(10):
                if (x1 - x2) ** 2 + (y1 - y2) ** 2 >= (TARGET_RADIUS_HARD * 2.2) ** 2:
                    break
                x2, y2 = center_pos()
            targets = [Target(x=x1, y=y1, kind=kind1), Target(x=x2, y=y2, kind=kind2)]
        else:
            x, y = center_pos()
            targets = [Target(x=x, y=y, kind=kind1)]
    else:
        x, y = center_pos()
        # Em modo normal, privilegiar frutas boas
        good_pool = [k for k in gun_pool if k in GUN_NORMAL_TARGETS] or gun_pool
        kind1 = random.choice(good_pool)
        targets = [Target(x=x, y=y, kind=kind1)]

    next_signal = now_s() + random.uniform(*SIGNAL_DELAY_S)
    return targets, next_signal

def run_gunslinger(
    frame: np.ndarray,
    mao: Optional[HandData],
    gs: GunslingerState,
    bank: SpriteBank,
    hard_mode: bool
) -> Tuple[np.ndarray, bool]:
    h, w = frame.shape[:2]
    t = now_s()

    remaining = max(0.0, GUN_TOTAL_ROUND_S - (t - gs.start_t))
    if remaining <= 0.0:
        frame = draw_end_screen(frame, "FIM!", f"Pontuaçao: {fmt_score(gs.score)}")
        return frame, True

    raio = TARGET_RADIUS_HARD if hard_mode else TARGET_RADIUS
    timeout = GUN_ROUND_TIMEOUT_HARD_S if hard_mode else GUN_ROUND_TIMEOUT_S

    # teleporte
    if hard_mode:
        for tgt in gs.targets:
            if tgt.is_teleport and (not tgt.hit) and t >= tgt.next_move_t:
                tgt.x, tgt.y = border_pos(w, h)
                tgt.next_move_t = t + HARD_TELEPORT_STEP_S

    if gs.phase == "WAIT" and t >= gs.next_signal_t:
        gs.phase = "SIGNAL"
        gs.signal_t = t
        gs.last_reaction = None
        gs.last_hits = 0
        gs.last_total = 0
        gs.last_bad_hits = 0
        gs.last_award = 0.0
        for tgt in gs.targets:
            tgt.hit = False
        gs.bloqueado_mao = (mao is not None)
        gs.t_mao_sumiu = None

    def gesture_ok_for_shot() -> bool:
        if mao is None:
            return False
        if not hard_mode:
            return mao.gesture in ("POINT", "PEACE", "OPEN")
        return mao.gesture == GUN_GESTO_HARD

    def counts_good_bad() -> Tuple[int, int, int]:
        good_hits = 0
        good_total = 0
        bad_hits = 0
        for tgt in gs.targets:
            if tgt.kind in GUN_BAD_TARGETS:
                if tgt.hit:
                    bad_hits += 1
            else:
                good_total += 1
                if tgt.hit:
                    good_hits += 1
        return good_hits, good_total, bad_hits

    def compute_award() -> Tuple[float, int, int, int]:
        good_hits, good_total, bad_hits = counts_good_bad()
        if hard_mode and good_total == 2:
            if good_hits == 2:
                base = 1.0
            elif good_hits == 1:
                base = HARD_DOUBLE_PARTIAL_SCORE
            else:
                base = 0.0
        else:
            base = 1.0 if good_hits >= 1 else 0.0
        award = base - bad_hits * GUN_VEG_PENALTY
        return award, good_hits, bad_hits, good_total

    def try_hit(hx: int, hy: int) -> bool:
        for tgt in gs.targets:
            if tgt.hit:
                continue
            if (hx - tgt.x) ** 2 + (hy - tgt.y) ** 2 <= raio ** 2:
                tgt.hit = True
                return True
        return False

    if gs.phase == "SIGNAL":
        # desbloqueio anti pre-shot
        if gs.bloqueado_mao:
            if mao is None:
                gs.t_mao_sumiu = gs.t_mao_sumiu or t
                if gs.t_mao_sumiu is not None and (t - gs.t_mao_sumiu) >= ANTI_PRESHOT_RESET_S:
                    gs.bloqueado_mao = False
            else:
                gs.t_mao_sumiu = None

        # disparo
        if (not gs.bloqueado_mao) and (mao is not None) and (t - gs.last_shot_t) >= SHOT_COOLDOWN_S and gesture_ok_for_shot():
            hx = int(clamp(mao.tip[0], 0.0, 1.0) * w)
            hy = int(clamp(mao.tip[1], 0.0, 1.0) * h)

            if try_hit(hx, hy):
                award, good_hits, bad_hits, good_total = compute_award()
                # primeira reação só conta se houver pelo menos um bom acerto
                if gs.last_reaction is None and good_hits > 0:
                    gs.last_reaction = t - gs.signal_t
                    gs.best_time = gs.last_reaction if gs.best_time is None else min(gs.best_time, gs.last_reaction)

                gs.last_hits = good_hits
                gs.last_total = good_total
                gs.last_bad_hits = bad_hits

                # termina a ronda quando todos os alvos tiverem sido atingidos (bons e maus)
                if all(tgt.hit for tgt in gs.targets):
                    gs.last_award = award
                    gs.score = max(0.0, gs.score + award)
                    gs.phase = "RESULT"
                    gs.last_shot_t = t

        # timeout
        if gs.phase == "SIGNAL" and (t - gs.signal_t) >= timeout:
            award, good_hits, bad_hits, good_total = compute_award()
            gs.last_award = award
            gs.last_hits = good_hits
            gs.last_total = good_total
            gs.last_bad_hits = bad_hits
            gs.score = max(0.0, gs.score + award)
            gs.phase = "RESULT"
            gs.last_shot_t = t
            if good_hits == 0:
                gs.last_reaction = None  # timeout sem bom hit

    if gs.phase == "RESULT" and (t - gs.last_shot_t) >= 1.2:
        gs.targets, gs.next_signal_t = gunslinger_new_targets(w, h, bank, hard_mode)
        gs.phase = "WAIT"
        gs.bloqueado_mao = False
        gs.t_mao_sumiu = None
        gs.last_reaction = None
        gs.last_hits = 0
        gs.last_total = 0
        gs.last_bad_hits = 0
        gs.last_award = 0.0

    titulo = "GUNSLINGER" if not hard_mode else "GUNSLINGER (HARD MODE)"

    # desenho 
    if gs.phase == "WAIT":
        for tgt in gs.targets:
            cv2.circle(frame, (tgt.x, tgt.y), raio, (140, 140, 140), -1, cv2.LINE_AA)
            cv2.circle(frame, (tgt.x, tgt.y), raio, (255, 255, 255), 2, cv2.LINE_AA)
            bank.draw_fruit(frame, tgt.kind, (tgt.x, tgt.y), size_px=raio * 2 - 6)
        draw_text(frame, f"{titulo}  |  Espera...", (20, 140), 0.9, 2)

    elif gs.phase == "SIGNAL":
        # pulse suave
        pulse = 0.6 + 0.4 * math.sin((t - gs.signal_t) * 4.0)  # ~4Hz
        v = int(110 + 70 * pulse)
        col = (v, v, 0)

        for tgt in gs.targets:
            base_col = (60, 200, 60) if tgt.hit else col
            cv2.circle(frame, (tgt.x, tgt.y), raio, base_col, -1, cv2.LINE_AA)
            cv2.circle(frame, (tgt.x, tgt.y), raio, (255, 255, 255), 2, cv2.LINE_AA)
            bank.draw_fruit(frame, tgt.kind, (tgt.x, tgt.y), size_px=raio * 2 - 6)

        if gs.bloqueado_mao:
            blend_rect(frame, 0, 0, w, h, 0.22, (0, 0, 255))
            draw_text(frame, "RETIRA A MAO DO ECRA!", (w // 2 - 235, h // 2 - 10), 1.0, 3)
            draw_text(frame, "Só depois podes disparar", (w // 2 - 225, h // 2 + 28), 0.8, 2)
        else:
            if hard_mode:
                if len(gs.targets) >= 2:
                    msg = "AGORA! (POINT) acerta nos alvos"
                elif gs.targets and gs.targets[0].is_teleport:
                    msg = "AGORA! (POINT) alvo teleporta na borda"
                else:
                    msg = "AGORA! (POINT) toca no alvo"
            else:
                msg = "AGORA! toca no alvo"
            draw_text(frame, msg, (20, 175), 0.8, 2)

        draw_text(frame, f"Tempo: {remaining:0.1f}s", (20, 242), 0.75, 2)

    else:  # RESULT
        for tgt in gs.targets:
            base_col = (60, 200, 60) if tgt.hit and tgt.kind not in GUN_BAD_TARGETS else (0, 200, 255)
            cv2.circle(frame, (tgt.x, tgt.y), raio, base_col, -1, cv2.LINE_AA)
            cv2.circle(frame, (tgt.x, tgt.y), raio, (255, 255, 255), 2, cv2.LINE_AA)
            bank.draw_fruit(frame, tgt.kind, (tgt.x, tgt.y), size_px=raio * 2 - 6)

        good_info = ""
        if gs.last_total > 0:
            good_info = f"{gs.last_hits}/{gs.last_total} frutas boas"
        veg_info = ""
        if gs.last_bad_hits > 0:
            veg_info = f"  |  Legumes acertados: {gs.last_bad_hits}"

        if gs.last_award > 0:
            msg = f"OK! (+{fmt_score(gs.last_award)}) {good_info}{veg_info}"
        elif gs.last_award < 0:
            msg = f"Ai! (-{fmt_score(-gs.last_award)}) {good_info}{veg_info}"
        else:
            if gs.last_hits == 0 and gs.last_bad_hits == 0:
                msg = "Sem acertos."
            else:
                msg = f"Equilíbrio 0. {good_info}{veg_info}"

        draw_text(frame, msg, (20, 175), 0.80, 2)
        draw_text(frame, f"Tempo: {remaining:0.1f}s", (20, 210), 0.75, 2)

    if mao is not None:
        hx = int(clamp(mao.tip[0], 0.0, 1.0) * w)
        hy = int(clamp(mao.tip[1], 0.0, 1.0) * h)
        cv2.circle(frame, (hx, hy), 10, (255, 255, 255), -1)
        cv2.circle(frame, (hx, hy), 10, (0, 0, 0), 2)
        draw_text(frame, f"gesto: {mao.gesture}", (20, 280), 0.7, 2)

    draw_text(frame, f"Score: {fmt_score(gs.score)}", (20, 315), 0.8, 2)
    if gs.best_time is not None:
        draw_text(frame, f"Melhor: {gs.best_time:0.3f}s", (20, 350), 0.8, 2)

    draw_text(frame, "Q: menu | ESC: sair", (20, h - 45), 0.6, 2)
    return frame, False

def draw_gunslinger_help_screen(frame: np.ndarray, bank: SpriteBank, hard_mode: bool) -> None:
    """Ecrã inicial de ajuda para o Gunslinger (frutas vs legumes)."""
    h, w = frame.shape[:2]
    tinted = overlay_tint(frame, (0, 0, 0), 0.65)
    frame[:] = tinted

    cx = w // 2
    draw_text(frame, "GUNSLINGER", (cx - 140, 70), 1.1, 3)
    draw_text(frame, "Dispara com o gesto POINT / mão aberta", (cx - 260, 110), 0.8, 2)

    base_y = 160
    icon_size = 50

    # Frutas boas
    draw_text(frame, "Frutas boas (acerta nestas):", (40, base_y), 0.7, 2)
    y_icons = base_y + 35
    x_icons = 80
    for kind in ("orange", "apple", "watermelon", "banana"):
        bank.draw_fruit(frame, kind, (x_icons, y_icons), icon_size)
        x_icons += icon_size + 24

    # Legumes maus
    y_veg_label = y_icons + 70
    draw_text(frame, "Legumes (evita disparar, tiram pontos):", (40, y_veg_label), 0.7, 2)
    y_veg = y_veg_label + 35
    x_veg = 110
    for kind in ("broccoli", "squash"):
        bank.draw_fruit(frame, kind, (x_veg, y_veg), icon_size)
        x_veg += icon_size + 60

    if hard_mode:
        extra_y = y_veg + 65
        draw_text(frame, "HARD MODE:", (40, extra_y), 0.7, 2)
        draw_text(frame, "- Pode haver 2 alvos ao mesmo tempo", (40, extra_y + 26), 0.6, 2)
        draw_text(frame, "- Pode haver um alvo que teleporta na borda", (40, extra_y + 48), 0.6, 2)

    footer_y = h - 60
    draw_text(frame, "Espaço: começar  |  Q: voltar ao menu", (cx - 230, footer_y), 0.75, 2)

# ----------------------------
# Menu render
# ----------------------------

def draw_menu(frame: np.ndarray, bank: SpriteBank, menu: MenuState, mao_menu: Optional[HandData]) -> None:
    h, w = frame.shape[:2]
    m = MENU_MARGIN_PX
    header_h = int(h * MENU_HEADER_H_RATIO)
    footer_h = int(h * MENU_FOOTER_H_RATIO)

    # header & footer
    blend_rect(frame, 0, 0, w, header_h, 0.35, (0, 0, 0))
    blend_rect(frame, 0, h - footer_h, w, h, 0.35, (0, 0, 0))

    # logo
    if bank.logo is not None:
        max_logo_w = int(w * 0.34)
        max_logo_h = int(header_h * 0.78)
        ratio = bank.logo.shape[0] / max(1, bank.logo.shape[1])
        logo_w = max_logo_w
        logo_h = int(logo_w * ratio)
        if logo_h > max_logo_h:
            logo_h = max_logo_h
            logo_w = int(logo_h / ratio) if ratio > 0 else max_logo_w
        logo_ok = bank.logo_scaled((max(2, logo_w), max(2, logo_h)))
        overlay_bgra(frame, logo_ok, (w // 2, int(header_h * 0.52)), (max(2, logo_w), max(2, logo_h)))

    pill_w = int(w * 0.30)
    pill_h = 42
    px1, py1 = m, int(header_h * 0.10)
    px2, py2 = px1 + pill_w, py1 + pill_h
    blend_rect(frame, px1, py1, px2, py2, 0.40, (0, 0, 0))
    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 2)
    draw_text(frame, f"HARD: {'ON' if menu.hard_mode else 'OFF'}", (px1 + 10, py1 + 28), 0.70, 2)

    # progress bar (toggle)
    if menu.t_hard_hold is not None:
        prog = clamp((now_s() - menu.t_hard_hold) / MENU_HARD_HOLD_S, 0.0, 1.0)
        bar_x1 = px1
        bar_y1 = py2 + 10
        bar_x2 = px2
        bar_y2 = bar_y1 + 10
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1)
        fill = int((bar_x2 - bar_x1) * prog)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill, bar_y2), (255, 255, 255), -1)
        draw_text(frame, f"THUMBS UP segurar {MENU_HARD_HOLD_S:0.1f}s", (bar_x1, bar_y2 + 22), 0.50, 2)
    else:
        draw_text(frame, f"THUMBS UP segurar {MENU_HARD_HOLD_S:0.1f}s p/ toggle", (m, int(header_h * 0.92)), 0.50, 2)

    # tiles
    center_y1 = header_h + m
    center_y2 = h - footer_h - m
    center_h = max(1, center_y2 - center_y1)

    tile_w = int((w - 2 * m - MENU_TILE_GAP_PX) / 2)
    tile_h = min(int(center_h), int(h * 0.46))
    tile_y = center_y1 + int((center_h - tile_h) / 2)

    xL1 = m
    xL2 = xL1 + tile_w
    xR1 = xL2 + MENU_TILE_GAP_PX
    xR2 = xR1 + tile_w

    cv2.rectangle(frame, (xL1, tile_y), (xL2, tile_y + tile_h), (40, 160, 40), -1)
    cv2.rectangle(frame, (xR1, tile_y), (xR2, tile_y + tile_h), (40, 40, 160), -1)

    icon_size = int(min(tile_w, tile_h) * 0.56)
    bank.draw_menu_icon(frame, "catcher", (xL1 + tile_w // 2, tile_y + int(tile_h * 0.42)), (icon_size, icon_size), "banana")
    bank.draw_menu_icon(frame, "gunslinger", (xR1 + tile_w // 2, tile_y + int(tile_h * 0.42)), (icon_size, icon_size), "watermelon")

    draw_text(frame, "Fruit Catcher", (xL1 + 18, tile_y + tile_h - 55), 0.85, 2)
    draw_text(frame, "Nariz move o cesto", (xL1 + 18, tile_y + tile_h - 25), 0.62, 2)
    draw_text(frame, "Gunslinger", (xR1 + 18, tile_y + tile_h - 55), 0.85, 2)
    draw_text(frame, "Indicador toca o alvo", (xR1 + 18, tile_y + tile_h - 25), 0.62, 2)

    if menu.selected == 0:
        cv2.rectangle(frame, (xL1 - 4, tile_y - 4), (xL2 + 4, tile_y + tile_h + 4), (255, 255, 255), 4)
        cv2.rectangle(frame, (xR1 - 2, tile_y - 2), (xR2 + 2, tile_y + tile_h + 2), (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (xR1 - 4, tile_y - 4), (xR2 + 4, tile_y + tile_h + 4), (255, 255, 255), 4)
        cv2.rectangle(frame, (xL1 - 2, tile_y - 2), (xL2 + 2, tile_y + tile_h + 2), (255, 255, 255), 2)

    # calibracao
    if menu.is_calibrating:
        draw_text(frame, "A calibrar... olha em frente", (m, header_h - 12), 0.55, 2)

    # flash toggle
    if now_s() < menu.hard_flash_until:
        blend_rect(frame, 0, 0, w, header_h, 0.18, (0, 0, 0))
        draw_text(frame, menu.hard_flash_txt, (w // 2 - 180, 32), 0.9, 3)

    # footer controls 
    y = h - footer_h + 34
    draw_text(frame, "Nariz Esq/Dir: escolher | Nariz Cima: confirmar | Nariz Baixo: sair", (m, y), 0.58, 2)
    y += 26
    draw_text(frame, "Boca aberta: confirmar | R: recalibrar | ESC: sair", (m, y), 0.58, 2)

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Não foi possível abrir a camara (ajusta CAMERA_INDEX).")

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

    contador_frames = 0
    contador_frames_mao = 0
    contador_frames_mao_menu = 0

    ultimo_face = FaceData((0.5, 0.5), 0.0, 1.0, False)
    ultima_mao: Optional[HandData] = None
    mao_menu: Optional[HandData] = None

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
        if key == 27:  # ESC
            break

        # face tracking
        if mode != Mode.GUN:
            contador_frames += 1
            if TRACK_CADA_N_FRAMES <= 1 or (contador_frames % TRACK_CADA_N_FRAMES) == 0 or (not ultimo_face.has_face):
                ultimo_face = face_tracker.process(frame_track)
        face = ultimo_face if mode != Mode.GUN else FaceData((0.5, 0.5), 0.0, 1.0, False)

        # hand tracking
        if mode == Mode.GUN:
            contador_frames_mao += 1
            if TRACK_CADA_N_FRAMES <= 1 or (contador_frames_mao % TRACK_CADA_N_FRAMES) == 0 or (ultima_mao is None):
                ultima_mao = hand_tracker.process(frame_track)

        if mode == Mode.MENU:
            contador_frames_mao_menu += 1
            if TRACK_CADA_N_FRAMES <= 1 or (contador_frames_mao_menu % TRACK_CADA_N_FRAMES) == 0:
                mao_menu = hand_tracker.process(frame_track)

        # ---------------- MENU ----------------
        if mode == Mode.MENU:
            menu_update(menu, face, mao_menu)
            draw_menu(frame, bank, menu, mao_menu)

            if key in (ord("r"), ord("R")):
                menu_reset(menu)

            do_confirm = (not menu.is_calibrating) and (menu_confirmed(menu) or key in (13, 10))
            if menu_exit(menu):
                break

            if do_confirm:
                if menu.selected == 0:
                    mode = Mode.FRUIT
                    catcher = CatcherState(start_t=0.0)  # será definido quando sair do ecrã de ajuda
                else:
                    mode = Mode.GUN
                    contador_frames_mao = 0
                    ultima_mao = None
                    targets, proximo = gunslinger_new_targets(w, h, bank, hard_mode=menu.hard_mode)
                    guns = GunslingerState(
                        start_t=0.0,  # será definido quando sair do ecrã de ajuda
                        phase="WAIT",
                        score=0.0,
                        best_time=None,
                        signal_t=0.0,
                        next_signal_t=proximo,
                        targets=targets,
                        last_shot_t=0.0,
                    )

        # ---------------- FRUIT ----------------
        elif mode == Mode.FRUIT:
            assert catcher is not None
            if catcher.showing_help:
                draw_catcher_help_screen(frame, bank, menu.hard_mode)
                if key == 32:  # espaço
                    catcher.showing_help = False
                    catcher.start_t = now_s()
                    catcher.next_spawn_t = catcher.start_t  # spawn imediato da primeira fruta
                elif key in (ord("q"), ord("Q")):
                    mode = Mode.MENU
                    menu_reset(menu)
                    catcher = None
            else:
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
            if guns.showing_help:
                draw_gunslinger_help_screen(frame, bank, menu.hard_mode)
                if key == 32:  # espaço
                    guns.showing_help = False
                    guns.start_t = now_s()
                    guns.targets, guns.next_signal_t = gunslinger_new_targets(w, h, bank, hard_mode=menu.hard_mode)
                    guns.phase = "WAIT"
                    guns.last_reaction = None
                    guns.last_hits = 0
                    guns.last_total = 0
                    guns.last_bad_hits = 0
                    guns.last_award = 0.0
                    guns.bloqueado_mao = False
                    guns.t_mao_sumiu = None
                elif key in (ord("q"), ord("Q")):
                    mode = Mode.MENU
                    menu_reset(menu)
                    guns = None
                    ultima_mao = None
            else:
                frame, done = run_gunslinger(frame, ultima_mao, guns, bank, hard_mode=menu.hard_mode)

                if done:
                    if key in (13, 10, ord("q"), ord("Q")):
                        mode = Mode.MENU
                        menu_reset(menu)
                        guns = None
                        ultima_mao = None
                else:
                    if key in (ord("q"), ord("Q")):
                        mode = Mode.MENU
                        menu_reset(menu)
                        guns = None
                        ultima_mao = None

        # debug do nariz
        if SHOW_NOSE_CURSOR and face.has_face and mode != Mode.GUN:
            cx = int(clamp(face.nose[0], 0.0, 1.0) * w)
            cy = int(clamp(face.nose[1], 0.0, 1.0) * h)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(NOME_JANELA, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
