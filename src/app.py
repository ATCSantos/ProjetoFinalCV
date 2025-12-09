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
TRACK_CADA_N_FRAMES = 2

PASTA_ASSETS = (Path(__file__).resolve().parent.parent / "assets")

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
MENU_HARD_HOLD_S = 0.85          # +0.05s para reduzir "finnicky"
MENU_HARD_COOLDOWN_S = 0.60
MENU_HARD_FLASH_S = 0.90
MENU_HARD_GRACE_S = 0.18         # tolerância a falhas rápidas no tracking (ligeiramente maior)

# Fruit Catcher
CATCHER_ROUND_S = 45
FRUIT_FALL_SPEED_PX = 4.2
FRUIT_FALL_SPEED_INC = 0.02
FRUIT_SPAWN_EVERY_S = (0.45, 0.90)
BASKET_W = 160
BASKET_H = 30

ENABLE_MOUTH_TO_CATCH = False
MOUTH_OPEN_THRESHOLD = 0.055

# Olhos fechados (Hard Mode) — mais exigente (evita cesto sempre ativo)
EYES_CLOSED_RATIO = 0.55
EYES_CLOSED_HOLD_S = 0.18

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

# Hard Mode extras (Gunslinger)
GUN_HARD_P_TWO_TARGETS = 0.35          # chance de 2 frutos ao mesmo tempo
GUN_HARD_P_BLINKER = 0.25              # chance de alvo piscante que "teleporta"
GUN_BLINK_INTERVAL_S = 0.12            # quão rápido teleporta
GUN_BLINK_BORDER_MARGIN = 22           # margem da borda

# Anti early bird
ANTI_PRESHOT_RESET_S = 0.20

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

def draw_end_screen(frame: np.ndarray, titulo: str, subtitulo: str) -> np.ndarray:
    h, w = frame.shape[:2]
    frame = overlay_tint(frame, (0, 0, 0), 0.62)  # ligeiramente mais escuro p/ consistência
    draw_text(frame, titulo, (w // 2 - 220, h // 2 - 10), 1.05, 3)
    draw_text(frame, subtitulo, (w // 2 - 240, h // 2 + 35), 0.85, 2)
    draw_text(frame, "ENTER ou Q: voltar ao menu", (w // 2 - 235, h // 2 + 80), 0.75, 2)
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
        # mais permissivo (reduz falhas do thumbs up)
        return lm[self.THUMB_TIP].y < (lm[self.THUMB_IP].y - 0.010)

    def _thumb_side(self, lm) -> bool:
        return abs(lm[self.THUMB_TIP].x - lm[self.THUMB_IP].x) > 0.030

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

        # thumbs: aceita também situações com 1 dedo extra (tracking imperfeito)
        if thumb_up and (cnt <= 2) and (not middle) and (not ring) and (not pinky):
            return "THUMBS"
        if thumb and (cnt <= 2) and (not middle) and (not ring) and (not pinky):
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
        sprite = self.fruits.get(kind)
        if sprite is None:
            cv2.circle(frame, center, size_px // 2, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, center, size_px // 2, (255, 255, 255), 2, cv2.LINE_AA)
            return
        sprite_scaled = self.fruit_scaled(kind, size_px)
        overlay_bgra(frame, sprite_scaled, center, (size_px, size_px))


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
    done = remaining <= 0.0

    if done:
        frame = draw_end_screen(frame, "FIM!", f"Pontuacao: {cs.score}")
        return frame, True

    bx = int(clamp(face.nose[0], 0.0, 1.0) * w) if face.has_face else w // 2
    basket_x1 = int(clamp(bx - BASKET_W // 2, 0, w - BASKET_W))
    basket_y1 = h - 70
    basket_x2 = basket_x1 + BASKET_W
    basket_y2 = basket_y1 + BASKET_H

    basket_active = (not cs.mouth_required) or (face.mouth_open >= MOUTH_OPEN_THRESHOLD)

    eyes_thr = max(0.06, float(eyes_base) * EYES_CLOSED_RATIO)
    olhos_fechados_raw = face.has_face and (face.eyes_open <= eyes_thr)

    if hard_mode:
        if olhos_fechados_raw:
            cs.t_olhos_fechados = cs.t_olhos_fechados or t
        else:
            cs.t_olhos_fechados = None

        olhos_fechados_ok = cs.t_olhos_fechados is not None and (t - cs.t_olhos_fechados) >= EYES_CLOSED_HOLD_S
        basket_active = basket_active and olhos_fechados_ok

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

    titulo = "FRUIT CATCHER" if not hard_mode else "FRUIT CATCHER (HARD MODE)"
    draw_text(frame, f"{titulo}  |  Score: {cs.score}", (20, 140), 0.9, 2)
    draw_text(frame, f"Tempo: {remaining:0.1f}s", (20, 175), 0.8, 2)

    if hard_mode:
        estado_olhos = "FECHADOS" if olhos_fechados_raw else "ABERTOS"
        estado_cesto = "ATIVO" if basket_active else "BLOQUEADO"
        draw_text(frame, f"Olhos: {estado_olhos}  (eyes:{face.eyes_open:0.3f} thr:{eyes_thr:0.3f})", (20, 210), 0.65, 2)
        draw_text(frame, f"Cesto: {estado_cesto} (fecha {EYES_CLOSED_HOLD_S:0.2f}s)", (20, 242), 0.70, 2)

    draw_text(frame, "Q: menu | ESC: sair", (20, h - 45), 0.6, 2)
    return frame, False


@dataclass
class GunslingerState:
    start_t: float
    phase: str
    score: int
    best_time: Optional[float]
    signal_t: float
    next_signal_t: float
    targets: List[Tuple[int, int]]
    target_kinds: List[str]
    last_shot_t: float
    last_reaction: Optional[float] = None

    blink: bool = False
    blink_next_move: float = 0.0

    bloqueado_mao: bool = False
    t_mao_sumiu: Optional[float] = None


def _rand_center_target(w: int, h: int) -> Tuple[int, int]:
    tx = random.randint(w // 3, int(w * 0.66))
    ty = random.randint(h // 3, int(h * 0.66))
    return (tx, ty)

def _rand_border_target(w: int, h: int, margin: int) -> Tuple[int, int]:
    side = random.randint(0, 3)
    if side == 0:
        return (random.randint(margin, w - margin), margin)
    if side == 1:
        return (random.randint(margin, w - margin), h - margin)
    if side == 2:
        return (margin, random.randint(margin, h - margin))
    return (w - margin, random.randint(margin, h - margin))

def gunslinger_new_round(w: int, h: int, bank: SpriteBank, hard_mode: bool) -> Tuple[List[Tuple[int, int]], List[str], float, bool]:
    next_signal = now_s() + random.uniform(*SIGNAL_DELAY_S)
    blink = False

    if hard_mode:
        r = random.random()
        if r < GUN_HARD_P_BLINKER:
            blink = True
            targets = [_rand_border_target(w, h, GUN_BLINK_BORDER_MARGIN)]
        elif r < (GUN_HARD_P_BLINKER + GUN_HARD_P_TWO_TARGETS):
            t1 = _rand_center_target(w, h)
            t2 = _rand_center_target(w, h)
            if (t1[0] - t2[0])**2 + (t1[1] - t2[1])**2 < (TARGET_RADIUS_HARD * 2.2) ** 2:
                t2 = (int(clamp(t2[0] + TARGET_RADIUS_HARD * 2, 0, w)), int(clamp(t2[1] - TARGET_RADIUS_HARD * 2, 0, h)))
            targets = [t1, t2]
        else:
            targets = [_rand_center_target(w, h)]
    else:
        targets = [_rand_center_target(w, h)]

    kinds = [(random.choice(bank.fruit_keys) if bank.fruit_keys else "orange") for _ in targets]
    return targets, kinds, next_signal, blink


def run_gunslinger(frame: np.ndarray, mao: Optional[HandData], gs: GunslingerState, bank: SpriteBank, hard_mode: bool) -> Tuple[np.ndarray, bool]:
    h, w = frame.shape[:2]
    t = now_s()

    remaining = max(0.0, GUN_TOTAL_ROUND_S - (t - gs.start_t))
    if remaining <= 0.0:
        frame = draw_end_screen(frame, "FIM!", f"Pontuacao: {gs.score}")
        return frame, True

    raio = TARGET_RADIUS_HARD if hard_mode else TARGET_RADIUS
    timeout = GUN_ROUND_TIMEOUT_HARD_S if hard_mode else GUN_ROUND_TIMEOUT_S

    if gs.phase == "WAIT" and t >= gs.next_signal_t:
        gs.phase = "SIGNAL"
        gs.signal_t = t
        gs.last_reaction = None
        gs.bloqueado_mao = (mao is not None)
        gs.t_mao_sumiu = None
        if gs.blink:
            gs.blink_next_move = t + GUN_BLINK_INTERVAL_S

    if gs.phase == "SIGNAL":
        if hard_mode and gs.blink and (t >= gs.blink_next_move):
            gs.targets[0] = _rand_border_target(w, h, GUN_BLINK_BORDER_MARGIN)
            gs.blink_next_move = t + GUN_BLINK_INTERVAL_S

        if gs.bloqueado_mao:
            if mao is None:
                gs.t_mao_sumiu = gs.t_mao_sumiu or t
                if gs.t_mao_sumiu is not None and (t - gs.t_mao_sumiu) >= ANTI_PRESHOT_RESET_S:
                    gs.bloqueado_mao = False
            else:
                gs.t_mao_sumiu = None

        if t - gs.signal_t >= timeout:
            gs.phase = "RESULT"
            gs.last_shot_t = t
            gs.last_reaction = None

        if (not gs.bloqueado_mao) and mao is not None and (t - gs.last_shot_t) >= SHOT_COOLDOWN_S:
            hx = int(clamp(mao.tip[0], 0.0, 1.0) * w)
            hy = int(clamp(mao.tip[1], 0.0, 1.0) * h)

            gesto_ok = mao.gesture in ("POINT", "PEACE", "OPEN")
            if hard_mode:
                gesto_ok = (mao.gesture == GUN_GESTO_HARD)

            if gesto_ok:
                for (tx, ty) in gs.targets:
                    if (hx - tx) ** 2 + (hy - ty) ** 2 <= raio ** 2:
                        rt = t - gs.signal_t
                        gs.last_reaction = rt
                        gs.best_time = rt if gs.best_time is None else min(gs.best_time, rt)
                        gs.score += 1
                        gs.phase = "RESULT"
                        gs.last_shot_t = t
                        break

    if gs.phase == "RESULT" and (t - gs.last_shot_t) >= 1.2:
        gs.targets, gs.target_kinds, gs.next_signal_t, gs.blink = gunslinger_new_round(w, h, bank, hard_mode=hard_mode)
        gs.phase = "WAIT"
        gs.bloqueado_mao = False
        gs.t_mao_sumiu = None
        gs.blink_next_move = 0.0

    titulo = "GUNSLINGER" if not hard_mode else "GUNSLINGER (HARD MODE)"

    if gs.phase == "WAIT":
        for (tx, ty), kind in zip(gs.targets, gs.target_kinds):
            cv2.circle(frame, (tx, ty), raio, (140, 140, 140), -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), raio, (255, 255, 255), 2, cv2.LINE_AA)
            bank.draw_fruit(frame, kind, (tx, ty), size_px=raio * 2 - 6)
        draw_text(frame, f"{titulo}  |  Espera...", (20, 140), 0.9, 2)

    elif gs.phase == "SIGNAL":
        pulse = 0.5 + 0.5 * math.sin((t - gs.signal_t) * 10.0)
        col = (int(80 + 175 * pulse), int(80 + 175 * pulse), 0)

        for (tx, ty), kind in zip(gs.targets, gs.target_kinds):
            cv2.circle(frame, (tx, ty), raio, col, -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), raio, (255, 255, 255), 2, cv2.LINE_AA)
            bank.draw_fruit(frame, kind, (tx, ty), size_px=raio * 2 - 6)

        if gs.bloqueado_mao:
            frame[:] = overlay_tint(frame, (0, 0, 255), 0.28)
            draw_text(frame, "RETIRA A MAO DO ECRA!", (w // 2 - 240, h // 2 - 10), 1.0, 3)
            draw_text(frame, "So depois podes disparar", (w // 2 - 220, h // 2 + 28), 0.8, 2)
            draw_text(frame, "(anti pre-shot ativo)", (w // 2 - 165, h // 2 + 60), 0.7, 2)
        else:
            msg = "AGORA! toca na fruta"
            if hard_mode:
                extras = []
                if len(gs.targets) == 2:
                    extras.append("2 alvos")
                if gs.blink:
                    extras.append("alvo piscante")
                extra_txt = (" | " + ", ".join(extras)) if extras else ""
                msg = f"AGORA! (só POINT){extra_txt}"
            draw_text(frame, msg, (20, 175), 0.8, 2)

        draw_text(frame, f"Tempo: {remaining:0.1f}s", (20, 210), 0.75, 2)

    else:
        for (tx, ty), kind in zip(gs.targets, gs.target_kinds):
            cv2.circle(frame, (tx, ty), raio, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), raio, (255, 255, 255), 2, cv2.LINE_AA)
            bank.draw_fruit(frame, kind, (tx, ty), size_px=raio * 2 - 6)
        if gs.last_reaction is not None:
            draw_text(frame, f"OK! {gs.last_reaction:0.3f}s", (20, 175), 0.85, 2)
        else:
            draw_text(frame, "Falhou (timeout)", (20, 175), 0.85, 2)
        draw_text(frame, f"Tempo: {remaining:0.1f}s", (20, 210), 0.75, 2)

    if mao is not None:
        hx = int(clamp(mao.tip[0], 0.0, 1.0) * w)
        hy = int(clamp(mao.tip[1], 0.0, 1.0) * h)
        cv2.circle(frame, (hx, hy), 10, (255, 255, 255), -1)
        cv2.circle(frame, (hx, hy), 10, (0, 0, 0), 2)
        draw_text(frame, f"gesto: {mao.gesture}", (20, 245), 0.7, 2)

    draw_text(frame, f"Score: {gs.score}", (20, 280), 0.8, 2)
    if gs.best_time is not None:
        draw_text(frame, f"Melhor: {gs.best_time:0.3f}s", (20, 315), 0.8, 2)

    draw_text(frame, "Q: menu | ESC: sair", (20, h - 45), 0.6, 2)
    return frame, False


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
        if key == 27:
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

        if mode == Mode.MENU:
            menu_update(menu, face, mao_menu)

            if logo is not None:
                target_w = int(w * 0.42)
                target_h = int(target_w * (logo.shape[0] / logo.shape[1]))
                logo_ok = bank.logo_scaled((target_w, target_h))
                overlay_bgra(frame, logo_ok, (w // 2, int(h * 0.16)), (target_w, target_h))

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

            t1 = "1) Fruit Catcher" + (" (Hard Mode)" if menu.hard_mode else "")
            t2 = "2) Gunslinger" + (" (Hard Mode)" if menu.hard_mode else "")
            draw_text(frame, t1, (x1 + 22, y0 + tile_h - 45), 0.80, 2)
            draw_text(frame, "Nariz move (cesto)", (x1 + 22, y0 + tile_h - 18), 0.58, 2)
            draw_text(frame, t2, (x3 + 22, y0 + tile_h - 45), 0.80, 2)
            draw_text(frame, "Dedo toca (fruta)", (x3 + 22, y0 + tile_h - 18), 0.58, 2)

            draw_text(frame, f"HARD MODE: {'ON' if menu.hard_mode else 'OFF'}", (20, 140), 0.95, 3)
            draw_text(frame, f"Toggle: thumbs up (segura {MENU_HARD_HOLD_S:0.1f}s)", (20, 175), 0.65, 2)

            if menu.t_hard_hold is not None:
                prog = clamp((now_s() - menu.t_hard_hold) / MENU_HARD_HOLD_S, 0.0, 1.0)
                draw_text(frame, f"👍 a segurar: {prog*100:0.0f}%", (20, 205), 0.70, 2)
            if mao_menu is not None:
                draw_text(frame, f"gesto: {mao_menu.gesture}", (20, 235), 0.65, 2)

            if menu.hard_mode:
                draw_text(frame, "Hard Mode:", (20, 265), 0.70, 2)
                draw_text(frame, "- Catcher: só conta com olhos fechados", (20, 292), 0.62, 2)
                draw_text(frame, "- Gunslinger: só POINT + 2 alvos / alvo piscante", (20, 318), 0.62, 2)

            draw_text(frame, "Controlo menu:", (20, h - 112), 0.62, 2)
            draw_text(frame, "Nariz Esq/Dir: escolher | Nariz Cima: confirmar | Nariz Baixo: sair", (20, h - 85), 0.58, 2)
            draw_text(frame, "Boca aberta: confirmar | R: recalibrar | ESC: sair", (20, h - 58), 0.58, 2)

            if now_s() < menu.hard_flash_until:
                frame[:] = overlay_tint(frame, (0, 0, 0), 0.20)
                draw_text(frame, menu.hard_flash_txt, (w // 2 - 230, 110), 1.0, 3)

            if menu.is_calibrating:
                draw_text(frame, "A calibrar... olha em frente", (20, 120), 0.65, 2)

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
                    contador_frames_mao = 0
                    ultima_mao = None
                    targets, kinds, next_sig, blink = gunslinger_new_round(w, h, bank, hard_mode=menu.hard_mode)
                    guns = GunslingerState(
                        start_t=now_s(),
                        phase="WAIT",
                        score=0,
                        best_time=None,
                        signal_t=0.0,
                        next_signal_t=next_sig,
                        targets=targets,
                        target_kinds=kinds,
                        last_shot_t=0.0,
                        blink=blink,
                        blink_next_move=0.0,
                    )

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

        elif mode == Mode.GUN:
            assert guns is not None
            mao = ultima_mao
            frame, done = run_gunslinger(frame, mao, guns, bank, hard_mode=menu.hard_mode)
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

        if face.has_face and mode != Mode.GUN:
            cx = int(clamp(face.nose[0], 0.0, 1.0) * w)
            cy = int(clamp(face.nose[1], 0.0, 1.0) * h)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 10, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(NOME_JANELA, frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
