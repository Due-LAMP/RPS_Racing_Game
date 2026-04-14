import tflite_runtime.interpreter as tflite
import numpy as np
import time
import cv2
import math
import threading

# ─── Model Setup ──────────────────────────────────────────
# modelPath = "best_float32.tflite"   # 10MB, 고정밀 but 느림
modelPath = "best_float16.tflite"      # 5MB, float32과 거의 동일 정밀도 + 2x 빠른 로딩
# modelPath = "best_full_integer_quant.tflite"  # 2.8MB, INT8 양자화
print("model path:", modelPath)

interpreter = tflite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]
output_dtype = output_details[0]["dtype"]
input_scale, input_zero = input_details[0]["quantization"]
output_scale, output_zero = output_details[0]["quantization"]
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]

# ─── Detection Constants ─────────────────────────────────
ansToText = {0: "scissors", 1: "rock", 2: "paper"}
ansIcon = {0: "S", 1: "R", 2: "P"}
colorList = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
IMG_SIZE = 320
CONF_TH = 0.5
IOU_TH = 0.45

# ─── Game Constants ──────────────────────────────────────
BOARD_SIZE = 20
GOAL_POS = BOARD_SIZE - 1
ROUND_SEC = 5.0       # 카운트다운 시간
RESOLVE_SEC = 2.8     # 결과 표시 + 애니메이션 시간
FLASH_SEC = 0.8       # 결과 플래시 (애니메이션 전 대기)
VOTE_WINDOW = 1.0     # 제스처 투표 윈도우 (마지막 N초)

SPECIAL_TILES = {
    3: "rocket",
    6: "broken",
    9: "ladder_up",
    12: "mine",
    15: "warp",
    17: "ladder_down",
}
LADDER_MAP = {9: 14, 17: 11}

# Phase
PHASE_COUNTDOWN = 0
PHASE_RESOLVE = 1
PHASE_GAMEOVER = 2

# Colors — neon palette
P1_COLOR = (100, 255, 140)
P1_MID   = (60, 200, 100)
P1_DARK  = (30, 120, 50)
P1_GLOW  = (80, 255, 120)
P2_COLOR = (255, 180, 80)
P2_MID   = (220, 130, 50)
P2_DARK  = (150, 80, 25)
P2_GLOW  = (255, 160, 60)
GOLD     = (80, 220, 255)
NEON_CYAN   = (240, 220, 60)
NEON_PINK   = (180, 80, 255)
BG_DARK     = (18, 18, 24)
BG_PANEL    = (25, 25, 32)
TILE_BORDER = (60, 60, 75)

TILE_COLORS = {
    "rocket":      (40, 70, 200),
    "broken":      (55, 50, 50),
    "ladder_up":   (30, 140, 55),
    "mine":        (30, 80, 170),
    "warp":        (160, 90, 30),
    "ladder_down": (55, 130, 165),
}
TILE_GLOW = {
    "rocket":      (80, 130, 255),
    "broken":      (100, 80, 80),
    "ladder_up":   (80, 230, 110),
    "mine":        (60, 130, 240),
    "warp":        (240, 170, 70),
    "ladder_down": (110, 200, 240),
}
TILE_ICON = {
    "rocket": ">>", "broken": "XX", "ladder_up": "^^",
    "mine": "**",   "warp": "<>",   "ladder_down": "vv",
}
TILE_LABEL = {
    "rocket": "ROCKET", "broken": "BROKEN", "ladder_up": "LADDER",
    "mine": "MINE",     "warp": "WARP",     "ladder_down": "SLIDE",
}


# ─── Detection Functions ─────────────────────────────────

def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)
    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))
    pad_w, pad_h = nw - new_w, nh - new_h
    pad_x, pad_y = pad_w // 2, pad_h // 2
    padded = cv2.copyMakeBorder(
        resized, pad_y, pad_h - pad_y, pad_x, pad_w - pad_x,
        cv2.BORDER_CONSTANT, value=color,
    )
    return padded, r, pad_x, pad_y


def nms(boxes, scores, iou_th):
    if not boxes:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)
        order = order[np.where(iou <= iou_th)[0] + 1]
    return keep


def infer_gestures(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_lb, r, pad_x, pad_y = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))
    img = img_lb.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    if input_dtype == np.int8:
        img = ((img / input_scale) + input_zero).astype(np.int8)
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_index)[0].transpose()
    if output_dtype == np.int8:
        raw = (raw.astype(np.float32) - output_zero) * output_scale

    boxes, scores, class_ids = [], [], []
    for det in raw:
        cx, cy, w, h = det[:4]
        cls_scores = det[4:7]
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id])
        if score < CONF_TH:
            continue
        cx *= IMG_SIZE; cy *= IMG_SIZE
        w *= IMG_SIZE;  h *= IMG_SIZE
        x1 = (cx - w / 2 - pad_x) / r
        y1 = (cy - h / 2 - pad_y) / r
        x2 = (cx + w / 2 - pad_x) / r
        y2 = (cy + h / 2 - pad_y) / r
        boxes.append([
            int(np.clip(x1, 0, frame.shape[1])),
            int(np.clip(y1, 0, frame.shape[0])),
            int(np.clip(x2, 0, frame.shape[1])),
            int(np.clip(y2, 0, frame.shape[0])),
        ])
        scores.append(score)
        class_ids.append(cls_id)

    keep = nms(boxes, scores, IOU_TH)
    detections = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cid, sc = class_ids[i], scores[i]
        cx = (x1 + x2) // 2
        detections.append({"box": (x1, y1, x2, y2), "cid": cid, "score": sc, "cx": cx})
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[cid], 2)
        cv2.putText(frame, f"{ansToText[cid]} {sc:.2f}",
                    (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_PLAIN, 1.2, colorList[cid], 2)
    return detections


def select_players(detections, frame_w):
    p1 = p2 = p1_det = p2_det = None
    if not detections:
        return p1, p2, p1_det, p2_det
    if len(detections) == 1:
        d = detections[0]
        if d["cx"] < frame_w // 2:
            p1, p1_det = d["cid"], d
        else:
            p2, p2_det = d["cid"], d
        return p1, p2, p1_det, p2_det
    left = min(detections, key=lambda d: d["cx"])
    right = max(detections, key=lambda d: d["cx"])
    return left["cid"], right["cid"], left, right


# ─── Gesture Voting ──────────────────────────────────────

def vote_gesture(history, now, window=VOTE_WINDOW):
    recent = [g for t, g in history if now - t < window]
    if not recent:
        return None
    counts = {}
    for g in recent:
        counts[g] = counts.get(g, 0) + 1
    return max(counts, key=counts.get)


# ─── Game Logic ──────────────────────────────────────────

def clamp_pos(pos):
    return int(np.clip(pos, 0, GOAL_POS))


def resolve_round(g1, g2, positions, stuns, round_idx):
    logs = []
    can_play = [True, True]
    for i in range(2):
        if stuns[i] > 0:
            can_play[i] = False
            stuns[i] -= 1

    g1_txt = ansToText.get(g1, "none") if g1 is not None else "none"
    g2_txt = ansToText.get(g2, "none") if g2 is not None else "none"

    delta = [0, 0]
    beats = {1: 0, 0: 2, 2: 1}

    if not can_play[0] and not can_play[1]:
        logs.append("Both stunned! No action.")
    elif can_play[0] and not can_play[1]:
        delta[0] += 2
        logs.append(f"P2 stunned -> P1 +2")
    elif not can_play[0] and can_play[1]:
        delta[1] += 2
        logs.append(f"P1 stunned -> P2 +2")
    elif g1 is None and g2 is None:
        logs.append("No hands detected!")
    elif g1 is None:
        delta[1] += 1
        logs.append(f"P1 missing -> P2 +1")
    elif g2 is None:
        delta[0] += 1
        logs.append(f"P2 missing -> P1 +1")
    elif g1 == g2:
        if g1 == 1:
            delta[0] -= 1; delta[1] -= 1
            logs.append("TIE Rock -> both -1")
        elif g1 == 0:
            positions[0], positions[1] = positions[1], positions[0]
            logs.append("TIE Scissors -> SWAP!")
        else:
            delta[0] += 1; delta[1] += 1
            logs.append("TIE Paper -> both +1")
    else:
        p1_win = beats[g1] == g2
        winner = 0 if p1_win else 1
        loser = 1 - winner
        wg = g1 if p1_win else g2
        if wg == 1:
            delta[loser] -= 3
            logs.append(f"Rock WIN! P{loser+1} -3")
        elif wg == 0:
            delta[winner] += 2
            logs.append(f"Scissors WIN! P{winner+1} +2")
        else:
            delta[winner] += 1; delta[loser] -= 1
            logs.append(f"Paper WIN! P{winner+1}+1 P{loser+1}-1")

    for i in range(2):
        positions[i] = clamp_pos(positions[i] + delta[i])

    apply_specials(positions, stuns, logs)

    gw = None
    if positions[0] >= GOAL_POS and positions[1] >= GOAL_POS:
        gw = -1
    elif positions[0] >= GOAL_POS:
        gw = 0
    elif positions[1] >= GOAL_POS:
        gw = 1

    return g1_txt, g2_txt, logs, gw


def apply_specials(positions, stuns, logs):
    for player in range(2):
        for _ in range(3):
            tile = positions[player]
            effect = SPECIAL_TILES.get(tile)
            if effect is None:
                break
            if effect == "rocket":
                positions[player] = clamp_pos(positions[player] + 4)
                logs.append(f"P{player+1} ROCKET! +4")
            elif effect == "broken":
                stuns[player] = max(stuns[player], 1)
                logs.append(f"P{player+1} BROKEN! skip next")
                break
            elif effect in ("ladder_up", "ladder_down"):
                dest = LADDER_MAP[tile]
                tag = "LADDER UP" if effect == "ladder_up" else "SLIDE DOWN"
                positions[player] = dest
                logs.append(f"P{player+1} {tag} {tile}->{dest}")
            elif effect == "mine":
                positions[player] = clamp_pos(positions[player] - 2)
                logs.append(f"P{player+1} MINE! -2")
                break
            elif effect == "warp":
                other = 1 - player
                positions[player], positions[other] = positions[other], positions[player]
                logs.append("WARP! P1<->P2")
                break


# ─── Drawing Helpers ─────────────────────────────────────

def rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def gradient_v(img, y1, y2, x1, x2, c_top, c_bot):
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    t = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1, 1)
    top = np.array(c_top, dtype=np.float32).reshape(1, 1, 3)
    bot = np.array(c_bot, dtype=np.float32).reshape(1, 1, 3)
    grad = (top + (bot - top) * t).astype(np.uint8)
    grad = np.broadcast_to(grad, (h, w, 3)).copy()
    img[y1:y2, x1:x2] = grad


def soft_ring(img, cx, cy, r, color):
    cv2.circle(img, (cx, cy), r + 6, tuple(max(0, c // 4) for c in color), 2)
    cv2.circle(img, (cx, cy), r + 3, tuple(max(0, c // 2) for c in color), 2)


def put_text_shadow(img, text, pos, font, scale, color, thick=2, shadow=(10, 10, 15)):
    cv2.putText(img, text, (pos[0] + 1, pos[1] + 1), font, scale, shadow, thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)


def text_centered(img, text, cx, cy, font, scale, color, thick=2):
    sz = cv2.getTextSize(text, font, scale, thick)[0]
    put_text_shadow(img, text, (cx - sz[0] // 2, cy + sz[1] // 2), font, scale, color, thick)


# ─── Board Drawing ───────────────────────────────────────

def build_cells(bw, bh, m_top=56, m_bottom=130):
    cols, rows = 5, 4
    mx = 12
    aw = bw - 2 * mx
    ah = bh - m_top - m_bottom
    cw = aw // cols
    ch = ah // rows
    cells = []
    for r in range(rows):
        y = bh - m_bottom - (r + 1) * ch
        order = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in order:
            cells.append((mx + c * cw, y, cw, ch))
    return cells[:BOARD_SIZE]


def cell_center(cells, idx):
    x, y, w, h = cells[idx]
    return (x + w // 2, y + h // 2)


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp_board(cells, old_p, new_p, t):
    if old_p == new_p:
        return cell_center(cells, new_p)
    fpos = old_p + (new_p - old_p) * t
    fpos = max(0.0, min(float(GOAL_POS), fpos))
    lo = int(fpos)
    hi = min(lo + 1, GOAL_POS)
    frac = fpos - lo
    c1 = cell_center(cells, lo)
    c2 = cell_center(cells, hi)
    return (int(c1[0] + (c2[0] - c1[0]) * frac),
            int(c1[1] + (c2[1] - c1[1]) * frac))


def draw_token(panel, px, py, player, pulse=False, trail=False):
    color = P1_COLOR if player == 0 else P2_COLOR
    mid   = P1_MID  if player == 0 else P2_MID
    dark  = P1_DARK if player == 0 else P2_DARK
    glow  = P1_GLOW if player == 0 else P2_GLOW

    r = 14
    if pulse:
        r += int(3 * math.sin(time.time() * 6))

    # lightweight outer glow (2 rings instead of full overlay copy)
    soft_ring(panel, px, py, r, glow)

    # shadow
    cv2.circle(panel, (px + 2, py + 3), r, (8, 8, 12), -1)

    # layered body
    cv2.circle(panel, (px, py), r, dark, -1)
    cv2.circle(panel, (px, py), r - 3, mid, -1)
    cv2.circle(panel, (px - 1, py - 1), r - 5, color, -1)

    # specular highlight
    cv2.circle(panel, (px - r // 3, py - r // 3), max(2, r // 4),
               (255, 255, 255), -1)

    # border
    cv2.circle(panel, (px, py), r, dark, 2)

    # label
    lbl = "P1" if player == 0 else "P2"
    sz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)[0]
    cv2.putText(panel, lbl, (px - sz[0] // 2, py + sz[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)


def draw_board(bw, bh, positions, old_positions, stuns, phase, elapsed,
               round_idx, live_g1, live_g2, rnd_g1_txt, rnd_g2_txt,
               result_logs, history_logs, anim_t, game_winner):
    panel = np.full((bh, bw, 3), BG_DARK, dtype=np.uint8)

    # ── gradient header bar
    gradient_v(panel, 0, 50, 0, bw, (40, 30, 65), BG_DARK)

    # neon accent line under header
    cv2.line(panel, (10, 50), (bw - 10, 50), NEON_CYAN, 1, cv2.LINE_AA)

    # title
    put_text_shadow(panel, "RPS RACING", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, GOLD, 2)
    put_text_shadow(panel, f"Round {round_idx}",
                    (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 210), 1)
    put_text_shadow(panel, "[Q]uit  [R]eset",
                    (bw - 130, 44), cv2.FONT_HERSHEY_PLAIN, 1.0, (130, 130, 145), 1)

    cells = build_cells(bw, bh)

    # ── ladder / slide connections (behind tiles)
    for start, end in LADDER_MAP.items():
        if start >= len(cells) or end >= len(cells):
            continue
        sc = cell_center(cells, start)
        ec = cell_center(cells, end)
        is_up = end > start
        col = (60, 210, 110) if is_up else (100, 180, 230)
        # dashed glow
        dist = math.sqrt((ec[0] - sc[0]) ** 2 + (ec[1] - sc[1]) ** 2)
        segs = max(6, int(dist / 12))
        for s in range(segs):
            if s % 2 == 0:
                t1, t2 = s / segs, (s + 1) / segs
                p1 = (int(sc[0] + (ec[0] - sc[0]) * t1), int(sc[1] + (ec[1] - sc[1]) * t1))
                p2 = (int(sc[0] + (ec[0] - sc[0]) * t2), int(sc[1] + (ec[1] - sc[1]) * t2))
                cv2.line(panel, p1, p2, col, 2, cv2.LINE_AA)
        # arrow head
        cv2.arrowedLine(panel, sc, ec, col, 1, cv2.LINE_AA, tipLength=0.1)

    # ── draw cells
    t_now = time.time()
    for idx, (x, y, cw, ch) in enumerate(cells):
        g = 4
        x1, y1, x2, y2 = x + g, y + g, x + cw - g, y + ch - g
        effect = SPECIAL_TILES.get(idx)

        # background color
        if idx == 0:
            bg_top = (50, 120, 55)
            bg_bot = (30, 80, 35)
            border_c = (80, 200, 100)
        elif idx == GOAL_POS:
            bg_top = (70, 55, 180)
            bg_bot = (40, 30, 120)
            border_c = (140, 120, 255)
        elif effect:
            base = TILE_COLORS.get(effect, (50, 50, 62))
            bg_top = tuple(min(255, c + 20) for c in base)
            bg_bot = tuple(max(0, c - 10) for c in base)
            border_c = TILE_GLOW.get(effect, (80, 80, 100))
        else:
            bg_top = (52, 52, 66)
            bg_bot = (38, 38, 48)
            border_c = TILE_BORDER

        # cell body (solid top half + solid bottom half = fast pseudo-gradient)
        rounded_rect(panel, (x1, y1), (x2, y2), bg_bot, radius=6)
        h_half = (y2 - y1) // 2
        cv2.rectangle(panel, (x1 + 2, y1 + 2), (x2 - 2, y1 + h_half), bg_top, -1)

        # border glow for special tiles (subtle pulse)
        if effect:
            pulse_a = 0.55 + 0.45 * math.sin(t_now * 3 + idx)
            bc = tuple(int(c * pulse_a) for c in border_c)
            rounded_rect(panel, (x1, y1), (x2, y2), bc, radius=6, thickness=2)
        else:
            rounded_rect(panel, (x1, y1), (x2, y2), border_c, radius=6, thickness=1)

        # cell number
        cv2.putText(panel, str(idx), (x1 + 4, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 195), 1, cv2.LINE_AA)

        # effect label
        if effect:
            lbl = TILE_LABEL.get(effect, "")
            cv2.putText(panel, lbl, (x1 + 4, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
            icon = TILE_ICON.get(effect, "")
            icx = (x1 + x2) // 2
            icy = (y1 + y2) // 2 + 2
            text_centered(panel, icon, icx, icy, cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_c, 1)
        elif idx == 0:
            text_centered(panel, "START", (x1 + x2) // 2, (y1 + y2) // 2 + 2,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 170), 1)
        elif idx == GOAL_POS:
            # sparkling goal
            pulse_b = 0.6 + 0.4 * math.sin(t_now * 4)
            gc = tuple(int(c * pulse_b) for c in (200, 200, 255))
            text_centered(panel, "GOAL", (x1 + x2) // 2, (y1 + y2) // 2 + 2,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, gc, 2)

    # ── tokens
    if phase == PHASE_RESOLVE and anim_t > 0:
        for p in range(2):
            pos = lerp_board(cells, old_positions[p], positions[p], anim_t)
            off = -16 if p == 0 else 16
            draw_token(panel, pos[0] + off, pos[1], p, trail=True)
    else:
        disp = old_positions if (phase == PHASE_RESOLVE and anim_t == 0) else positions
        for p in range(2):
            c = cell_center(cells, disp[p])
            off = -16 if p == 0 else 16
            draw_token(panel, c[0] + off, c[1], p, pulse=(phase == PHASE_COUNTDOWN))

    # ─── HUD Panel (bottom) ─────────────────────────
    iy = bh - 125

    # dark HUD background with border
    rounded_rect(panel, (6, iy - 5), (bw - 6, bh - 4), (22, 22, 30), radius=8)
    rounded_rect(panel, (6, iy - 5), (bw - 6, bh - 4), (50, 50, 65), radius=8, thickness=1)

    # player status cards
    card_w = (bw - 24) // 2
    for p in range(2):
        cx = 10 + p * (card_w + 4)
        col = P1_COLOR if p == 0 else P2_COLOR
        dark = P1_DARK if p == 0 else P2_DARK
        rounded_rect(panel, (cx, iy), (cx + card_w, iy + 22), dark, 4)
        stun_tag = " STUNNED" if stuns[p] > 0 else ""
        cv2.putText(panel, f"P{p+1}  Tile:{positions[p]}/{GOAL_POS}{stun_tag}",
                    (cx + 6, iy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # ── COUNTDOWN HUD
    if phase == PHASE_COUNTDOWN:
        remain = max(0.0, ROUND_SEC - elapsed)
        progress = min(1.0, elapsed / ROUND_SEC)

        # color stages
        if remain > 3:
            bc = (60, 230, 110)
        elif remain > 1.5:
            bc = (50, 230, 230)
        else:
            bc = (50, 80, 240)

        # neon progress bar (fast: solid fill + shine line)
        bar_x, bar_y, bar_w, bar_h = 10, iy + 28, bw - 20, 12
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (35, 35, 42), -1)
        fill_w = max(1, int(bar_w * progress))
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bc, -1)
        # shine line on top
        shine_c = tuple(min(255, c + 60) for c in bc)
        cv2.line(panel, (bar_x, bar_y + 1), (bar_x + fill_w, bar_y + 1), shine_c, 1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (70, 70, 85), 1)

        # big countdown number with glow
        cd = f"{remain:.1f}"
        sz = cv2.getTextSize(cd, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0]
        tx = (bw - sz[0]) // 2
        ty = iy + 72
        # glow behind text
        glow_c = tuple(int(c * 0.3) for c in bc)
        cv2.putText(panel, cd, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.8, glow_c, 6, cv2.LINE_AA)
        cv2.putText(panel, cd, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (8, 8, 12), 4, cv2.LINE_AA)
        cv2.putText(panel, cd, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.8, bc, 3, cv2.LINE_AA)

        # blink alert
        if remain < 1.5 and int(t_now * 7) % 2:
            at = "SHOW YOUR HAND!"
            asz = cv2.getTextSize(at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            atx = (bw - asz[0]) // 2
            cv2.putText(panel, at, (atx, iy + 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_PINK, 2, cv2.LINE_AA)

        # live gesture indicators
        g1t = ansToText.get(live_g1, "---") if live_g1 is not None else "---"
        g2t = ansToText.get(live_g2, "---") if live_g2 is not None else "---"
        # P1 gesture box
        gx1 = 10
        rounded_rect(panel, (gx1, iy + 100), (gx1 + card_w, iy + 120), P1_DARK, 4)
        cv2.putText(panel, f"P1: {g1t}", (gx1 + 8, iy + 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, P1_COLOR, 1, cv2.LINE_AA)
        # P2 gesture box
        gx2 = 14 + card_w
        rounded_rect(panel, (gx2, iy + 100), (gx2 + card_w, iy + 120), P2_DARK, 4)
        cv2.putText(panel, f"P2: {g2t}", (gx2 + 8, iy + 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, P2_COLOR, 1, cv2.LINE_AA)

    # ── RESOLVE HUD
    elif phase == PHASE_RESOLVE:
        # VS banner
        vs_line = f"P1: {rnd_g1_txt}   VS   P2: {rnd_g2_txt}"
        vsz = cv2.getTextSize(vs_line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        vx = (bw - vsz[0]) // 2
        # banner bg
        rounded_rect(panel, (vx - 10, iy + 25), (vx + vsz[0] + 10, iy + 48), (40, 35, 55), 5)
        cv2.putText(panel, vs_line, (vx, iy + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GOLD, 2, cv2.LINE_AA)

        # result logs
        for i, line in enumerate(result_logs[-3:]):
            cv2.putText(panel, line, (14, iy + 66 + i * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 245, 160), 1, cv2.LINE_AA)

        # flash burst
        if elapsed < FLASH_SEC:
            pulse_r = 0.5 + 0.5 * math.sin(t_now * 10)
            flash_c = tuple(int(c * pulse_r) for c in (100, 240, 255))
            text_centered(panel, "RESULT!", bw // 2, iy + 115,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, flash_c, 2)

    # ── GAMEOVER HUD
    elif phase == PHASE_GAMEOVER:
        if game_winner == -1:
            msg = "DRAW!"
            mc = (0, 240, 240)
        else:
            msg = f"P{game_winner + 1} WINS!"
            mc = P1_COLOR if game_winner == 0 else P2_COLOR

        # banner bg
        bsz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        bx = (bw - bsz[0]) // 2
        rounded_rect(panel, (bx - 16, iy + 20), (bx + bsz[0] + 16, iy + 60), (30, 25, 50), 8)

        # pulsing glow text
        pulse_w = 0.6 + 0.4 * math.sin(t_now * 4)
        gc = tuple(int(c * pulse_w) for c in mc)
        text_centered(panel, msg, bw // 2, iy + 45,
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, gc, 3)

        cv2.putText(panel, "Press [R] to restart",
                    (bw // 2 - 70, iy + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1, cv2.LINE_AA)

    # ── history log (very bottom)
    for i, line in enumerate(reversed(history_logs[-2:])):
        cv2.putText(panel, line, (10, bh - 6 - i * 13),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (90, 90, 100), 1, cv2.LINE_AA)

    return panel


# ─── Camera & Window ─────────────────────────────────────
CAM_W, CAM_H = 640, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

BOARD_H = 460
cv2.namedWindow("RPS Racing", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RPS Racing", CAM_W, CAM_H + BOARD_H)

# ─── Threaded Inference ──────────────────────────────────
# 추론은 별도 스레드에서 실행, 결과를 공유 변수로 전달
# 메인 스레드는 카메라 캡처 + 화면 표시만 담당 → 지연 없음

infer_lock = threading.Lock()
infer_frame = None           # 추론할 프레임 (메인→추론 스레드)
infer_result = []            # 최신 detection 결과 (추론→메인 스레드)
infer_busy = False           # 추론 중인지 여부
infer_running = True         # 스레드 종료 플래그


def inference_thread():
    global infer_frame, infer_result, infer_busy
    while infer_running:
        # 새 프레임이 있으면 가져가기
        with infer_lock:
            if infer_frame is None:
                frame_copy = None
            else:
                frame_copy = infer_frame.copy()
                infer_frame = None
                infer_busy = True

        if frame_copy is None:
            time.sleep(0.005)  # CPU 과점 방지
            continue

        # 추론 실행 (블로킹이지만 별도 스레드)
        img_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        img_lb, r, pad_x, pad_y = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))
        img = img_lb.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        if input_dtype == np.int8:
            img = ((img / input_scale) + input_zero).astype(np.int8)
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_index)[0].transpose()
        if output_dtype == np.int8:
            raw = (raw.astype(np.float32) - output_zero) * output_scale

        fh, fw = frame_copy.shape[:2]
        boxes, scores, class_ids = [], [], []
        for det in raw:
            cx, cy, w, h = det[:4]
            cls_scores = det[4:7]
            cls_id = int(np.argmax(cls_scores))
            score = float(cls_scores[cls_id])
            if score < CONF_TH:
                continue
            cx *= IMG_SIZE; cy *= IMG_SIZE
            w *= IMG_SIZE;  h *= IMG_SIZE
            x1 = (cx - w / 2 - pad_x) / r
            y1 = (cy - h / 2 - pad_y) / r
            x2 = (cx + w / 2 - pad_x) / r
            y2 = (cy + h / 2 - pad_y) / r
            boxes.append([
                int(np.clip(x1, 0, fw)),
                int(np.clip(y1, 0, fh)),
                int(np.clip(x2, 0, fw)),
                int(np.clip(y2, 0, fh)),
            ])
            scores.append(score)
            class_ids.append(cls_id)

        keep = nms(boxes, scores, IOU_TH)
        dets = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            cid, sc = class_ids[i], scores[i]
            cx = (x1 + x2) // 2
            dets.append({"box": (x1, y1, x2, y2), "cid": cid, "score": sc, "cx": cx})

        with infer_lock:
            infer_result = dets
            infer_busy = False


# 추론 스레드 시작
t_infer = threading.Thread(target=inference_thread, daemon=True)
t_infer.start()

# ─── Game State ──────────────────────────────────────────
positions = [0, 0]
old_positions = [0, 0]
stuns = [0, 0]
round_idx = 1
phase = PHASE_COUNTDOWN
phase_start = time.time()
game_winner = None

result_logs = []
history_logs = ["Left hand=P1 / Right hand=P2"]
rnd_g1_txt = ""
rnd_g2_txt = ""

p1_vote_buf = []
p2_vote_buf = []
live_g1 = None
live_g2 = None

prev_time = time.time()
cached_detections = []  # 마지막 추론 결과 캐시

# ─── Main Loop ───────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    elapsed = now - phase_start

    # ── 추론 스레드에 최신 프레임 전달 (이전 프레임 덮어쓰기 = 최신만 추론)
    with infer_lock:
        if not infer_busy:
            infer_frame = frame  # 참조만 전달, 스레드에서 copy
        # 추론 결과가 있으면 가져오기
        if infer_result:
            cached_detections = infer_result

    # ── 캐시된 detection으로 화면에 BB 그리기 + 플레이어 판별
    detections = cached_detections
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cid = d["cid"]
        sc = d["score"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[cid], 2)
        cv2.putText(frame, f"{ansToText[cid]} {sc:.2f}",
                    (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_PLAIN, 1.2, colorList[cid], 2)

    p1_g, p2_g, p1_det, p2_det = select_players(detections, frame.shape[1])

    # player labels on camera
    if p1_det:
        bx = p1_det["box"]
        cv2.putText(frame, "P1", (bx[0], bx[1] + 16),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, P1_COLOR, 2)
    if p2_det:
        bx = p2_det["box"]
        cv2.putText(frame, "P2", (bx[0], bx[1] + 16),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, P2_COLOR, 2)

    # ── phase state machine
    anim_t = 0.0

    if phase == PHASE_COUNTDOWN:
        # accumulate votes
        if p1_g is not None:
            p1_vote_buf.append((now, p1_g))
        if p2_g is not None:
            p2_vote_buf.append((now, p2_g))
        live_g1 = vote_gesture(p1_vote_buf, now)
        live_g2 = vote_gesture(p2_vote_buf, now)

        # time's up → resolve
        if elapsed >= ROUND_SEC and game_winner is None:
            final_g1 = vote_gesture(p1_vote_buf, now)
            final_g2 = vote_gesture(p2_vote_buf, now)
            old_positions[:] = positions[:]
            rnd_g1_txt, rnd_g2_txt, rlogs, game_winner = resolve_round(
                final_g1, final_g2, positions, stuns, round_idx
            )
            result_logs = rlogs
            history_logs.extend(rlogs)
            round_idx += 1
            p1_vote_buf.clear()
            p2_vote_buf.clear()

            phase = PHASE_GAMEOVER if game_winner is not None else PHASE_RESOLVE
            phase_start = now

    elif phase == PHASE_RESOLVE:
        if elapsed < FLASH_SEC:
            anim_t = 0.0
        else:
            raw_t = (elapsed - FLASH_SEC) / max(0.01, RESOLVE_SEC - FLASH_SEC)
            anim_t = smoothstep(min(1.0, raw_t))

        if elapsed >= RESOLVE_SEC:
            phase = PHASE_COUNTDOWN
            phase_start = now
            live_g1 = None
            live_g2 = None

    elif phase == PHASE_GAMEOVER:
        pass

    # ── FPS
    fps = 1.0 / max(1e-6, now - prev_time)
    prev_time = now
    cv2.putText(frame, f"FPS:{fps:.0f}", (5, 16),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 255), 1)

    # ── lightweight status bar on camera (no frame.copy)
    bar_h = 28
    fh, fw = frame.shape[:2]
    frame[fh - bar_h:fh, :] = frame[fh - bar_h:fh, :] // 2  # darken in-place
    if phase == PHASE_COUNTDOWN:
        remain = max(0.0, ROUND_SEC - elapsed)
        cv2.putText(frame, f"Round {round_idx}  {remain:.1f}s",
                    (6, fh - 8), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1)
    elif phase == PHASE_RESOLVE:
        cv2.putText(frame, f"P1:{rnd_g1_txt} vs P2:{rnd_g2_txt}",
                    (6, fh - 8), cv2.FONT_HERSHEY_PLAIN, 1.1, GOLD, 1)

    # ── draw board panel
    board = draw_board(
        frame.shape[1], BOARD_H, positions, old_positions, stuns,
        phase, elapsed, round_idx, live_g1, live_g2,
        rnd_g1_txt, rnd_g2_txt, result_logs, history_logs,
        anim_t, game_winner,
    )

    view = cv2.vconcat([frame, board])
    cv2.imshow("RPS Racing", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        positions[:] = [0, 0]
        old_positions[:] = [0, 0]
        stuns[:] = [0, 0]
        round_idx = 1
        phase = PHASE_COUNTDOWN
        phase_start = time.time()
        game_winner = None
        result_logs = []
        history_logs = ["RESET! Left=P1 Right=P2"]
        rnd_g1_txt = ""
        rnd_g2_txt = ""
        p1_vote_buf.clear()
        p2_vote_buf.clear()
        live_g1 = None
        live_g2 = None

infer_running = False
t_infer.join(timeout=2)
cap.release()
cv2.destroyAllWindows()
