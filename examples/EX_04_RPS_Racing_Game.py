import tflite_runtime.interpreter as tflite
import numpy as np
import time
import cv2
import math

# ─── Model Setup ──────────────────────────────────────────
modelPath = "best_float32.tflite"
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

# Colors
P1_COLOR = (80, 230, 120)
P1_DARK = (40, 140, 60)
P2_COLOR = (255, 160, 60)
P2_DARK = (160, 90, 30)
GOLD = (80, 215, 255)

TILE_COLORS = {
    "rocket": (50, 90, 230),
    "broken": (75, 65, 65),
    "ladder_up": (40, 160, 70),
    "mine": (40, 95, 190),
    "warp": (180, 110, 40),
    "ladder_down": (70, 155, 190),
}
TILE_ICON = {
    "rocket": ">>", "broken": "XX", "ladder_up": "^^",
    "mine": "**",   "warp": "<>",   "ladder_down": "vv",
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


# ─── Board Drawing ───────────────────────────────────────

def build_cells(bw, bh, m_top=50, m_bottom=105):
    cols, rows = 5, 4
    mx = 6
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


def draw_token(panel, px, py, player, pulse=False):
    color = P1_COLOR if player == 0 else P2_COLOR
    dark = P1_DARK if player == 0 else P2_DARK
    r = 11
    if pulse:
        r += int(2 * math.sin(time.time() * 5))
    # shadow
    cv2.circle(panel, (px + 2, py + 2), r, (15, 15, 20), -1)
    # main
    cv2.circle(panel, (px, py), r, color, -1)
    # border
    cv2.circle(panel, (px, py), r, dark, 2)
    # shine
    cv2.circle(panel, (px - 3, py - 3), max(2, r // 3),
               tuple(min(255, c + 70) for c in color), -1)
    # label
    lbl = "P1" if player == 0 else "P2"
    cv2.putText(panel, lbl, (px - 8, py + 5),
                cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 2)


def draw_board(bw, bh, positions, old_positions, stuns, phase, elapsed,
               round_idx, live_g1, live_g2, rnd_g1_txt, rnd_g2_txt,
               result_logs, history_logs, anim_t, game_winner):
    panel = np.full((bh, bw, 3), (30, 30, 38), dtype=np.uint8)

    # ── gradient header
    for y in range(42):
        a = 1.0 - y / 42.0
        cv2.line(panel, (0, y), (bw, y), (int(50 * a), int(42 * a), int(58 * a)))

    # title
    cv2.putText(panel, "RPS RACING", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GOLD, 2)
    cv2.putText(panel, f"Round {round_idx}   [q]uit [r]eset",
                (6, 40), cv2.FONT_HERSHEY_PLAIN, 0.95, (170, 170, 180), 1)

    cells = build_cells(bw, bh)

    # ── ladder lines (behind cells)
    for start, end in LADDER_MAP.items():
        if start < len(cells) and end < len(cells):
            sc = cell_center(cells, start)
            ec = cell_center(cells, end)
            col = (55, 190, 100) if end > start else (90, 170, 210)
            cv2.arrowedLine(panel, sc, ec, col, 2, cv2.LINE_AA, tipLength=0.15)

    # ── cells
    for idx, (x, y, cw, ch) in enumerate(cells):
        g = 3
        x1, y1, x2, y2 = x + g, y + g, x + cw - g, y + ch - g
        effect = SPECIAL_TILES.get(idx)

        if idx == 0:
            bg = (45, 100, 45)
        elif idx == GOAL_POS:
            bg = (55, 55, 165)
        elif effect:
            bg = TILE_COLORS.get(effect, (50, 50, 62))
        else:
            bg = (50, 50, 62)

        cv2.rectangle(panel, (x1, y1), (x2, y2), bg, -1)
        # 3D edge highlights
        lt = tuple(min(255, c + 30) for c in bg)
        dk = tuple(max(0, c - 20) for c in bg)
        cv2.line(panel, (x1, y1), (x2, y1), lt, 1)
        cv2.line(panel, (x1, y1), (x1, y2), lt, 1)
        cv2.line(panel, (x1, y2), (x2, y2), dk, 1)
        cv2.line(panel, (x2, y1), (x2, y2), dk, 1)

        cv2.putText(panel, str(idx), (x1 + 2, y1 + 12),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (210, 210, 220), 1)
        if effect:
            cv2.putText(panel, TILE_ICON[effect], (x1 + 2, y2 - 4),
                        cv2.FONT_HERSHEY_PLAIN, 0.85, (255, 255, 255), 1)
        if idx == 0:
            cv2.putText(panel, "START", (x1 + 2, y2 - 4),
                        cv2.FONT_HERSHEY_PLAIN, 0.65, (180, 255, 180), 1)
        elif idx == GOAL_POS:
            cv2.putText(panel, "GOAL", (x1 + 2, y2 - 4),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 200, 255), 1)

    # ── tokens
    if phase == PHASE_RESOLVE and anim_t > 0:
        for p in range(2):
            pos = lerp_board(cells, old_positions[p], positions[p], anim_t)
            off = -14 if p == 0 else 14
            draw_token(panel, pos[0] + off, pos[1], p)
    else:
        disp = old_positions if (phase == PHASE_RESOLVE and anim_t == 0) else positions
        for p in range(2):
            c = cell_center(cells, disp[p])
            off = -14 if p == 0 else 14
            draw_token(panel, c[0] + off, c[1], p, pulse=(phase == PHASE_COUNTDOWN))

    # ─── Info area (bottom portion) ──────────────────
    iy = bh - 100

    # player status line
    for p in range(2):
        col = P1_COLOR if p == 0 else P2_COLOR
        stun_tag = " [STUN]" if stuns[p] > 0 else ""
        bx = 6 + p * (bw // 2)
        cv2.putText(panel, f"P{p+1} pos:{positions[p]}{stun_tag}",
                    (bx, iy), cv2.FONT_HERSHEY_PLAIN, 0.95, col, 1)

    # ── COUNTDOWN phase display
    if phase == PHASE_COUNTDOWN:
        remain = max(0.0, ROUND_SEC - elapsed)
        # color transition
        if remain > 3:
            bc = (60, 220, 100)
        elif remain > 1.5:
            bc = (50, 220, 220)
        else:
            bc = (60, 100, 240)

        # progress bar
        bx2, by2, bw2, bh2 = 6, iy + 8, bw - 12, 10
        progress = min(1.0, elapsed / ROUND_SEC)
        cv2.rectangle(panel, (bx2, by2), (bx2 + bw2, by2 + bh2), (45, 45, 50), -1)
        cv2.rectangle(panel, (bx2, by2), (bx2 + int(bw2 * progress), by2 + bh2), bc, -1)
        cv2.rectangle(panel, (bx2, by2), (bx2 + bw2, by2 + bh2), (100, 100, 110), 1)

        # big countdown number
        cd = f"{remain:.1f}"
        sz = cv2.getTextSize(cd, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)[0]
        tx = (bw - sz[0]) // 2
        ty = iy + 52
        cv2.putText(panel, cd, (tx + 2, ty + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (10, 10, 12), 4)
        cv2.putText(panel, cd, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.6, bc, 3)

        # alert blink
        if remain < 1.5:
            if int(time.time() * 6) % 2:
                at = "SHOW YOUR HAND!"
                asz = cv2.getTextSize(at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.putText(panel, at, ((bw - asz[0]) // 2, iy + 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 2)

        # live gesture display
        g1t = ansToText.get(live_g1, "?") if live_g1 is not None else "?"
        g2t = ansToText.get(live_g2, "?") if live_g2 is not None else "?"
        cv2.putText(panel, f"P1:[{g1t}]", (6, iy + 90), cv2.FONT_HERSHEY_PLAIN, 1.0, P1_COLOR, 1)
        cv2.putText(panel, f"P2:[{g2t}]", (bw // 2, iy + 90), cv2.FONT_HERSHEY_PLAIN, 1.0, P2_COLOR, 1)

    # ── RESOLVE phase display
    elif phase == PHASE_RESOLVE:
        # VS banner
        vs_line = f"P1:{rnd_g1_txt}  VS  P2:{rnd_g2_txt}"
        vsz = cv2.getTextSize(vs_line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        vx = (bw - vsz[0]) // 2
        cv2.putText(panel, vs_line, (vx, iy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GOLD, 2)

        # result logs
        for i, line in enumerate(result_logs[-3:]):
            cv2.putText(panel, line, (8, iy + 40 + i * 15),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 240, 140), 1)

        # flash label
        if elapsed < FLASH_SEC and int(time.time() * 5) % 2:
            rt = "RESULT!"
            rsz = cv2.getTextSize(rt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(panel, rt, ((bw - rsz[0]) // 2, iy + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    # ── GAMEOVER display
    elif phase == PHASE_GAMEOVER:
        if game_winner == -1:
            msg = "DRAW! Both at GOAL!"
        else:
            msg = f"Player {game_winner + 1} WINS!"
        msz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.putText(panel, msg, ((bw - msz[0]) // 2, iy + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(panel, "Press [r] to restart",
                    (bw // 2 - 55, iy + 60), cv2.FONT_HERSHEY_PLAIN, 1.0, (170, 170, 170), 1)

    # ── tiny history at very bottom
    ly = bh - 4
    for line in reversed(history_logs[-2:]):
        cv2.putText(panel, line, (6, ly), cv2.FONT_HERSHEY_PLAIN, 0.75, (120, 120, 130), 1)
        ly -= 12

    return panel


# ─── Camera & Window ─────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

BOARD_H = 380
cv2.namedWindow("RPS Racing", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RPS Racing", 320, 240 + BOARD_H)

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

# ─── Main Loop ───────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    elapsed = now - phase_start

    # ── detection (always running)
    detections = infer_gestures(frame)
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

    # ── countdown bar on camera feed
    if phase == PHASE_COUNTDOWN:
        remain = max(0.0, ROUND_SEC - elapsed)
        ov = frame.copy()
        cv2.rectangle(ov, (0, frame.shape[0] - 28), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"Round {round_idx}  {remain:.1f}s",
                    (6, frame.shape[0] - 8), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1)
    elif phase == PHASE_RESOLVE:
        ov = frame.copy()
        cv2.rectangle(ov, (0, frame.shape[0] - 28), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"P1:{rnd_g1_txt} vs P2:{rnd_g2_txt}",
                    (6, frame.shape[0] - 8), cv2.FONT_HERSHEY_PLAIN, 1.1, GOLD, 1)

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

cap.release()
cv2.destroyAllWindows()
