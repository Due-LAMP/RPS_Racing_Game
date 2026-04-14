import numpy as np
import time
import cv2
import math
import multiprocessing as mp

# ═══════════════════════════════════════════════════════════
#  Architecture: Main process (camera + display + game)
#              + Child process (YOLO inference)
#  Communication: mp.Queue (bypasses Python GIL)
# ═══════════════════════════════════════════════════════════

# ─── Inference Worker (runs in separate process) ─────────

def inference_worker(frame_q, result_q, model_path):
    """Separate process with its own GIL => true parallelism."""
    import tflite_runtime.interpreter as tflite

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    input_index  = inp["index"]
    output_index = out["index"]
    input_dtype  = inp["dtype"]
    output_dtype = out["dtype"]
    input_scale, input_zero   = inp["quantization"]
    output_scale, output_zero = out["quantization"]

    IMG_SIZE = 320
    CONF_TH  = 0.5
    IOU_TH   = 0.45

    def letterbox(img):
        h, w = img.shape[:2]
        r = min(IMG_SIZE / w, IMG_SIZE / h)
        nw, nh = int(w * r), int(h * r)
        resized = cv2.resize(img, (nw, nh))
        pw, ph = IMG_SIZE - nw, IMG_SIZE - nh
        px, py = pw // 2, ph // 2
        padded = cv2.copyMakeBorder(
            resized, py, ph - py, px, pw - px,
            cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, r, px, py

    def nms(boxes, scores):
        if not boxes:
            return []
        b = np.array(boxes); s = np.array(scores)
        x1, y1, x2, y2 = b.T
        areas = (x2 - x1) * (y2 - y1)
        order = s.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)
            order = order[np.where(iou <= IOU_TH)[0] + 1]
        return keep

    while True:
        # drain queue — only keep the LATEST frame
        frame_data = None
        try:
            while True:
                frame_data = frame_q.get_nowait()
        except Exception:
            pass
        if frame_data is None:
            try:
                frame_data = frame_q.get(timeout=0.05)
            except Exception:
                continue
        if frame_data is None:
            continue

        frame = frame_data
        fh, fw = frame.shape[:2]

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_lb, r, pad_x, pad_y = letterbox(img_rgb)
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
            cls = det[4:7]
            cid = int(np.argmax(cls))
            sc = float(cls[cid])
            if sc < CONF_TH:
                continue
            cx *= IMG_SIZE; cy *= IMG_SIZE
            w *= IMG_SIZE;  h *= IMG_SIZE
            x1 = (cx - w/2 - pad_x) / r
            y1 = (cy - h/2 - pad_y) / r
            x2 = (cx + w/2 - pad_x) / r
            y2 = (cy + h/2 - pad_y) / r
            boxes.append([int(np.clip(x1,0,fw)), int(np.clip(y1,0,fh)),
                          int(np.clip(x2,0,fw)), int(np.clip(y2,0,fh))])
            scores.append(sc)
            class_ids.append(cid)

        keep = nms(boxes, scores)
        dets = [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                 class_ids[i], scores[i]) for i in keep]

        # push latest result (drain old ones)
        try:
            while not result_q.empty():
                result_q.get_nowait()
        except Exception:
            pass
        result_q.put(dets)


# ─── Constants ───────────────────────────────────────────

ansToText  = {0: "scissors", 1: "rock", 2: "paper"}
colorList  = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

BOARD_SIZE  = 20
GOAL_POS    = BOARD_SIZE - 1
ROUND_SEC   = 5.0
RESOLVE_SEC = 2.8
FLASH_SEC   = 0.8
VOTE_WINDOW = 1.0

SPECIAL_TILES = {
    3: "rocket", 6: "broken", 9: "ladder_up",
    12: "mine", 15: "warp", 17: "ladder_down",
}
LADDER_MAP = {9: 14, 17: 11}

PHASE_COUNTDOWN = 0
PHASE_RESOLVE   = 1
PHASE_GAMEOVER  = 2

P1_COLOR = (100, 255, 140)
P1_MID   = (60, 200, 100)
P1_DARK  = (30, 120, 50)
P1_GLOW  = (80, 255, 120)
P2_COLOR = (255, 180, 80)
P2_MID   = (220, 130, 50)
P2_DARK  = (150, 80, 25)
P2_GLOW  = (255, 160, 60)
GOLD     = (80, 220, 255)
NEON_CYAN  = (240, 220, 60)
NEON_PINK  = (180, 80, 255)
BG_DARK    = (18, 18, 24)
TILE_BORDER = (60, 60, 75)

TILE_COLORS = {
    "rocket": (40,70,200), "broken": (55,50,50),
    "ladder_up": (30,140,55), "mine": (30,80,170),
    "warp": (160,90,30), "ladder_down": (55,130,165),
}
TILE_GLOW = {
    "rocket": (80,130,255), "broken": (100,80,80),
    "ladder_up": (80,230,110), "mine": (60,130,240),
    "warp": (240,170,70), "ladder_down": (110,200,240),
}
TILE_ICON = {
    "rocket": ">>", "broken": "XX", "ladder_up": "^^",
    "mine": "**", "warp": "<>", "ladder_down": "vv",
}
TILE_LABEL = {
    "rocket": "ROCKET", "broken": "BROKEN", "ladder_up": "LADDER",
    "mine": "MINE", "warp": "WARP", "ladder_down": "SLIDE",
}


# ─── Player Selection ────────────────────────────────────

def select_players(detections, frame_w):
    p1 = p2 = p1_det = p2_det = None
    if not detections:
        return p1, p2, p1_det, p2_det
    if len(detections) == 1:
        d = detections[0]
        cx = (d[0] + d[2]) // 2
        if cx < frame_w // 2:
            p1, p1_det = d[4], d
        else:
            p2, p2_det = d[4], d
        return p1, p2, p1_det, p2_det
    left  = min(detections, key=lambda d: (d[0]+d[2])//2)
    right = max(detections, key=lambda d: (d[0]+d[2])//2)
    return left[4], right[4], left, right


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
        logs.append("Both stunned!")
    elif can_play[0] and not can_play[1]:
        delta[0] += 2; logs.append("P2 stunned -> P1 +2")
    elif not can_play[0] and can_play[1]:
        delta[1] += 2; logs.append("P1 stunned -> P2 +2")
    elif g1 is None and g2 is None:
        logs.append("No hands detected!")
    elif g1 is None:
        delta[1] += 1; logs.append("P1 missing -> P2 +1")
    elif g2 is None:
        delta[0] += 1; logs.append("P2 missing -> P1 +1")
    elif g1 == g2:
        if g1 == 1:
            delta[0] -= 1; delta[1] -= 1; logs.append("TIE Rock -> both -1")
        elif g1 == 0:
            positions[0], positions[1] = positions[1], positions[0]
            logs.append("TIE Scissors -> SWAP!")
        else:
            delta[0] += 1; delta[1] += 1; logs.append("TIE Paper -> both +1")
    else:
        p1w = beats[g1] == g2
        w = 0 if p1w else 1; l = 1 - w
        wg = g1 if p1w else g2
        if wg == 1:
            delta[l] -= 3; logs.append(f"Rock WIN! P{l+1} -3")
        elif wg == 0:
            delta[w] += 2; logs.append(f"Scissors WIN! P{w+1} +2")
        else:
            delta[w] += 1; delta[l] -= 1
            logs.append(f"Paper WIN! P{w+1}+1 P{l+1}-1")

    for i in range(2):
        positions[i] = clamp_pos(positions[i] + delta[i])
    apply_specials(positions, stuns, logs)

    gw = None
    if positions[0] >= GOAL_POS and positions[1] >= GOAL_POS: gw = -1
    elif positions[0] >= GOAL_POS: gw = 0
    elif positions[1] >= GOAL_POS: gw = 1
    return g1_txt, g2_txt, logs, gw


def apply_specials(positions, stuns, logs):
    for p in range(2):
        for _ in range(3):
            t = positions[p]
            e = SPECIAL_TILES.get(t)
            if e is None: break
            if e == "rocket":
                positions[p] = clamp_pos(positions[p] + 4)
                logs.append(f"P{p+1} ROCKET! +4")
            elif e == "broken":
                stuns[p] = max(stuns[p], 1)
                logs.append(f"P{p+1} BROKEN! skip next"); break
            elif e in ("ladder_up", "ladder_down"):
                dest = LADDER_MAP[t]
                positions[p] = dest
                logs.append(f"P{p+1} {'UP' if e=='ladder_up' else 'DOWN'} {t}->{dest}")
            elif e == "mine":
                positions[p] = clamp_pos(positions[p] - 2)
                logs.append(f"P{p+1} MINE! -2"); break
            elif e == "warp":
                o = 1 - p
                positions[p], positions[o] = positions[o], positions[p]
                logs.append("WARP! P1<->P2"); break


# ─── Drawing Helpers ─────────────────────────────────────

def rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    x1, y1 = pt1; x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness == -1:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        cv2.circle(img, (x1+r, y1+r), r, color, -1)
        cv2.circle(img, (x2-r, y1+r), r, color, -1)
        cv2.circle(img, (x1+r, y2-r), r, color, -1)
        cv2.circle(img, (x2-r, y2-r), r, color, -1)
    else:
        cv2.line(img, (x1+r,y1), (x2-r,y1), color, thickness)
        cv2.line(img, (x1+r,y2), (x2-r,y2), color, thickness)
        cv2.line(img, (x1,y1+r), (x1,y2-r), color, thickness)
        cv2.line(img, (x2,y1+r), (x2,y2-r), color, thickness)
        cv2.ellipse(img, (x1+r,y1+r), (r,r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r,y1+r), (r,r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+r,y2-r), (r,r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r,y2-r), (r,r), 0, 0, 90, color, thickness)


def put_text_s(img, text, pos, font, scale, color, thick=2):
    cv2.putText(img, text, (pos[0]+1,pos[1]+1), font, scale, (10,10,15), thick+1)
    cv2.putText(img, text, pos, font, scale, color, thick)


def text_c(img, text, cx, cy, font, scale, color, thick=2):
    sz = cv2.getTextSize(text, font, scale, thick)[0]
    put_text_s(img, text, (cx-sz[0]//2, cy+sz[1]//2), font, scale, color, thick)


# ─── Board Drawing ───────────────────────────────────────

def build_cells(bw, bh, m_top=56, m_bot=130):
    cols, rows = 5, 4
    mx = 12
    cw = (bw - 2*mx) // cols
    ch = (bh - m_top - m_bot) // rows
    cells = []
    for r in range(rows):
        y = bh - m_bot - (r+1)*ch
        order = range(cols) if r % 2 == 0 else range(cols-1,-1,-1)
        for c in order:
            cells.append((mx + c*cw, y, cw, ch))
    return cells[:BOARD_SIZE]


def cell_center(cells, idx):
    x, y, w, h = cells[idx]
    return (x + w//2, y + h//2)


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2*t)


def lerp_pos(cells, old_p, new_p, t):
    if old_p == new_p:
        return cell_center(cells, new_p)
    f = old_p + (new_p - old_p) * t
    f = max(0.0, min(float(GOAL_POS), f))
    lo = int(f); hi = min(lo+1, GOAL_POS); fr = f - lo
    c1 = cell_center(cells, lo); c2 = cell_center(cells, hi)
    return (int(c1[0]+(c2[0]-c1[0])*fr), int(c1[1]+(c2[1]-c1[1])*fr))


def draw_token(panel, px, py, player, pulse=False):
    color = P1_COLOR if player == 0 else P2_COLOR
    mid   = P1_MID   if player == 0 else P2_MID
    dark  = P1_DARK  if player == 0 else P2_DARK
    glow  = P1_GLOW  if player == 0 else P2_GLOW
    r = 14
    if pulse:
        r += int(3 * math.sin(time.time() * 6))
    cv2.circle(panel, (px, py), r+6, tuple(max(0, c//4) for c in glow), 2)
    cv2.circle(panel, (px, py), r+3, tuple(max(0, c//2) for c in glow), 2)
    cv2.circle(panel, (px+2, py+3), r, (8,8,12), -1)
    cv2.circle(panel, (px, py), r, dark, -1)
    cv2.circle(panel, (px, py), r-3, mid, -1)
    cv2.circle(panel, (px-1, py-1), r-5, color, -1)
    cv2.circle(panel, (px-r//3, py-r//3), max(2, r//4), (255,255,255), -1)
    cv2.circle(panel, (px, py), r, dark, 2)
    lbl = "P1" if player == 0 else "P2"
    sz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)[0]
    cv2.putText(panel, lbl, (px-sz[0]//2, py+sz[1]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 2)


# ─── Board State (cached for performance) ───────────────

_board_cache = None        # immutable static board image
_board_cache_size = None
_cells_cache = None        # cached cell positions
_dirty_rects = []          # (y1, y2, x1, x2) regions to restore from cache


def _build_board_cache(bw, bh):
    global _board_cache, _board_cache_size, _cells_cache
    if _board_cache is not None and _board_cache_size == (bw, bh):
        return
    panel = np.full((bh, bw, 3), BG_DARK, dtype=np.uint8)
    h_hdr = 50
    t = np.linspace(0, 1, h_hdr, dtype=np.float32).reshape(h_hdr, 1, 1)
    top = np.array((40, 30, 65), dtype=np.float32)
    bot = np.array(BG_DARK, dtype=np.float32)
    grad = (top + (bot - top) * t).astype(np.uint8)
    panel[:h_hdr, :] = np.broadcast_to(grad, (h_hdr, bw, 3))
    cv2.line(panel, (10, 50), (bw-10, 50), NEON_CYAN, 1)

    _cells_cache = build_cells(bw, bh)

    for start, end in LADDER_MAP.items():
        if start >= len(_cells_cache) or end >= len(_cells_cache): continue
        sc = cell_center(_cells_cache, start); ec = cell_center(_cells_cache, end)
        col = (60,210,110) if end > start else (100,180,230)
        cv2.arrowedLine(panel, sc, ec, col, 2, tipLength=0.1)

    for idx, (x, y, cw, ch) in enumerate(_cells_cache):
        g = 4
        x1, y1, x2, y2 = x+g, y+g, x+cw-g, y+ch-g
        effect = SPECIAL_TILES.get(idx)
        if idx == 0:
            bg_top, bg_bot, border_c = (50,120,55), (30,80,35), (80,200,100)
        elif idx == GOAL_POS:
            bg_top, bg_bot, border_c = (70,55,180), (40,30,120), (140,120,255)
        elif effect:
            base = TILE_COLORS.get(effect, (50,50,62))
            bg_top = tuple(min(255, c+20) for c in base)
            bg_bot = tuple(max(0, c-10) for c in base)
            border_c = TILE_GLOW.get(effect, (80,80,100))
        else:
            bg_top, bg_bot, border_c = (52,52,66), (38,38,48), TILE_BORDER
        rounded_rect(panel, (x1,y1), (x2,y2), bg_bot, radius=6)
        cv2.rectangle(panel, (x1+2,y1+2), (x2-2, y1+(y2-y1)//2), bg_top, -1)
        th = 2 if effect else 1
        rounded_rect(panel, (x1,y1), (x2,y2), border_c, radius=6, thickness=th)
        cv2.putText(panel, str(idx), (x1+4, y1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,195), 1)
        if effect:
            cv2.putText(panel, TILE_LABEL.get(effect,""), (x1+4, y2-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1)
            text_c(panel, TILE_ICON.get(effect,""), (x1+x2)//2, (y1+y2)//2+2,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_c, 1)
        elif idx == 0:
            text_c(panel, "START", (x1+x2)//2, (y1+y2)//2+2,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,255,170), 1)
        elif idx == GOAL_POS:
            text_c(panel, "GOAL", (x1+x2)//2, (y1+y2)//2+2,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,255), 2)

    _board_cache = panel
    _board_cache_size = (bw, bh)


def _restore_dirty(panel, bh, bw):
    """Restore previously dirtied regions from static cache."""
    global _dirty_rects
    for (ry1, ry2, rx1, rx2) in _dirty_rects:
        panel[ry1:ry2, rx1:rx2] = _board_cache[ry1:ry2, rx1:rx2]
    _dirty_rects = []


def _mark_dirty(y1, y2, x1, x2, bh, bw):
    _dirty_rects.append((max(0, y1), min(bh, y2), max(0, x1), min(bw, x2)))


def draw_board(panel, bw, bh, positions, old_positions, stuns, phase, elapsed,
               round_idx, live_g1, live_g2, rnd_g1_txt, rnd_g2_txt,
               result_logs, history_logs, anim_t, game_winner):
    """Draw dynamic elements onto persistent board panel (selective restore)."""

    _build_board_cache(bw, bh)
    cells = _cells_cache

    # restore only regions dirtied in previous frame
    _restore_dirty(panel, bh, bw)

    # ── title (top 56px)
    _mark_dirty(0, 56, 0, bw, bh, bw)
    panel[:56] = _board_cache[:56]
    put_text_s(panel, "RPS RACING", (12,24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GOLD, 2)
    put_text_s(panel, f"Round {round_idx}", (12,44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,210), 1)
    cv2.putText(panel, "[Q]uit [R]eset", (bw-120,44), cv2.FONT_HERSHEY_PLAIN, 1.0, (130,130,145), 1)

    # ── tokens (small patches around each token)
    TR = 24  # token radius + glow margin
    if phase == PHASE_RESOLVE and anim_t > 0:
        for p in range(2):
            pos = lerp_pos(cells, old_positions[p], positions[p], anim_t)
            off = -16 if p == 0 else 16
            tx, ty = pos[0]+off, pos[1]
            _mark_dirty(ty-TR, ty+TR, tx-TR, tx+TR, bh, bw)
            draw_token(panel, tx, ty, p)
    else:
        disp = old_positions if (phase == PHASE_RESOLVE and anim_t == 0) else positions
        for p in range(2):
            c = cell_center(cells, disp[p])
            off = -16 if p == 0 else 16
            tx, ty = c[0]+off, c[1]
            _mark_dirty(ty-TR, ty+TR, tx-TR, tx+TR, bh, bw)
            draw_token(panel, tx, ty, p, pulse=(phase == PHASE_COUNTDOWN))

    # ── HUD area (bottom 130px)
    iy = bh - 125
    _mark_dirty(iy-5, bh, 0, bw, bh, bw)
    panel[max(0,iy-5):bh] = _board_cache[max(0,iy-5):bh]

    rounded_rect(panel, (6, iy-5), (bw-6, bh-4), (22,22,30), radius=8)
    rounded_rect(panel, (6, iy-5), (bw-6, bh-4), (50,50,65), radius=8, thickness=1)

    card_w = (bw - 24) // 2
    for p in range(2):
        cx = 10 + p*(card_w+4)
        col = P1_COLOR if p == 0 else P2_COLOR
        dk = P1_DARK if p == 0 else P2_DARK
        rounded_rect(panel, (cx, iy), (cx+card_w, iy+22), dk, 4)
        stun_tag = " STUNNED" if stuns[p] > 0 else ""
        cv2.putText(panel, f"P{p+1} Tile:{positions[p]}/{GOAL_POS}{stun_tag}",
                    (cx+6, iy+16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    t_now = time.time()

    if phase == PHASE_COUNTDOWN:
        remain = max(0.0, ROUND_SEC - elapsed)
        progress = min(1.0, elapsed / ROUND_SEC)
        bc = (60,230,110) if remain > 3 else (50,230,230) if remain > 1.5 else (50,80,240)

        bx, by, bww, bhh = 10, iy+28, bw-20, 12
        cv2.rectangle(panel, (bx,by), (bx+bww, by+bhh), (35,35,42), -1)
        fw = max(1, int(bww * progress))
        cv2.rectangle(panel, (bx,by), (bx+fw, by+bhh), bc, -1)
        cv2.line(panel, (bx,by+1), (bx+fw,by+1), tuple(min(255,c+60) for c in bc), 1)
        cv2.rectangle(panel, (bx,by), (bx+bww, by+bhh), (70,70,85), 1)

        cd = f"{remain:.1f}"
        sz = cv2.getTextSize(cd, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0]
        tx = (bw-sz[0])//2; ty = iy+72
        cv2.putText(panel, cd, (tx+1,ty+1), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (8,8,12), 4)
        cv2.putText(panel, cd, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 1.8, bc, 3)

        if remain < 1.5 and int(t_now*7) % 2:
            at = "SHOW YOUR HAND!"
            asz = cv2.getTextSize(at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            cv2.putText(panel, at, ((bw-asz[0])//2, iy+92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_PINK, 2)

        g1t = ansToText.get(live_g1, "---") if live_g1 is not None else "---"
        g2t = ansToText.get(live_g2, "---") if live_g2 is not None else "---"
        gx1 = 10; gx2 = 14 + card_w
        rounded_rect(panel, (gx1, iy+100), (gx1+card_w, iy+120), P1_DARK, 4)
        cv2.putText(panel, f"P1: {g1t}", (gx1+8, iy+116), cv2.FONT_HERSHEY_SIMPLEX, 0.4, P1_COLOR, 1)
        rounded_rect(panel, (gx2, iy+100), (gx2+card_w, iy+120), P2_DARK, 4)
        cv2.putText(panel, f"P2: {g2t}", (gx2+8, iy+116), cv2.FONT_HERSHEY_SIMPLEX, 0.4, P2_COLOR, 1)

    elif phase == PHASE_RESOLVE:
        vs = f"P1:{rnd_g1_txt}  VS  P2:{rnd_g2_txt}"
        vsz = cv2.getTextSize(vs, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        vx = (bw-vsz[0])//2
        rounded_rect(panel, (vx-10,iy+25), (vx+vsz[0]+10,iy+48), (40,35,55), 5)
        cv2.putText(panel, vs, (vx,iy+44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GOLD, 2)
        for i, line in enumerate(result_logs[-3:]):
            cv2.putText(panel, line, (14,iy+66+i*17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,245,160), 1)
        if elapsed < FLASH_SEC and int(t_now*5) % 2:
            text_c(panel, "RESULT!", bw//2, iy+115, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,240,255), 2)

    elif phase == PHASE_GAMEOVER:
        if game_winner == -1:
            msg, mc = "DRAW!", (0,240,240)
        else:
            msg = f"P{game_winner+1} WINS!"
            mc = P1_COLOR if game_winner == 0 else P2_COLOR
        bsz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        bx = (bw-bsz[0])//2
        rounded_rect(panel, (bx-16,iy+20), (bx+bsz[0]+16,iy+60), (30,25,50), 8)
        pw = 0.6 + 0.4 * math.sin(t_now * 4)
        text_c(panel, msg, bw//2, iy+45, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
               tuple(int(c*pw) for c in mc), 3)
        cv2.putText(panel, "Press [R] to restart", (bw//2-70, iy+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,160), 1)

    for i, line in enumerate(reversed(history_logs[-2:])):
        cv2.putText(panel, line, (10, bh-6-i*13), cv2.FONT_HERSHEY_PLAIN, 0.8, (90,90,100), 1)


# ─── Main ────────────────────────────────────────────────

def main():
    global _board_cache, _dirty_rects
    modelPath = "./models/best_float32.tflite"
    # modelPath = "./models/best_float16.tflite"  
    # modelPath = "./models/best_full_integer_quant.tflite"  # fastest on Pi CPU

    frame_q  = mp.Queue(maxsize=2)
    result_q = mp.Queue(maxsize=2)
    proc = mp.Process(target=inference_worker,
                      args=(frame_q, result_q, modelPath), daemon=True)
    proc.start()

    # Camera: 320x240 for fast capture, resize to 640x480 for display
    CAM_W, CAM_H = 320, 240
    DISP_W, DISP_H = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    BOARD_H = 460
    cv2.namedWindow("RPS Racing", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RPS Racing", DISP_W, DISP_H + BOARD_H)

    # ── Pre-allocated buffers (no per-frame allocation)
    view = np.empty((DISP_H + BOARD_H, DISP_W, 3), dtype=np.uint8)
    board_panel = None  # persistent working panel (initialized after cache built)

    positions     = [0, 0]
    old_positions = [0, 0]
    stuns = [0, 0]
    round_idx = 1
    phase = PHASE_COUNTDOWN
    phase_start = time.time()
    game_winner = None
    result_logs = []
    history_logs = ["Left=P1 / Right=P2"]
    rnd_g1_txt = rnd_g2_txt = ""
    p1_vote_buf = []
    p2_vote_buf = []
    live_g1 = live_g2 = None
    prev_time = time.time()
    cached_dets = []

    sx = DISP_W / CAM_W
    sy = DISP_H / CAM_H

    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret:
            break

        now = time.time()
        elapsed = now - phase_start

        # send frame to inference process (non-blocking)
        try:
            frame_q.put_nowait(frame_raw)
        except Exception:
            pass

        # get latest detections (non-blocking)
        try:
            while not result_q.empty():
                cached_dets = result_q.get_nowait()
        except Exception:
            pass

        # resize frame for display
        frame = cv2.resize(frame_raw, (DISP_W, DISP_H))

        # draw detections (scale coords to display)
        scaled_dets = []
        for d in cached_dets:
            x1, y1, x2, y2, cid, sc = d
            dx1, dy1 = int(x1*sx), int(y1*sy)
            dx2, dy2 = int(x2*sx), int(y2*sy)
            cv2.rectangle(frame, (dx1,dy1), (dx2,dy2), colorList[cid], 2)
            cv2.putText(frame, f"{ansToText[cid]} {sc:.2f}",
                        (dx1, max(15, dy1-8)),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, colorList[cid], 2)
            scaled_dets.append((dx1, dy1, dx2, dy2, cid, sc))

        p1_g, p2_g, p1_det, p2_det = select_players(scaled_dets, DISP_W)

        if p1_det:
            cv2.putText(frame, "P1", (p1_det[0], p1_det[1]+16),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, P1_COLOR, 2)
        if p2_det:
            cv2.putText(frame, "P2", (p2_det[0], p2_det[1]+16),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, P2_COLOR, 2)

        anim_t = 0.0

        if phase == PHASE_COUNTDOWN:
            if p1_g is not None: p1_vote_buf.append((now, p1_g))
            if p2_g is not None: p2_vote_buf.append((now, p2_g))
            live_g1 = vote_gesture(p1_vote_buf, now)
            live_g2 = vote_gesture(p2_vote_buf, now)

            if elapsed >= ROUND_SEC and game_winner is None:
                final_g1 = vote_gesture(p1_vote_buf, now)
                final_g2 = vote_gesture(p2_vote_buf, now)
                old_positions[:] = positions[:]
                rnd_g1_txt, rnd_g2_txt, rlogs, game_winner = resolve_round(
                    final_g1, final_g2, positions, stuns, round_idx)
                result_logs = rlogs
                history_logs.extend(rlogs)
                round_idx += 1
                p1_vote_buf.clear(); p2_vote_buf.clear()
                phase = PHASE_GAMEOVER if game_winner is not None else PHASE_RESOLVE
                phase_start = now

        elif phase == PHASE_RESOLVE:
            if elapsed < FLASH_SEC:
                anim_t = 0.0
            else:
                anim_t = smoothstep(
                    min(1.0, (elapsed - FLASH_SEC) / max(0.01, RESOLVE_SEC - FLASH_SEC)))
            if elapsed >= RESOLVE_SEC:
                phase = PHASE_COUNTDOWN
                phase_start = now
                live_g1 = live_g2 = None

        fps = 1.0 / max(1e-6, now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS:{fps:.0f}", (5, 22),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2)

        fh, fw = frame.shape[:2]
        frame[fh-30:fh, :] = frame[fh-30:fh, :] // 2
        if phase == PHASE_COUNTDOWN:
            remain = max(0.0, ROUND_SEC - elapsed)
            cv2.putText(frame, f"Round {round_idx}  {remain:.1f}s",
                        (8, fh-8), cv2.FONT_HERSHEY_PLAIN, 1.3, (255,255,255), 1)
        elif phase == PHASE_RESOLVE:
            cv2.putText(frame, f"P1:{rnd_g1_txt} vs P2:{rnd_g2_txt}",
                        (8, fh-8), cv2.FONT_HERSHEY_PLAIN, 1.3, GOLD, 1)

        # ── Board: draw onto persistent panel (selective restore)
        if board_panel is None:
            _build_board_cache(DISP_W, BOARD_H)
            board_panel = _board_cache.copy()
        draw_board(board_panel, DISP_W, BOARD_H, positions, old_positions, stuns,
                   phase, elapsed, round_idx, live_g1, live_g2,
                   rnd_g1_txt, rnd_g2_txt, result_logs, history_logs,
                   anim_t, game_winner)

        # ── Compose into pre-allocated view (no vconcat allocation)
        view[:DISP_H] = frame
        view[DISP_H:] = board_panel
        cv2.imshow("RPS Racing", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            positions[:] = [0,0]; old_positions[:] = [0,0]; stuns[:] = [0,0]
            round_idx = 1; phase = PHASE_COUNTDOWN; phase_start = time.time()
            game_winner = None; result_logs = []
            history_logs = ["RESET! Left=P1 Right=P2"]
            rnd_g1_txt = rnd_g2_txt = ""
            p1_vote_buf.clear(); p2_vote_buf.clear()
            live_g1 = live_g2 = None
            _board_cache = None
            _dirty_rects = []
            board_panel = None

    proc.terminate()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()