import tflite_runtime.interpreter as tflite
import numpy as np
import time
import cv2

modelPath = "best_float32.tflite"
# modelPath = "best_float16.tflite"
# modelPath = "best_full_integer_quant.tflite"
# modelPath = "best_int8.tflite"
# modelPath = "best_integer_quant.tflite"
print("model path:", modelPath)

interpreter = tflite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]
output_dtype = output_details[0]["dtype"]
input_scale, input_zero = input_details[0]["quantization"]
output_scale, output_zero = output_details[0]["quantization"]
height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]
print("model input shape:", (height, width))

ansToText = {0: "scissors", 1: "rock", 2: "paper"}
colorList = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

IMG_SIZE = 320
CONF_TH = 0.5
IOU_TH = 0.45

BOARD_SIZE = 20
GOAL_POS = BOARD_SIZE - 1
TURN_HOLD_SEC = 1.0
ROUND_COOLDOWN_SEC = 0.8

SPECIAL_TILES = {
    4: "rocket",
    7: "broken",
    10: "ladder_up",
    13: "mine",
    16: "warp",
    18: "ladder_down",
}
LADDER_MAP = {10: 15, 18: 12}


def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)

    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = nw - new_w
    pad_h = nh - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        pad_h - pad_y,
        pad_x,
        pad_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=color,
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


def clamp_pos(pos):
    return int(np.clip(pos, 0, GOAL_POS))


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

        cx *= IMG_SIZE
        cy *= IMG_SIZE
        w *= IMG_SIZE
        h *= IMG_SIZE

        x1 = (cx - w / 2 - pad_x) / r
        y1 = (cy - h / 2 - pad_y) / r
        x2 = (cx + w / 2 - pad_x) / r
        y2 = (cy + h / 2 - pad_y) / r

        boxes.append(
            [
                int(np.clip(x1, 0, frame.shape[1])),
                int(np.clip(y1, 0, frame.shape[0])),
                int(np.clip(x2, 0, frame.shape[1])),
                int(np.clip(y2, 0, frame.shape[0])),
            ]
        )
        scores.append(score)
        class_ids.append(cls_id)

    keep = nms(boxes, scores, IOU_TH)

    detections = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cid = class_ids[i]
        sc = scores[i]
        cx = (x1 + x2) // 2
        detections.append({"box": (x1, y1, x2, y2), "cid": cid, "score": sc, "cx": cx})

        cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[cid], 2)
        cv2.putText(
            frame,
            f"{ansToText[cid]} {sc:.2f}",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_PLAIN,
            1.2,
            colorList[cid],
            2,
        )

    return detections


def select_players(detections, frame_w):
    p1 = None
    p2 = None
    p1_det = None
    p2_det = None

    if not detections:
        return p1, p2, p1_det, p2_det

    if len(detections) == 1:
        only = detections[0]
        if only["cx"] < frame_w // 2:
            p1 = only["cid"]
            p1_det = only
        else:
            p2 = only["cid"]
            p2_det = only
        return p1, p2, p1_det, p2_det

    left = min(detections, key=lambda d: d["cx"])
    right = max(detections, key=lambda d: d["cx"])
    p1 = left["cid"]
    p2 = right["cid"]
    p1_det = left
    p2_det = right
    return p1, p2, p1_det, p2_det


def resolve_round(g1, g2, positions, stuns, round_idx):
    logs = [f"Round {round_idx}"]
    can_play = [True, True]
    for i in range(2):
        if stuns[i] > 0:
            can_play[i] = False
            stuns[i] -= 1

    delta = [0, 0]
    beats = {1: 0, 0: 2, 2: 1}

    if not can_play[0] and not can_play[1]:
        logs.append("Both players are broken this turn.")
    elif can_play[0] and not can_play[1]:
        delta[0] += 2
        logs.append("P2 skip. P1 +2")
    elif not can_play[0] and can_play[1]:
        delta[1] += 2
        logs.append("P1 skip. P2 +2")
    else:
        if g1 == g2:
            if g1 == 1:
                delta[0] -= 1
                delta[1] -= 1
                logs.append("Tie: rock -> both -1")
            elif g1 == 0:
                positions[0], positions[1] = positions[1], positions[0]
                logs.append("Tie: scissors -> swap")
            else:
                delta[0] += 1
                delta[1] += 1
                logs.append("Tie: paper -> both +1")
        else:
            p1_win = beats[g1] == g2
            winner = 0 if p1_win else 1
            loser = 1 - winner
            win_gesture = g1 if p1_win else g2

            if win_gesture == 1:
                delta[loser] -= 3
                logs.append(f"Rock win: P{loser + 1} -3")
            elif win_gesture == 0:
                delta[winner] += 2
                logs.append(f"Scissors win: P{winner + 1} +2")
            else:
                delta[winner] += 1
                delta[loser] -= 1
                logs.append(f"Paper win: P{winner + 1} +1 / P{loser + 1} -1")

    for i in range(2):
        positions[i] = clamp_pos(positions[i] + delta[i])

    apply_specials(positions, stuns, logs)

    winner = None
    if positions[0] >= GOAL_POS and positions[1] >= GOAL_POS:
        winner = -1
    elif positions[0] >= GOAL_POS:
        winner = 0
    elif positions[1] >= GOAL_POS:
        winner = 1

    return logs, winner


def apply_specials(positions, stuns, logs):
    for player in range(2):
        for _ in range(3):
            tile = positions[player]
            effect = SPECIAL_TILES.get(tile)
            if effect is None:
                break

            if effect == "rocket":
                positions[player] = clamp_pos(positions[player] + 4)
                logs.append(f"P{player + 1} ROCKET +4")
                continue

            if effect == "broken":
                stuns[player] = max(stuns[player], 1)
                logs.append(f"P{player + 1} BROKEN: skip next")
                break

            if effect == "ladder_up":
                positions[player] = LADDER_MAP[10]
                logs.append(f"P{player + 1} LADDER 10->15")
                continue

            if effect == "ladder_down":
                positions[player] = LADDER_MAP[18]
                logs.append(f"P{player + 1} LADDER 18->12")
                continue

            if effect == "mine":
                positions[player] = clamp_pos(positions[player] - 2)
                logs.append(f"P{player + 1} MINE -2")
                break

            if effect == "warp":
                other = 1 - player
                positions[player], positions[other] = positions[other], positions[player]
                logs.append("WARP: P1 <-> P2")
                break


def build_board_cells(board_w, board_h):
    cols = 5
    rows = 4
    top = 44
    pad = 8
    cw = (board_w - (pad * 2)) // cols
    ch = (board_h - top - pad) // rows

    cells = []
    for r in range(rows):
        y = board_h - pad - (r + 1) * ch
        if r % 2 == 0:
            order = range(cols)
        else:
            order = range(cols - 1, -1, -1)

        for c in order:
            x = pad + c * cw
            cells.append((x, y, cw, ch))

    return cells[:BOARD_SIZE]


def draw_board(width, height, positions, stuns, p1_gesture, p2_gesture, round_idx, status, logs):
    panel = np.full((height, width, 3), (28, 28, 35), dtype=np.uint8)
    cells = build_board_cells(width, height)

    cv2.putText(panel, "YOLO RPS Racing", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 230, 120), 2)
    cv2.putText(panel, f"Round {round_idx} | q:quit r:reset", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.1, (220, 220, 220), 1)

    tile_color = {
        "rocket": (70, 110, 255),
        "broken": (70, 70, 70),
        "ladder_up": (60, 180, 80),
        "mine": (40, 110, 210),
        "warp": (190, 120, 40),
        "ladder_down": (80, 170, 200),
    }
    short_name = {
        "rocket": "R",
        "broken": "B",
        "ladder_up": "L+",
        "mine": "M",
        "warp": "W",
        "ladder_down": "L-",
    }

    for idx, (x, y, cw, ch) in enumerate(cells):
        effect = SPECIAL_TILES.get(idx)
        c = tile_color.get(effect, (55, 55, 65))
        cv2.rectangle(panel, (x, y), (x + cw - 2, y + ch - 2), c, -1)
        cv2.rectangle(panel, (x, y), (x + cw - 2, y + ch - 2), (220, 220, 220), 1)
        cv2.putText(panel, str(idx), (x + 4, y + 14), cv2.FONT_HERSHEY_PLAIN, 0.9, (250, 250, 250), 1)
        if effect is not None:
            cv2.putText(panel, short_name[effect], (x + 4, y + ch - 8), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

    p1x, p1y, p1w, p1h = cells[positions[0]]
    p2x, p2y, p2w, p2h = cells[positions[1]]

    cv2.circle(panel, (p1x + p1w // 2 - 10, p1y + p1h // 2), 10, (30, 220, 80), -1)
    cv2.circle(panel, (p2x + p2w // 2 + 10, p2y + p2h // 2), 10, (40, 120, 255), -1)
    cv2.putText(panel, "1", (p1x + p1w // 2 - 14, p1y + p1h // 2 + 4), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)
    cv2.putText(panel, "2", (p2x + p2w // 2 + 6, p2y + p2h // 2 + 4), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 1)

    p1_txt = "skip" if stuns[0] > 0 else (ansToText[p1_gesture] if p1_gesture is not None else "-")
    p2_txt = "skip" if stuns[1] > 0 else (ansToText[p2_gesture] if p2_gesture is not None else "-")
    cv2.putText(panel, f"P1:{p1_txt} pos:{positions[0]} stun:{stuns[0]}", (10, height - 56), cv2.FONT_HERSHEY_PLAIN, 1.0, (120, 255, 160), 1)
    cv2.putText(panel, f"P2:{p2_txt} pos:{positions[1]} stun:{stuns[1]}", (10, height - 38), cv2.FONT_HERSHEY_PLAIN, 1.0, (120, 180, 255), 1)
    cv2.putText(panel, status, (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 240, 150), 1)

    y = 60
    for line in logs[-3:]:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (210, 210, 210), 1)
        y += 14

    return panel


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("RPS Racing", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RPS Racing", 400, 600)

positions = [0, 0]
stuns = [0, 0]
round_idx = 1
hold_start = None
last_round_time = time.time()
winner = None
status = "Show your hand"
history_logs = ["Rule: left hand=P1, right hand=P2"]

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = infer_gestures(frame)
    p1_gesture, p2_gesture, p1_det, p2_det = select_players(detections, frame.shape[1])

    if p1_det is not None:
        x1, y1, _, _ = p1_det["box"]
        cv2.putText(frame, "P1", (x1, y1 + 16), cv2.FONT_HERSHEY_PLAIN, 1.2, (100, 255, 120), 2)
    if p2_det is not None:
        x1, y1, _, _ = p2_det["box"]
        cv2.putText(frame, "P2", (x1, y1 + 16), cv2.FONT_HERSHEY_PLAIN, 1.2, (100, 180, 255), 2)

    now = time.time()
    can_play_now = [stuns[0] == 0, stuns[1] == 0]

    if winner is None:
        if (not can_play_now[0]) and (not can_play_now[1]):
            status = "Both stunned: auto resolve"
            if now - last_round_time >= ROUND_COOLDOWN_SEC:
                logs, winner = resolve_round(None, None, positions, stuns, round_idx)
                history_logs.extend(logs)
                round_idx += 1
                last_round_time = now
                hold_start = None
        else:
            ready = ((not can_play_now[0]) or (p1_gesture is not None)) and ((not can_play_now[1]) or (p2_gesture is not None))

            if ready:
                if hold_start is None:
                    hold_start = now
                remain = max(0.0, TURN_HOLD_SEC - (now - hold_start))
                status = f"Locking gestures... {remain:.1f}s"

                if (now - hold_start >= TURN_HOLD_SEC) and (now - last_round_time >= ROUND_COOLDOWN_SEC):
                    logs, winner = resolve_round(p1_gesture, p2_gesture, positions, stuns, round_idx)
                    history_logs.extend(logs)
                    round_idx += 1
                    last_round_time = now
                    hold_start = None
            else:
                hold_start = None
                status = "Need both players (or skip state)"
    else:
        if winner == -1:
            status = "Draw! both reached goal"
        else:
            status = f"Player {winner + 1} wins! press r"

    cur_time = time.time()
    fps = 1.0 / max(1e-6, cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2)

    board_panel = draw_board(
        frame.shape[1],
        300,
        positions,
        stuns,
        p1_gesture,
        p2_gesture,
        round_idx,
        status,
        history_logs,
    )
    view = cv2.vconcat([frame, board_panel])
    cv2.imshow("RPS Racing", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        positions = [0, 0]
        stuns = [0, 0]
        round_idx = 1
        hold_start = None
        last_round_time = time.time()
        winner = None
        status = "Reset complete"
        history_logs = ["Rule: left hand=P1, right hand=P2"]

cap.release()
cv2.destroyAllWindows()
