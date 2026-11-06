"""Görselleştirme: kutular, top, ofsayt çizgileri ve kuşbakışı mini saha."""
import cv2
import numpy as np

from .calibration import pitch_to_image, image_to_pitch
from .pitch import HALF_X, PITCH_LENGTH, PITCH_WIDTH, template_lines

# BGR renkler
TEAM_COLORS = {0: (255, 140, 0), 1: (40, 40, 230), None: (200, 200, 200)}
ROLE_COLORS = {"goalkeeper": (0, 215, 255), "referee": (30, 30, 30)}
# Karar renkleri (arayüzdeki karar kartıyla aynı: src/ui.py VERDICT)
DECISION_COLORS = {"OFSAYT": (111, 67, 232), "OFSAYT DEĞİL": (107, 163, 47), "SINIRDA": (49, 165, 242)}
# Arayüz paleti (BGR): bubblegum pink, galactic violet, açık lila, marshmallow beyazı
PINK = (182, 126, 255)
VIOLET = (122, 42, 74)
VIOLET_2 = (160, 63, 107)
LILAC = (245, 167, 201)
PINK_LIGHT = (230, 211, 255)
MARSHMALLOW = (243, 248, 253)
# cv2.putText Türkçe karakter çizemediği için görüntü üzerindeki yazılar ASCII
DECISION_ASCII = {"OFSAYT": "OFSAYT", "OFSAYT DEĞİL": "OFSAYT DEGIL", "SINIRDA": "SINIRDA"}


def det_color(d):
    return ROLE_COLORS.get(d.role, TEAM_COLORS.get(d.team, TEAM_COLORS[None]))


def draw_detections(img, dets, excluded=(), highlight=None, scale=1.0):
    out = img.copy()
    t = max(1, int(round(2 * scale)))
    for d in dets:
        if d.id in excluded:
            continue
        x1, y1, x2, y2 = map(int, d.box)
        c = det_color(d)
        thick = t * 2 if d.id == highlight else t
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thick)
        label = str(d.id) + ("K" if d.role == "goalkeeper" else "H" if d.role == "referee" else "")
        fs = 0.5 * scale
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t, cv2.LINE_AA)
        fx, fy = map(int, d.foot)
        cv2.circle(out, (fx, fy), 3 * t, c, -1)
    return out


def draw_point(img, pt, color=(255, 255, 255), r=8, label=None):
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(img, (x, y), r, VIOLET, 4)
    cv2.circle(img, (x, y), r, color, 2)
    if label:
        cv2.putText(img, label, (x + r + 3, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.6, VIOLET, 4, cv2.LINE_AA)
        cv2.putText(img, label, (x + r + 3, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def draw_pitch_x_line(img, H, x, color, thickness=2):
    """Saha koordinatında x = sabit olan doğruyu (kale çizgisine paralel) görüntüye çizer.
    Perspektif nedeniyle ekranda eğik görünür — bu doğru geometridir."""
    ys = np.linspace(0, PITCH_WIDTH, 60)
    pts = pitch_to_image(np.stack([np.full_like(ys, x), ys], 1), H)
    pts = pts[~np.isnan(pts).any(1)]
    if len(pts) > 1:
        cv2.polylines(img, [np.int32(pts)], False, color, thickness, cv2.LINE_AA)
    return img


def draw_result(img, dets, H, result, attacker_id, defender_ids, ball_px=None, excluded=()):
    out = draw_detections(img, dets, excluded, highlight=attacker_id)
    # Ofsayt çizgisi (pembe, mor gölgeli) ve hücumcu çizgisi (açık lila) — TV'deki VAR çizimine benzer
    # Hücumcu çizgisi önce, ofsayt çizgisi en üstte (çok yakınken asıl çizgi görünür kalsın)
    draw_pitch_x_line(out, H, result.attacker_x, VIOLET, 4)
    draw_pitch_x_line(out, H, result.attacker_x, LILAC, 2)
    draw_pitch_x_line(out, H, result.line_x, VIOLET, 6)
    draw_pitch_x_line(out, H, result.line_x, PINK, 3)
    if ball_px is not None:
        draw_point(out, ball_px, MARSHMALLOW, 7, "top")
    for d in dets:
        if d.id in defender_ids:
            cv2.circle(out, tuple(map(int, d.foot)), 10, VIOLET, 4)
            cv2.circle(out, tuple(map(int, d.foot)), 10, PINK, 2)
    # Karar etiketi: mor zemin, solda karar renginde şerit
    color = DECISION_COLORS[result.decision]
    text = f"{DECISION_ASCII[result.decision]}  ({result.diff_m:+.2f} m)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    cv2.rectangle(out, (10, 10), (44 + tw, 30 + th + 12), VIOLET, -1)
    cv2.rectangle(out, (10, 10), (20, 30 + th + 12), color, -1)
    cv2.rectangle(out, (10, 30 + th + 8), (44 + tw, 30 + th + 12), PINK, -1)
    cv2.putText(out, text, (30, 22 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.1, MARSHMALLOW, 3, cv2.LINE_AA)
    return out


def keypoint_guide(side, target=None, done=(), scale=6, margin=14):
    """Kalibrasyon rehberi: sahanın görüntüdeki yarısı; hedef nokta kırmızı halka, tıklananlar yeşil.

    Kamera açısına benzesin diye yakın yan çizgi (y = 0) ALTTA çizilir.
    """
    from .pitch import KEYPOINTS
    x_lo, x_hi = (HALF_X - 6, PITCH_LENGTH) if side == "SAĞ" else (0.0, HALF_X + 6)
    W = int((x_hi - x_lo) * scale) + 2 * margin
    Hh = int(PITCH_WIDTH * scale) + 2 * margin
    img = np.full((Hh, W, 3), VIOLET, np.uint8)

    def to_px(p):
        return int(margin + (p[0] - x_lo) * scale), int(margin + (PITCH_WIDTH - p[1]) * scale)

    for line in template_lines():
        cv2.polylines(img, [np.int32([to_px(p) for p in line])], False, MARSHMALLOW, 1, cv2.LINE_AA)
    for n in done:
        if n in KEYPOINTS:
            cv2.circle(img, to_px(KEYPOINTS[n]), 5, PINK_LIGHT, -1)
    if target in KEYPOINTS:
        c = to_px(KEYPOINTS[target])
        cv2.circle(img, c, 11, PINK, 3)
        cv2.circle(img, c, 3, PINK, -1)
    cv2.putText(img, "kamera tarafi (yakin)", (margin, Hh - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def minimap(dets, H, excluded=(), result=None, ball_px=None, attacker_id=None, scale=8, margin=20):
    """Kuşbakışı saha: oyuncuların homography ile hesaplanan gerçek konumları."""
    W = int(PITCH_LENGTH * scale) + 2 * margin
    Hh = int(PITCH_WIDTH * scale) + 2 * margin
    img = np.full((Hh, W, 3), VIOLET, np.uint8)
    # hafif çim şeritleri (mor tonlarında)
    for k in range(0, int(PITCH_LENGTH), 10):
        if (k // 10) % 2 == 0:
            x0 = int(margin + k * scale)
            cv2.rectangle(img, (x0, margin), (min(W - margin, x0 + int(10 * scale)), Hh - margin), VIOLET_2, -1)

    def to_px(p):
        # y = 0 (yakın yan çizgi) altta görünsün diye y eksenini ters çeviriyoruz
        return int(margin + p[0] * scale), int(margin + (PITCH_WIDTH - p[1]) * scale)

    for line in template_lines():
        cv2.polylines(img, [np.int32([to_px(p) for p in line])], False, MARSHMALLOW, 2, cv2.LINE_AA)
    if result is not None:
        lx = to_px((result.line_x, 0))[0]
        cv2.line(img, (lx, margin), (lx, Hh - margin), PINK, 3)
        ax = to_px((result.attacker_x, 0))[0]
        cv2.line(img, (ax, margin), (ax, Hh - margin), LILAC, 2)
    for d in dets:
        if d.id in excluded:
            continue
        p = image_to_pitch([d.foot], H)[0]
        if not (-10 < p[0] < PITCH_LENGTH + 10 and -10 < p[1] < PITCH_WIDTH + 10):
            continue
        c = to_px(p)
        cv2.circle(img, c, 9 if d.id == attacker_id else 7, det_color(d), -1)
        cv2.circle(img, c, 9 if d.id == attacker_id else 7, (255, 255, 255), 1)
        cv2.putText(img, str(d.id), (c[0] + 8, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if ball_px is not None:
        b = image_to_pitch([ball_px], H)[0]
        cv2.circle(img, to_px(b), 6, MARSHMALLOW, -1)
        cv2.circle(img, to_px(b), 6, PINK, 2)
    return img


def var_zoom(img, dets, H, result, attacker_id, line_defender_id=None, ball_px=None, out_w=960, aspect=16 / 9):
    """VAR yakın çekimi: hücumcu ile çizgiyi belirleyen savunmacının (veya topun) çevresini kırpıp büyütür.

    Çizgiler büyütülmüş görüntüye YENİDEN çizilir (kırpılan çizimi büyütmek çizgileri kalınlaştırırdı):
    kırpma + ölçek bir A dönüşümüdür, yeni görüntü için homografi H' = H · A⁻¹ olur.
    Dönüş: BGR görüntü veya kırpılacak oyuncu bulunamazsa None.
    """
    by_id = {d.id: d for d in dets}
    att = by_id.get(attacker_id)
    if att is None:
        return None
    boxes = [att.box]
    if line_defender_id in by_id:
        boxes.append(by_id[line_defender_id].box)
    elif ball_px is not None:
        bx, by = ball_px
        boxes.append((bx - 10, by - 10, bx + 10, by + 10))
    b = np.array(boxes, float)
    x1, y1 = b[:, 0].min(), b[:, 1].min()
    x2, y2 = b[:, 2].max(), b[:, 3].max()
    h_img, w_img = img.shape[:2]
    ph = max(b[:, 3] - b[:, 1])                        # oyuncu boyu kadar pay
    x1, x2, y1, y2 = x1 - ph, x2 + ph, y1 - 0.6 * ph, y2 + 0.6 * ph
    cw = max(x2 - x1, (y2 - y1) * aspect, 0.22 * w_img)
    ch = cw / aspect
    if cw > w_img or ch > h_img:                       # oyuncular çok uzak: tüm kare
        cw, ch = w_img, w_img / aspect
        if ch > h_img:
            ch, cw = h_img, h_img * aspect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0 = int(np.clip(cx - cw / 2, 0, w_img - cw))
    y0 = int(np.clip(cy - ch / 2, 0, h_img - ch))
    cw, ch = int(cw), int(ch)
    sc = out_w / cw
    out = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (out_w, int(round(ch * sc))), interpolation=cv2.INTER_CUBIC)
    A = np.array([[sc, 0, -sc * x0], [0, sc, -sc * y0], [0, 0, 1]], float)
    Hc = H @ np.linalg.inv(A)

    def tb(box):
        return (int(sc * (box[0] - x0)), int(sc * (box[1] - y0)), int(sc * (box[2] - x0)), int(sc * (box[3] - y0)))

    # Çizgiler: ofsayt çizgisi pembe, hücumcu lila (mor gölgeli)
    draw_pitch_x_line(out, Hc, result.attacker_x, VIOLET, 5)
    draw_pitch_x_line(out, Hc, result.attacker_x, LILAC, 2)
    draw_pitch_x_line(out, Hc, result.line_x, VIOLET, 7)
    draw_pitch_x_line(out, Hc, result.line_x, PINK, 3)
    labels = [(att, "HUCUMCU", LILAC)]
    if line_defender_id in by_id:
        labels.append((by_id[line_defender_id], "SAVUNMACI", PINK))
    for d, text, col in labels:
        bx1, by1, bx2, by2 = tb(d.box)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), VIOLET, 5)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), col, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (bx1, by1 - th - 12), (bx1 + tw + 12, by1), VIOLET, -1)
        cv2.putText(out, text, (bx1 + 6, by1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    if ball_px is not None:
        bx, by = sc * (ball_px[0] - x0), sc * (ball_px[1] - y0)
        if 0 <= bx < out.shape[1] and 0 <= by < out.shape[0]:
            draw_point(out, (bx, by), MARSHMALLOW, 10, "top")
    # Başlık şeridi
    text = f"VAR YAKIN CEKIM  |  {DECISION_ASCII[result.decision]}  {result.diff_m:+.2f} m"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(out, (0, 0), (tw + 40, th + 22), VIOLET, -1)
    cv2.rectangle(out, (0, 0), (8, th + 22), DECISION_COLORS[result.decision], -1)
    cv2.rectangle(out, (0, th + 18), (tw + 40, th + 22), PINK, -1)
    cv2.putText(out, text, (20, th + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.75, MARSHMALLOW, 2, cv2.LINE_AA)
    return out
