"""Görselleştirme: kutular, top, ofsayt çizgileri ve kuşbakışı mini saha."""
import cv2
import numpy as np

from .calibration import pitch_to_image, image_to_pitch
from .pitch import HALF_X, PITCH_LENGTH, PITCH_WIDTH, template_lines

# BGR renkler
TEAM_COLORS = {0: (255, 140, 0), 1: (40, 40, 230), None: (200, 200, 200)}
ROLE_COLORS = {"goalkeeper": (0, 215, 255), "referee": (30, 30, 30)}
DECISION_COLORS = {"OFSAYT": (0, 0, 220), "OFSAYT DEĞİL": (0, 170, 0), "SINIRDA": (0, 165, 255)}
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
    cv2.circle(img, (x, y), r, (0, 0, 0), 3)
    cv2.circle(img, (x, y), r, color, 2)
    if label:
        cv2.putText(img, label, (x + r + 3, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
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
    # Ofsayt çizgisi (kırmızı) ve hücumcu çizgisi (mavi) — TV'deki VAR çizimine benzer
    draw_pitch_x_line(out, H, result.line_x, (0, 0, 255), 3)
    draw_pitch_x_line(out, H, result.attacker_x, (255, 120, 0), 2)
    if ball_px is not None:
        draw_point(out, ball_px, (255, 255, 255), 7, "top")
    for d in dets:
        if d.id in defender_ids:
            cv2.circle(out, tuple(map(int, d.foot)), 9, (0, 0, 255), 2)
    color = DECISION_COLORS[result.decision]
    text = f"{DECISION_ASCII[result.decision]}  ({result.diff_m:+.2f} m)"
    cv2.rectangle(out, (10, 10), (20 + 22 * len(text), 60), color, -1)
    cv2.putText(out, text, (20, 47), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    return out


def keypoint_guide(side, target=None, done=(), scale=6, margin=14):
    """Kalibrasyon rehberi: sahanın görüntüdeki yarısı; hedef nokta kırmızı halka, tıklananlar yeşil.

    Kamera açısına benzesin diye yakın yan çizgi (y = 0) ALTTA çizilir.
    """
    from .pitch import KEYPOINTS
    x_lo, x_hi = (HALF_X - 6, PITCH_LENGTH) if side == "SAĞ" else (0.0, HALF_X + 6)
    W = int((x_hi - x_lo) * scale) + 2 * margin
    Hh = int(PITCH_WIDTH * scale) + 2 * margin
    img = np.full((Hh, W, 3), (60, 140, 60), np.uint8)

    def to_px(p):
        return int(margin + (p[0] - x_lo) * scale), int(margin + (PITCH_WIDTH - p[1]) * scale)

    for line in template_lines():
        cv2.polylines(img, [np.int32([to_px(p) for p in line])], False, (235, 235, 235), 1, cv2.LINE_AA)
    for n in done:
        if n in KEYPOINTS:
            cv2.circle(img, to_px(KEYPOINTS[n]), 5, (0, 220, 0), -1)
    if target in KEYPOINTS:
        c = to_px(KEYPOINTS[target])
        cv2.circle(img, c, 11, (0, 0, 255), 3)
        cv2.circle(img, c, 3, (0, 0, 255), -1)
    cv2.putText(img, "kamera tarafi (yakin)", (margin, Hh - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def minimap(dets, H, excluded=(), result=None, ball_px=None, attacker_id=None, scale=8, margin=20):
    """Kuşbakışı saha: oyuncuların homography ile hesaplanan gerçek konumları."""
    W = int(PITCH_LENGTH * scale) + 2 * margin
    Hh = int(PITCH_WIDTH * scale) + 2 * margin
    img = np.full((Hh, W, 3), (60, 140, 60), np.uint8)

    def to_px(p):
        # y = 0 (yakın yan çizgi) altta görünsün diye y eksenini ters çeviriyoruz
        return int(margin + p[0] * scale), int(margin + (PITCH_WIDTH - p[1]) * scale)

    for line in template_lines():
        cv2.polylines(img, [np.int32([to_px(p) for p in line])], False, (255, 255, 255), 2, cv2.LINE_AA)
    if result is not None:
        lx = to_px((result.line_x, 0))[0]
        cv2.line(img, (lx, margin), (lx, Hh - margin), (0, 0, 255), 2)
        ax = to_px((result.attacker_x, 0))[0]
        cv2.line(img, (ax, margin), (ax, Hh - margin), (255, 120, 0), 1)
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
        cv2.circle(img, to_px(b), 5, (255, 255, 255), -1)
        cv2.circle(img, to_px(b), 5, (0, 0, 0), 1)
    return img
