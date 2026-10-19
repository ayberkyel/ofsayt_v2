"""
"Sahne" (scene) = bir ofsayt pozisyonunun karar için gereken tüm bilgisi.
Arayüz bir sahneyi JSON olarak kaydeder; aynı fonksiyon (decide) hem arayüzde,
hem duyarlılık analizinde, hem de toplu değerlendirmede kullanılır. Böylece raporda
sunulan sayılar arayüzün verdiği kararla birebir aynı kodla üretilir.

Sahne sözlüğü:
{
  "image": "sahne.jpg", "image_size": [w, h],
  "calib": [{"name": "...", "px": [u, v], "pitch": [x, y]}, ...],
  "ball_px": [u, v] | null,
  "detections": [Detection.to_dict(), ...],
  "excluded_ids": [..],               # yanlış tespit olarak işaretlenenler
  "attack_right": true,               # hücum x=105 kalesine doğru mu
  "attacking_team": 0 | 1,
  "attacker_id": 7,
  "body_point": "center" | "nearest", # alt-orta nokta / kaleye en yakın alt köşe
  "band": 0.5,
  "gt": "OFSAYT" | "OFSAYT DEĞİL" | null   # (değerlendirme için gerçek karar)
}
"""
import numpy as np

from .calibration import compute_homography, image_to_pitch
from .detector import Detection
from .offside import evaluate
from .pitch import HALF_X, PITCH_LENGTH


def player_pitch_x(det, H, attack_right, body_point="center", offset=(0.0, 0.0)):
    """Oyuncunun ofsayt için kullanılan saha x'i. offset: duyarlılık analizi için piksel gürültüsü."""
    if body_point == "nearest":
        pts = [det.foot, *det.foot_corners()]
    else:
        pts = [det.foot]
    pts = np.asarray(pts) + np.asarray(offset)
    xs = image_to_pitch(pts, H)[:, 0]
    u = xs if attack_right else PITCH_LENGTH - xs
    return float(xs[int(np.argmax(u))])


def offpitch_ids(dets, H, tol=1.0):
    """Ayağı saha dışında kalan kişiler (top toplayıcı, yedek, yan hakem, fotoğrafçı...).

    Bunlar savunmacı sayılırsa ofsayt çizgisi tamamen yanlış yere kayar. Örnek: testgörseli.jpg'de
    kale arkasındaki görevli (x = 109 m) "savunmacı" sayılınca çizgi kalecinin hizasına (99 m)
    gitti ve bariz ofsayttaki oyuncu "OFSAYT DEĞİL" çıktı. Bir oyuncu sahanın içindedir; tol
    (metre) kalibrasyon hatası payıdır.
    """
    from .pitch import PITCH_WIDTH
    out = set()
    for d in dets:
        x, y = image_to_pitch([d.foot], H)[0]
        if x < -tol or x > PITCH_LENGTH + tol or y < -tol or y > PITCH_WIDTH + tol:
            out.add(d.id)
    return out


def effective_excluded(scene, dets, H):
    """Kullanıcının yok saydıkları + (aksi istenmedikçe) saha dışındakiler."""
    excluded = set(scene.get("excluded_ids", []))
    if scene.get("exclude_offpitch", True):
        excluded |= offpitch_ids(dets, H) - set(scene.get("force_include_ids", []))
    return excluded


def split_sides(scene, dets, H):
    """Hücumcu ve savunmacı listelerini oluşturur."""
    attack_right = scene["attack_right"]
    defending_team = 1 - scene["attacking_team"]
    excluded = effective_excluded(scene, dets, H)
    defenders = []
    for d in dets:
        if d.id in excluded or d.role == "referee":
            continue
        if d.role == "player" and d.team == defending_team:
            defenders.append(d)
        elif d.role == "goalkeeper":
            # Kaleci: savunulan kaleye yakın yarıdaysa savunan takımın kalecisidir
            gx = image_to_pitch([d.foot], H)[0, 0]
            if (gx > HALF_X) == attack_right:
                defenders.append(d)
    return defenders


def decide(scene, calib_px=None, foot_noise=None, rng=None):
    """Sahneden karar üretir.

    calib_px   : kalibrasyon piksellerinin yerine kullanılacak (gürültülü) değerler
    foot_noise : piksel cinsinden; verilirse her oyuncu ayak noktasına ±foot_noise eklenir
    Dönüş: (OffsideResult, H, info)
    """
    rng = rng or np.random.default_rng()
    img_pts = calib_px if calib_px is not None else [c["px"] for c in scene["calib"]]
    pitch_pts = [c["pitch"] for c in scene["calib"]]
    H, inliers = compute_homography(img_pts, pitch_pts)

    dets = [Detection.from_dict(d) for d in scene["detections"]]
    by_id = {d.id: d for d in dets}
    attacker = by_id[scene["attacker_id"]]
    attack_right = scene["attack_right"]
    bp = scene.get("body_point", "center")

    def noise():
        if not foot_noise:
            return (0.0, 0.0)
        return tuple(rng.uniform(-foot_noise, foot_noise, size=2))

    defenders = split_sides(scene, dets, H)
    def_xs = [player_pitch_x(d, H, attack_right, bp, noise()) for d in defenders]
    att_x = player_pitch_x(attacker, H, attack_right, bp, noise())
    ball_x = None
    if scene.get("ball_px") is not None:
        ball_x = float(image_to_pitch([np.asarray(scene["ball_px"]) + noise()], H)[0, 0])

    res = evaluate(att_x, def_xs, ball_x, attack_right, band=scene.get("band", 0.5))
    info = {
        "defender_ids": [d.id for d in defenders],
        "defender_xs": def_xs,
        "inliers": inliers.tolist(),
        "excluded_ids": sorted(effective_excluded(scene, dets, H)),
        "offpitch_ids": sorted(offpitch_ids(dets, H)),
    }
    return res, H, info
