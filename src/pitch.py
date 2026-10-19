"""
Saha şablonu (FIFA/IFAB standart ölçüleri, metre).

Koordinat sistemi (rapor Bölüm 8.2):
    x: saha uzunluğu, 0 .. 105  ->  x = 0 EKRANDA SOLDAKİ kale çizgisi
    y: saha genişliği, 0 .. 68  ->  y = 0 KAMERAYA YAKIN (ekranda alttaki) yan çizgi
                                     y = 68 KAMERAYA UZAK (ekranda üstteki) yan çizgi

Bu dosyada iki şey var:
  1) KEYPOINTS : Kullanıcının kalibrasyonda tıklayabileceği, gerçek konumu bilinen noktalar
  2) template_lines() : Kalibrasyonu doğrulamak için görüntüye geri çizilen saha çizgileri
"""
import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_X = PITCH_LENGTH / 2          # 52.5

# Standart ölçüler
PENALTY_DEPTH = 16.5               # ceza sahası derinliği
PENALTY_HALF_W = 20.16             # ceza sahası yarı genişliği (40.32 / 2)
GOAL_AREA_DEPTH = 5.5              # altıpas derinliği
GOAL_AREA_HALF_W = 9.16            # altıpas yarı genişliği (18.32 / 2)
PENALTY_SPOT = 11.0
CIRCLE_R = 9.15
CY = PITCH_WIDTH / 2               # 34

# Ceza yayının ceza sahası çizgisini kestiği nokta: penaltı noktasından 9.15 m uzakta
_ARC_DY = float(np.sqrt(CIRCLE_R ** 2 - (PENALTY_DEPTH - PENALTY_SPOT) ** 2))  # ~7.31


def _left_side_points():
    """Sol kale tarafındaki noktalar. Sağ taraf bunların aynası (x -> 105 - x)."""
    return {
        "köşe bayrağı (yakın)": (0.0, 0.0),
        "köşe bayrağı (uzak)": (0.0, PITCH_WIDTH),
        "ceza sahası - kale çizgisi, yakın köşe": (0.0, CY - PENALTY_HALF_W),
        "ceza sahası - kale çizgisi, uzak köşe": (0.0, CY + PENALTY_HALF_W),
        "ceza sahası - saha içi, yakın köşe": (PENALTY_DEPTH, CY - PENALTY_HALF_W),
        "ceza sahası - saha içi, uzak köşe": (PENALTY_DEPTH, CY + PENALTY_HALF_W),
        "altıpas - kale çizgisi, yakın köşe": (0.0, CY - GOAL_AREA_HALF_W),
        "altıpas - kale çizgisi, uzak köşe": (0.0, CY + GOAL_AREA_HALF_W),
        "altıpas - saha içi, yakın köşe": (GOAL_AREA_DEPTH, CY - GOAL_AREA_HALF_W),
        "altıpas - saha içi, uzak köşe": (GOAL_AREA_DEPTH, CY + GOAL_AREA_HALF_W),
        "penaltı noktası": (PENALTY_SPOT, CY),
        "ceza yayı - ceza sahası kesişimi, yakın": (PENALTY_DEPTH, CY - _ARC_DY),
        "ceza yayı - ceza sahası kesişimi, uzak": (PENALTY_DEPTH, CY + _ARC_DY),
    }


def _build_keypoints():
    kp = {}
    for name, (x, y) in _left_side_points().items():
        kp[f"SOL {name}"] = (x, y)
        kp[f"SAĞ {name}"] = (PITCH_LENGTH - x, y)
    kp["ORTA çizgi - yan çizgi, yakın"] = (HALF_X, 0.0)
    kp["ORTA çizgi - yan çizgi, uzak"] = (HALF_X, PITCH_WIDTH)
    kp["ORTA nokta (santra)"] = (HALF_X, CY)
    kp["ORTA yuvarlak - orta çizgi, yakın"] = (HALF_X, CY - CIRCLE_R)
    kp["ORTA yuvarlak - orta çizgi, uzak"] = (HALF_X, CY + CIRCLE_R)
    return kp


KEYPOINTS = _build_keypoints()

# Kalibrasyonda önerilen sıra. Ceza sahası / altıpas / yay ölçüleri HER sahada aynıdır (IFAB).
# Köşe bayrakları ve yan çizgi kesişimleri sahanın genişliğine/uzunluğuna bağlıdır
# (64-75 m x 100-110 m) ve şablon 105x68 varsaydığı için en sona konur.
# Sıra, ilk 4 noktanın hiçbir 3'ü aynı doğru üzerinde olmayacak şekilde seçildi (homography için şart).
# Ceza yayı kesişimleri ceza sahası ön çizgisi üzerindedir; bu yüzden ek nokta olarak sonda.
# Ölçüm (testgörsel2, ±2 px tıklama): sadece altıpas -> oyuncu bölgesinde %95 hata 2.6 m;
# ceza sahası 3 köşe + altıpas 2 -> 0.3 m. Bu yüzden ceza sahası köşeleri önce gelir.
_RECOMMENDED_SIDE = [
    "ceza sahası - saha içi, yakın köşe",
    "ceza sahası - saha içi, uzak köşe",
    "ceza sahası - kale çizgisi, uzak köşe",
    "ceza sahası - kale çizgisi, yakın köşe",
    "altıpas - kale çizgisi, uzak köşe",
    "altıpas - saha içi, uzak köşe",
    "altıpas - kale çizgisi, yakın köşe",
    "altıpas - saha içi, yakın köşe",
    "penaltı noktası",
    "ceza yayı - ceza sahası kesişimi, yakın",
    "ceza yayı - ceza sahası kesişimi, uzak",
]
_CENTER = ["ORTA nokta (santra)", "ORTA yuvarlak - orta çizgi, uzak", "ORTA yuvarlak - orta çizgi, yakın",
           "ORTA çizgi - yan çizgi, uzak", "ORTA çizgi - yan çizgi, yakın"]
SIZE_DEPENDENT = {"köşe bayrağı (yakın)", "köşe bayrağı (uzak)",
                  "ORTA çizgi - yan çizgi, uzak", "ORTA çizgi - yan çizgi, yakın"}


def ordered_keypoints(side):
    """side: 'SAĞ' veya 'SOL' (görüntüdeki kale). Önerilen sırada nokta isimleri döndürür."""
    names = [f"{side} {n}" for n in _RECOMMENDED_SIDE] + _CENTER
    names += [f"{side} köşe bayrağı (uzak)", f"{side} köşe bayrağı (yakın)"]
    return names


def is_size_dependent(name):
    return any(name.endswith(s) for s in SIZE_DEPENDENT)


def _segment(p, q, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) * np.array(p) + t * np.array(q)


def _arc(cx, cy, r, a0, a1, n=60):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def template_lines():
    """Saha çizgilerini metre cinsinden polyline listesi olarak döndürür."""
    L, W = PITCH_LENGTH, PITCH_WIDTH
    lines = [
        _segment((0, 0), (L, 0)), _segment((L, 0), (L, W)),
        _segment((L, W), (0, W)), _segment((0, W), (0, 0)),
        _segment((HALF_X, 0), (HALF_X, W)),
        _arc(HALF_X, CY, CIRCLE_R, 0, 2 * np.pi, 120),
    ]
    for side in (0, 1):
        def fx(x):
            return x if side == 0 else L - x
        for depth, hw in ((PENALTY_DEPTH, PENALTY_HALF_W), (GOAL_AREA_DEPTH, GOAL_AREA_HALF_W)):
            lines.append(_segment((fx(0), CY - hw), (fx(depth), CY - hw)))
            lines.append(_segment((fx(depth), CY - hw), (fx(depth), CY + hw)))
            lines.append(_segment((fx(depth), CY + hw), (fx(0), CY + hw)))
        # Ceza yayı (yalnızca ceza sahası dışında kalan kısım)
        a = np.arccos((PENALTY_DEPTH - PENALTY_SPOT) / CIRCLE_R)
        if side == 0:
            lines.append(_arc(PENALTY_SPOT, CY, CIRCLE_R, -a, a))
        else:
            lines.append(_arc(L - PENALTY_SPOT, CY, CIRCLE_R, np.pi - a, np.pi + a))
    return lines
