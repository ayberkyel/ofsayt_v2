"""
Çekirdek geometri ve karar motoru testleri (gerçek görüntü gerekmez).

Sentetik bir "kamera" (bilinen homography) kurulur, oyuncular sahada BİLİNEN metre
konumlarına yerleştirilir, görüntüdeki kutuları üretilir ve sistemin bu konumları geri
hesaplayıp doğru kararı verip vermediği kontrol edilir.

Çalıştırma:  python tests/test_core.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import compute_homography, image_to_pitch, pitch_to_image, reprojection_error  # noqa
from src.offside import evaluate, OFFSIDE, ONSIDE, BORDERLINE  # noqa
from src.pipeline import decide  # noqa
from src.pitch import KEYPOINTS  # noqa

# Sağ ceza sahasını gösteren, perspektifli sahte bir kamera (saha metre -> piksel)
PITCH_QUAD = np.float32([[70, 0], [105, 0], [105, 68], [70, 68]])
IMAGE_QUAD = np.float32([[100, 1000], [1850, 1040], [1500, 250], [500, 230]])
H_TRUE_P2I = cv2.getPerspectiveTransform(PITCH_QUAD, IMAGE_QUAD)


def proj(p):
    return cv2.perspectiveTransform(np.float32([[p]]), H_TRUE_P2I)[0, 0]


def make_det(i, pitch_xy, role="player", team=None):
    fx, fy = proj(pitch_xy)
    return {"id": i, "role": role, "conf": 0.9, "box": [fx - 15, fy - 70, fx + 15, fy], "team": team}


def scene_with(attacker_x, second_def_x, ball_x=80.0):
    names = ["SAĞ ceza sahası - kale çizgisi, yakın köşe", "SAĞ ceza sahası - kale çizgisi, uzak köşe",
             "SAĞ ceza sahası - saha içi, yakın köşe", "SAĞ ceza sahası - saha içi, uzak köşe",
             "ORTA çizgi - yan çizgi, yakın"]
    calib = [{"name": n, "pitch": list(KEYPOINTS[n]), "px": proj(KEYPOINTS[n]).tolist()} for n in names]
    dets = [
        make_det(0, (attacker_x, 30), team=0),           # hücumcu
        make_det(1, (75, 40), team=0),                   # diğer hücumcu
        make_det(2, (second_def_x, 20), team=1),         # sondan ikinci savunmacı
        make_det(3, (second_def_x - 4, 45), team=1),     # daha geride olmayan savunmacı
        make_det(4, (103, 34), role="goalkeeper"),       # savunan kaleci
        make_det(5, (90, 10), role="referee"),
    ]
    return {"calib": calib, "detections": dets, "ball_px": proj((ball_x, 34)).tolist(),
            "attack_right": True, "attacking_team": 0, "attacker_id": 0, "band": 0.3}


def test_homography_roundtrip():
    names = list(KEYPOINTS)[:20]
    pitch = [KEYPOINTS[n] for n in names if 70 <= KEYPOINTS[n][0]]
    img = [proj(p) for p in pitch]
    H, _ = compute_homography(img, pitch)
    back = image_to_pitch(img, H)
    assert np.allclose(back, pitch, atol=1e-3), back
    assert np.allclose(pitch_to_image(pitch, H), img, atol=1e-2)
    err, _ = reprojection_error(img, pitch, H)
    assert err < 0.05
    print("  homography gidiş-dönüş: OK (yeniden izdüşüm hatası %.4f px)" % err)


def test_real_image_sign():
    """Gerçek görüntü (testgörsel2.jpg) kalibrasyonu. OpenCV burada H'yi negatif işaretli
    döndürüyor; işaret düzeltilmezse tüm noktalar 'kameranın arkasında' sayılıp NaN oluyordu."""
    px = [(196.0, 715.3), (1394.6, 444.9), (794.5, 454.6), (1129.5, 353.5), (883.9, 393.1), (655.3, 430.0)]
    names = ["SOL ceza sahası - saha içi, yakın köşe", "SOL ceza sahası - saha içi, uzak köşe",
             "SOL altıpas - saha içi, uzak köşe", "SOL köşe bayrağı (uzak)",
             "SOL ceza sahası - kale çizgisi, uzak köşe", "SOL altıpas - kale çizgisi, uzak köşe"]
    pitch = [KEYPOINTS[n] for n in names]
    H, _ = compute_homography(px, pitch)
    proj = pitch_to_image(pitch, H)
    assert not np.isnan(proj).any(), "görünen noktalar NaN olmamalı"
    err, _ = reprojection_error(px, pitch, H)
    assert err < 5, err
    print("  gerçek görüntü kalibrasyonu (işaret düzeltmesi): OK (hata %.2f px)" % err)


def test_engine_rules():
    # Sağa hücum: savunmacılar 98 (kaleci 103), top 80
    assert evaluate(95, [103, 90], 80, True, 0.3).decision == OFFSIDE
    assert evaluate(85, [103, 90], 80, True, 0.3).decision == ONSIDE
    assert evaluate(90.1, [103, 90], 80, True, 0.3).decision == BORDERLINE
    # Toptan geride -> ofsayt değil (top çizgiyi belirler)
    r = evaluate(93, [103, 90], 95, True, 0.3)
    assert r.decision == ONSIDE and r.line_source == "top"
    # Kendi yarı sahasında -> ofsayt değil
    assert evaluate(50, [60, 55], 40, True, 0.3).decision == ONSIDE
    # Sola hücum (ayna simetri): x = 105 - x
    assert evaluate(105 - 95, [2, 15], 25, False, 0.3).decision == OFFSIDE
    assert evaluate(105 - 85, [2, 15], 25, False, 0.3).decision == ONSIDE
    # Tek savunmacı -> uyarı
    assert evaluate(95, [103], 80, True).warnings
    print("  karar kuralları: OK")


def test_full_pipeline():
    for att, sec, expected in [(96.0, 92.0, OFFSIDE), (90.0, 92.0, ONSIDE), (92.1, 92.0, BORDERLINE)]:
        res, H, info = decide(scene_with(att, sec))
        assert res.decision == expected, (att, sec, res)
        assert abs(res.diff_m - (att - sec)) < 0.01, res.diff_m
        assert 4 in info["defender_ids"] and 5 not in info["defender_ids"]   # kaleci dahil, hakem hariç
    print("  uçtan uca sahne (kalibrasyon + motor): OK")


def test_offpitch_people_ignored():
    """Kale arkasındaki top toplayıcı (savunan takım renginde) ve çizgi dışındaki yan hakem
    savunmacı sayılmamalı. Sayılırsa ofsayt çizgisi kalecinin hizasına kayar (testgörseli.jpg hatası)."""
    sc = scene_with(96.0, 92.0)
    sc["detections"] += [make_det(20, (108.5, 40), team=1),     # kale arkasında, savunan takım rengi
                         make_det(21, (80.0, -2.5), team=1)]    # yan çizginin dışında (yan hakem)
    res, _, info = decide(sc)
    assert res.decision == OFFSIDE and abs(res.diff_m - 4.0) < 0.01, res
    assert {20, 21} <= set(info["excluded_ids"]) and 20 not in info["defender_ids"]
    res2, _, _ = decide({**sc, "exclude_offpitch": False})
    assert res2.decision == ONSIDE, "kontrol: saha dışı dahil edilince çizgi yanlış yere kaymalı"
    print("  saha dışındaki kişiler hesaba katılmıyor: OK")


if __name__ == "__main__":
    test_homography_roundtrip()
    test_real_image_sign()
    test_engine_rules()
    test_full_pipeline()
    test_offpitch_people_ignored()
    print("Tüm testler geçti.")
