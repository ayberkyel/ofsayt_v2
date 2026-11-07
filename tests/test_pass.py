"""
Otomatik pas anı tespiti testi (sentetik video, bilinen cevaplı).

Video: dokulu bir zemin üzerinde kamera sürekli sağa kayar (yayın kamerası gibi). Oyuncu topu
ayağında sürerken PASS_FRAME karesinde topa vurur ve top hızla uzaklaşır. Gerçek YOLO yerine
beyaz topu ve oyuncu kutusunu renkten bulan sahte bir model kullanılır; böylece test, iz takibi,
kamera hareketi telafisi ve pas skorunu ölçer.

Çalıştırma:  python tests/test_pass.py
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pass_detect import BALL_CLASS, PERSON_CLASS, detect_pass  # noqa: E402

W, H, FPS, N = 960, 540, 25, 120
PASS_FRAME = 60       # son temas karesi: top bu kareden sonraki karede ayaktan ayrılmış olur
PAN = 4.0             # kamera kayması, px/kare


FREEZE = 40           # maç özetlerindeki gibi pas anında görüntünün donduğu kare sayısı


def make_video(path, freeze=0):
    """freeze > 0 ise PASS_FRAME karesi freeze kez tekrar yazılır (yayıncının pas anında dondurması)."""
    rng = np.random.default_rng(0)
    world = cv2.GaussianBlur((rng.random((H, W * 3)) * 255).astype(np.uint8), (0, 0), 3)
    world = cv2.cvtColor(cv2.merge([world // 3, world // 2 + 60, world // 3]), cv2.COLOR_BGR2BGR555)
    world = cv2.cvtColor(world, cv2.COLOR_BGR5552BGR)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        cam = PAN * i
        img = world[:, int(cam):int(cam) + W].copy()
        px = 300 + 2.0 * i                           # oyuncu (dünya koordinatı) yavaşça koşuyor
        if i <= PASS_FRAME:
            bx, by = px + 12, 400                    # top ayağında
        else:
            bx, by = px + 12 + 18.0 * (i - PASS_FRAME), 400 - 6.0 * (i - PASS_FRAME)
        # oyuncu: kırmızı dikdörtgen, top: beyaz daire (ekran koordinatı = dünya - kamera)
        sx = int(px - cam)
        cv2.rectangle(img, (sx - 15, 300), (sx + 15, 405), (0, 0, 255), -1)
        cv2.circle(img, (int(bx - cam), int(by)), 7, (255, 255, 255), -1)
        vw.write(img)
        if i == PASS_FRAME:
            for _ in range(freeze):
                vw.write(img)
    vw.release()


class FakeModel:
    """YOLO.predict arayüzünü taklit eder: beyaz alan = top, kırmızı alan = kişi."""

    def predict(self, frame, **kw):
        boxes = []
        for mask, cls in ((cv2.inRange(frame, (230, 230, 230), (255, 255, 255)), BALL_CLASS),
                          (cv2.inRange(frame, (0, 0, 200), (60, 60, 255)), PERSON_CLASS)):
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for x, y, w, h, area in stats[1:]:
                if area < 20:
                    continue
                boxes.append(SimpleNamespace(xyxy=[np.array([x, y, x + w, y + h], float)],
                                             cls=cls, conf=0.9))
        return [SimpleNamespace(boxes=boxes)]


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "pas.mp4"
        make_video(path)
        for stride in (1, 3):
            res = detect_pass(path, FakeModel(), stride=stride)
            best = res.best
            assert best is not None, "pas bulunamadı"
            err = best.frame - PASS_FRAME
            print(f"  stride={stride}: bulunan kare {best.frame} (gerçek {PASS_FRAME}, hata {err:+d}), "
                  f"top oranı {res.ball_ratio:.0%}, oyuncuya yakın: {best.near_player}")
            assert abs(err) <= 1, f"pas karesi hatalı: {best.frame}"
            assert best.near_player
            assert best.ball_px is not None

        # Donmuş görüntü: pas karesi 40 kare (1,6 sn) tekrar ediyor; donmanın ilk karesi seçilmeli
        path = Path(d) / "donmus.mp4"
        make_video(path, freeze=FREEZE)
        for stride in (1, 2):
            res = detect_pass(path, FakeModel(), stride=stride)
            print(f"  donmuş görüntü, stride={stride}: bulunan kare {res.best.frame} (gerçek {PASS_FRAME}), "
                  f"donma {res.freeze}")
            assert res.freeze is not None, "donma bulunamadı"
            assert abs(res.best.frame - PASS_FRAME) <= stride, f"donmuş pas karesi hatalı: {res.best.frame}"
    print("test_pass: OK")


if __name__ == "__main__":
    main()
