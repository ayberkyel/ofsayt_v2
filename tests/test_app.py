"""
Arayüzün tüm adımlarının hatasız çalıştığını kontrol eden duman (smoke) testi.
Tıklamalar sahte değerlerle session_state'e yazılır; gerçek model ve gerçek görsel kullanılır.

Çalıştırma:  python tests/test_app.py [görsel_yolu]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
IMG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "testgörsel2.jpg"


def main():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    frame = cv2.imdecode(np.fromfile(str(IMG), np.uint8), cv2.IMREAD_COLOR)
    at.session_state["frame"] = frame
    at.session_state["frame_name"] = IMG.stem
    at.session_state["ball"] = (frame.shape[1] * 0.5, frame.shape[0] * 0.6)

    # Adım 2: top (büyüteç varsayılan açık -> önce bölge seçimi istenir)
    at.sidebar.radio[0].set_value("2 · Top konumu").run()
    assert not at.exception, at.exception
    assert any("bölgeye tıklayın" in i.value for i in at.info), "büyüteç bölge seçimi beklenirdi"
    at.toggle(key="ball_click_zoom_on").set_value(False).run()
    assert not at.exception, at.exception
    print("  top sayfası (büyüteç açık/kapalı): OK")

    # Adım 3: tespit
    at.sidebar.radio[0].set_value("3 · Oyuncu tespiti").run()
    assert not at.exception, at.exception
    at.button(key=None)  # noqa
    [b for b in at.button if "Tespit et" in b.label][0].click().run()
    assert not at.exception, at.exception
    dets = at.session_state["dets"]
    print(f"  tespit: {len(dets)} kişi, takımlar: "
          f"{sum(d.team == 0 for d in dets)} A / {sum(d.team == 1 for d in dets)} B")

    # Adım 4: kalibrasyon — testgörsel2.jpg üzerinde çizgi kesişimlerinden ölçülmüş GERÇEK noktalar
    from src.pitch import KEYPOINTS
    good = {"SOL ceza sahası - saha içi, yakın köşe": (196.0, 715.3),
            "SOL ceza sahası - saha içi, uzak köşe": (1394.6, 444.9),
            "SOL ceza sahası - kale çizgisi, uzak köşe": (883.9, 393.1),
            "SOL altıpas - kale çizgisi, uzak köşe": (655.3, 430.0),
            "SOL ceza yayı - ceza sahası kesişimi, yakın": (740.7, 592.4),
            "SOL ceza yayı - ceza sahası kesişimi, uzak": (1132.6, 504.0)}
    at.session_state["calib"] = [{"name": n, "px": list(p), "pitch": list(KEYPOINTS[n])} for n, p in good.items()]
    at.sidebar.radio[0].set_value("4 · Kalibrasyon").run()
    assert not at.exception, at.exception
    assert any("kalenin hangi tarafta" in i.value for i in at.info), "önce kale tarafı sorulmalı"
    at.radio(key="w_calib_side").set_value("SOL").run()
    assert not at.exception, at.exception
    assert at.selectbox(key="calib_name").value not in good, "kullanılmış nokta tekrar önerilmemeli"
    assert not [e for e in at.error if "tutarsız" in e.value], "doğru kalibrasyonda uyarı çıkmamalı"
    print("  kalibrasyon sayfası (doğru isimler): OK", [m.value for m in at.metric])

    # Kullanıcının yaşadığı hata: ceza sahası köşelerine tıklanıp isim olarak köşe bayrakları kalmış
    wrong = ["SOL köşe bayrağı (yakın)", "SAĞ köşe bayrağı (yakın)", "SOL köşe bayrağı (uzak)", "SAĞ köşe bayrağı (uzak)"]
    pts = list(good.values())[:4]
    at.session_state["calib"] = [{"name": n, "px": list(p), "pitch": list(KEYPOINTS[n])} for n, p in zip(wrong, pts)]
    at.run()
    assert [e for e in at.error if "tutarsız" in e.value], "yanlış isimlerde tutarsızlık uyarısı çıkmalı"
    print("  yanlış isimli kalibrasyon uyarısı: OK")

    # Kullanıcının ikinci denemesi: 4 noktanın 3'ü ceza sahası ön çizgisinde (aynı doğru)
    coll = ["SOL ceza sahası - saha içi, yakın köşe", "SOL ceza sahası - saha içi, uzak köşe",
            "SOL ceza sahası - kale çizgisi, uzak köşe", "SOL ceza yayı - ceza sahası kesişimi, yakın"]
    at.session_state["calib"] = [{"name": n, "px": list(good[n]), "pitch": list(KEYPOINTS[n])} for n in coll]
    at.run()
    assert not at.exception, at.exception
    assert [e for e in at.error if "AYNI DOĞRU" in e.value], "doğrusal noktalar uyarısı çıkmalı"
    print("  doğrusal nokta uyarısı: OK")
    at.session_state["calib"] = [{"name": n, "px": list(p), "pitch": list(KEYPOINTS[n])} for n, p in good.items()]

    # Adım 5: sonuç
    at.sidebar.radio[0].set_value("5 · Sonuç").run()
    assert not at.exception, at.exception
    print("  sonuç sayfası:", [m.value for m in at.metric], [md.value[:60] for md in at.markdown][:2])
    for st_err in at.error:
        print("  HATA:", st_err.value)
    print("Arayüz duman testi geçti.")


if __name__ == "__main__":
    main()
