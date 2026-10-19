"""
Ofsayt Tespit Sistemi — Streamlit arayüzü (rapor Bölüm 6 ve 10).

Çalıştırma:   streamlit run app.py

Akış (sol menüdeki adımlar):
  1 Kare        : video yükle, slider ile pas anını seç (görsel de yüklenebilir)
  2 Top         : topun yere temas noktasına tıkla
  3 Tespit      : YOLOv8 + forma rengine göre takım ayrımı; gerekirse elle düzelt
  4 Kalibrasyon : sahada bilinen en az 4 noktaya tıkla -> homography
  5 Sonuç       : hücum yönü / hücumcu seç -> ofsayt çizgisi ve karar

Streamlit her etkileşimde betiği baştan çalıştırır. Bu yüzden tüm durum
(st.session_state) içinde tutulur ve model @st.cache_resource ile bir kez yüklenir.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.click_image import click_image
from src.calibration import (compute_homography, draw_template, geometry_warnings, has_general_position,
                             outside_pitch_ratio, player_width_m, position_uncertainty, reprojection_error)
from src.detector import Detection, PlayerDetector, assign_teams
from src.drawing import draw_detections, draw_point, draw_result, keypoint_guide, minimap
from src.offside import OFFSIDE, ONSIDE, BORDERLINE
from src.calibration import image_to_pitch
from src.pipeline import decide, offpitch_ids
from src.pitch import KEYPOINTS, PITCH_LENGTH, is_size_dependent, ordered_keypoints
from src.video import read_frame, video_info

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "data" / "uploads"
SCENES = ROOT / "scenes"
TEAM_NAMES = {0: "Takım A", 1: "Takım B"}
STEPS = ["1 · Kare seçimi", "2 · Top konumu", "3 · Oyuncu tespiti", "4 · Kalibrasyon", "5 · Sonuç"]

st.set_page_config(page_title="Ofsayt Tespit Sistemi", layout="wide")


# ------------------------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------------------------
def default_weights():
    for p in (ROOT / "models" / "best.pt", ROOT.parent / "models" / "best.pt"):
        if p.exists():
            return str(p)
    return str(ROOT / "models" / "best.pt")


@st.cache_resource(show_spinner="Model yükleniyor...")
def load_detector(path):
    return PlayerDetector(path)


def ss():
    return st.session_state


def init_state():
    defaults = dict(src_key=None, video_path=None, info=None, is_image=False, frame_idx=0,
                    frame=None, frame_name=None, ball=None, dets=[], editor_ver=0, calib=[],
                    clicks={}, attack_right=True, attacking_team=0, attacker_id=None)
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset_after_frame():
    s = ss()
    s.ball, s.dets, s.calib, s.attacker_id = None, [], [], None
    s.editor_ver += 1


def to_rgb(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def clickable(img_bgr, key, width):
    """Görüntüyü gösterir; YENİ bir tıklama olduysa ORİJİNAL çözünürlükteki (x, y) değerini döndürür.

    Rapor Bölüm 10'daki en sık hata: görüntü ekranda küçültülerek gösterildiğinde tıklama
    koordinatı ekran pikselidir. Bunu orijinal boyuta geri ölçeklemek gerekir.
    """
    h, w = img_bgr.shape[:2]
    width = int(min(width, w))
    # Kendi bileşenimiz (src/click_image): görsel CSS ile her zaman çerçeveye sığar ve tıklama,
    # ekranda GERÇEKTE çizilen genişlikle birlikte gelir. Görseli burada ekran boyutuna küçültmek
    # yalnızca sayfaya giden veriyi azaltır (1920 px yerine ~1000 px JPEG).
    shown = cv2.resize(img_bgr, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA) \
        if width != w else img_bgr
    val = click_image(to_rgb(shown), key=key, width=width)
    if not val:
        return None
    sig = (val.get("x"), val.get("y"), val.get("unix_time"))
    if ss().clicks.get(key) == sig:          # aynı tıklama yeniden çalıştırmada tekrar gelir
        return None
    ss().clicks[key] = sig
    scale = w / float(val.get("width") or width)
    return (val["x"] * scale, val["y"] * scale)


def apply_affine(A, pt):
    """Orijinal görüntü koordinatını, A (3x3) ile gösterilen görüntünün koordinatına çevirir."""
    return (A[0, 0] * pt[0] + A[0, 2], A[1, 1] * pt[1] + A[1, 2])


def zoomable(img_bgr, key, width, overlay=None, pan_on_click=False, default_on=False):
    """Büyüteçli, tıklanabilir görüntü. YENİ tıklamayı ORİJİNAL piksel olarak döndürür.

    Büyüteç kapalıyken: tüm kare gösterilir (clickable ile aynı).
    Büyüteç açıkken:    1) önce yakınlaştırılacak bölgeye tıklanır,
                        2) o bölge büyütülerek gösterilir ve asıl tıklama burada yapılır.
    Böylece küçük top veya uzak saha çizgisi kesişimi piksel hassasiyetinde işaretlenebilir.

    overlay(img, A): işaretleri (top, noktalar, şablon) çizen fonksiyon. A, orijinal koordinatı
    gösterilen görüntü koordinatına taşıyan 3x3 matristir. İşaretler büyütmeden SONRA çizilir;
    böylece çizgiler kalınlaşıp altındaki topu/çizgiyi örtmez.
    pan_on_click: True ise büyütülmüş görüntüye tıklamak yalnızca bölgeyi o noktaya kaydırır.
    """
    zooms = ss().setdefault("zoom_centers", {})
    h, w = img_bgr.shape[:2]
    c1, c2, c3 = st.columns([2, 3, 2])
    on = c1.toggle("🔍 Büyüteç", value=default_on, key=f"{key}_zoom_on",
                   help="Önce bölgeye tıklayın, sonra büyütülmüş görüntüde tam noktaya tıklayın.")
    factor = c2.select_slider("Yakınlaştırma", [2, 3, 4, 6, 8], value=4, key=f"{key}_zoom_f",
                              disabled=not on, format_func=lambda v: f"{v}x")

    def draw(im, A):
        return overlay(im, A) if overlay else im

    if not on:
        return clickable(draw(img_bgr.copy(), np.eye(3)), key, width)

    center = zooms.get(key)
    if center is None:
        st.info("👆 Yakınlaştırmak istediğiniz bölgeye tıklayın.")
        pt = clickable(draw(img_bgr.copy(), np.eye(3)), f"{key}_pick", width)
        if pt:
            zooms[key] = pt
            st.rerun()
        return None

    # Büyütülecek bölge (görüntü dışına taşmayacak şekilde)
    cw, ch = int(w / factor), int(h / factor)
    x0 = int(np.clip(center[0] - cw / 2, 0, w - cw))
    y0 = int(np.clip(center[1] - ch / 2, 0, h - ch))
    out_w = int(width)
    sc = out_w / cw
    big = cv2.resize(img_bgr[y0:y0 + ch, x0:x0 + cw], (out_w, int(round(ch * sc))), interpolation=cv2.INTER_CUBIC)
    A = np.array([[sc, 0, -sc * x0], [0, sc, -sc * y0], [0, 0, 1]], dtype=float)
    big = draw(big, A)

    # Konum göstergesi: küçük kare üzerinde büyütülen bölge
    thumb = cv2.resize(img_bgr, (240, int(240 * h / w)))
    t = 240 / w
    cv2.rectangle(thumb, (int(x0 * t), int(y0 * t)), (int((x0 + cw) * t), int((y0 + ch) * t)), (0, 255, 255), 2)
    c3.image(to_rgb(thumb), width=160)
    if c3.button("↩ Tüm görüntü", key=f"{key}_zoom_back"):
        zooms.pop(key, None)
        st.rerun()

    pt = clickable(big, f"{key}_z{x0}_{y0}_{factor}", out_w)
    if pt is None:
        return None
    orig = (x0 + pt[0] / sc, y0 + pt[1] / sc)
    if pan_on_click:
        zooms[key] = orig
        st.rerun()
    return orig


def build_scene():
    s = ss()
    return {
        "image": s.frame_name, "image_size": [int(s.frame.shape[1]), int(s.frame.shape[0])],
        "frame_idx": int(s.frame_idx),
        "calib": [dict(c) for c in s.calib],
        "ball_px": list(s.ball) if s.ball else None,
        "detections": [d.to_dict() for d in s.dets],
        "excluded_ids": list(s.get("excluded_ids", [])),
        "force_include_ids": list(s.get("force_include_ids", [])),
        "attack_right": bool(s.attack_right), "attacking_team": int(s.attacking_team),
        "attacker_id": s.attacker_id, "body_point": s.get("body_point_val", "center"),
        "band": float(s.get("band", 0.5)),
    }


# ------------------------------------------------------------------------------------
init_state()
s = ss()

with st.sidebar:
    st.title("⚽ Ofsayt Tespit")
    step = st.radio("Adım", STEPS, key="step")
    st.divider()
    disp_w = st.slider("Görüntü genişliği (ekran)", 600, 1600, 1000, 50,
                       help="Görselin sağ kenarı kesiliyorsa küçültün; tıklanamayan bölge kalmamalı.")
    weights = st.text_input("Model ağırlıkları", default_weights())
    st.caption("Durum")
    st.write(("✅" if s.frame is not None else "⬜") + " Kare",
             ("✅" if s.ball else "⬜") + " Top",
             ("✅" if s.dets else "⬜") + " Tespit",
             ("✅" if len(s.calib) >= 4 else "⬜") + f" Kalibrasyon ({len(s.calib)})")
    if st.button("🔄 Her şeyi sıfırla", width="stretch"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ------------------------------------------------------------------------------------
# ADIM 1 — Kare seçimi
# ------------------------------------------------------------------------------------
if step == STEPS[0]:
    st.header("1 · Video yükle ve pas anını seç")
    up = st.file_uploader("Maç videosu (10-15 sn) veya tek kare görsel",
                          type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"])
    local = st.text_input("...veya bilgisayardaki dosyanın yolu (büyük videolar için)", "")
    local = local.strip().strip('"')
    src = None
    if up is not None and s.src_key != (up.name, up.size):
        UPLOADS.mkdir(parents=True, exist_ok=True)
        src = UPLOADS / up.name
        src.write_bytes(up.getbuffer())
        key = (up.name, up.size)
    elif local and Path(local).is_file() and s.src_key != (local, 0):
        src, key = Path(local), (local, 0)
    elif local and not Path(local).is_file():
        st.error("Dosya bulunamadı.")
    if src is not None:
        path = src
        s.src_key = key
        s.is_image = path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        s.video_path = str(path)
        s.frame_idx = 0
        s.info = None if s.is_image else video_info(path)
        s.frame = None
        reset_after_frame()

    if s.video_path and s.is_image:
        img = cv2.imdecode(np.fromfile(s.video_path, np.uint8), cv2.IMREAD_COLOR)
        st.image(to_rgb(img), width=disp_w)
        if st.button("✅ Bu görseli kullan", type="primary"):
            s.frame, s.frame_name = img, Path(s.video_path).name
            reset_after_frame()
            st.success("Görsel seçildi. Sol menüden **2 · Top konumu** adımına geçin.")

    elif s.video_path:
        info = s.info
        st.caption(f"{info['frames']} kare · {info['fps']:.1f} FPS · {info['width']}x{info['height']} · "
                   f"1 kare ≈ {1000/info['fps']:.0f} ms")

        def shift(d):
            s.frame_idx = int(np.clip(s.frame_idx + d, 0, info["frames"] - 1))
        c1, c2, c3, c4, c5 = st.columns([1, 1, 6, 1, 1])
        c1.button("⏪ -10", on_click=shift, args=(-10,), width="stretch")
        c2.button("◀ -1", on_click=shift, args=(-1,), width="stretch")
        c3.slider("Kare", 0, max(info["frames"] - 1, 0), key="frame_idx", label_visibility="collapsed")
        c4.button("+1 ▶", on_click=shift, args=(1,), width="stretch")
        c5.button("+10 ⏩", on_click=shift, args=(10,), width="stretch")

        frame = read_frame(s.video_path, s.frame_idx)
        st.image(to_rgb(frame), width=disp_w,
                 caption=f"Kare {s.frame_idx}  ·  t = {s.frame_idx / info['fps']:.2f} s")
        if st.button("✅ Pas anı bu kare", type="primary"):
            s.frame, s.frame_name = frame, f"{Path(s.video_path).stem}_kare{s.frame_idx}"
            reset_after_frame()
            st.success("Kare seçildi. Sol menüden **2 · Top konumu** adımına geçin.")
        st.info("İpucu: Pasın verildiği İLK kareyi seçin (topun ayaktan ayrıldığı an). "
                "25 FPS'de bir kare 40 ms'dir ve bu sürede oyuncular 20-30 cm yer değiştirebilir.")
    else:
        st.info("Başlamak için bir video veya görsel yükleyin.")

# ------------------------------------------------------------------------------------
# Diğer adımlar bir kare gerektirir
# ------------------------------------------------------------------------------------
elif s.frame is None:
    st.warning("Önce **1 · Kare seçimi** adımında bir kare seçin.")

# ------------------------------------------------------------------------------------
# ADIM 2 — Top konumu
# ------------------------------------------------------------------------------------
elif step == STEPS[1]:
    st.header("2 · Topa tıklayın")
    st.caption("Top yerdeyse topun zemine değdiği noktaya, ayaktaysa ayağın zemine bastığı noktaya tıklayın. "
               "Homography yalnızca ZEMİN düzlemindeki noktalar için doğrudur (havadaki top yanlış konumlanır).")
    def ball_overlay(im, A):
        if s.ball:
            draw_point(im, apply_affine(A, s.ball), (255, 255, 255), 9, "top")
        return im

    pt = zoomable(s.frame, "ball_click", disp_w, overlay=ball_overlay, default_on=True)
    if pt:
        s.ball = pt
        st.rerun()
    if s.ball:
        st.success(f"Top: ({s.ball[0]:.0f}, {s.ball[1]:.0f}) px. Değiştirmek için tekrar tıklayın.")
        if st.button("Topu kaldır (top kuralını kullanma)"):
            s.ball = None
            st.rerun()

# ------------------------------------------------------------------------------------
# ADIM 3 — Oyuncu tespiti ve takım ayrımı
# ------------------------------------------------------------------------------------
elif step == STEPS[2]:
    st.header("3 · Oyuncu tespiti")
    c1, c2, c3 = st.columns([1, 1, 2])
    conf = c1.slider("Güven eşiği", 0.05, 0.9, 0.25, 0.05)
    imgsz = c2.select_slider("Model giriş boyutu", [640, 960, 1280], 960)
    if c3.button("🔍 Tespit et", type="primary", width="stretch"):
        if not Path(weights).exists():
            st.error(f"Model bulunamadı: {weights}")
        else:
            det = load_detector(weights)
            t0 = time.time()
            s.dets = assign_teams(s.frame, det.detect(s.frame, conf=conf, imgsz=imgsz))
            s.excluded_ids = []
            s.editor_ver += 1
            st.toast(f"{len(s.dets)} kişi tespit edildi ({time.time()-t0:.2f} s)")

    if s.dets:
        left, right = st.columns([3, 2])
        with left:
            add_mode = st.toggle("➕ Eksik oyuncu ekleme modu (tıklanan nokta = ayak)")
            if add_mode:
                a1, a2 = st.columns(2)
                add_role = a1.selectbox("Rol", ["player", "goalkeeper", "referee"])
                add_team = a2.selectbox("Takım", [0, 1], format_func=TEAM_NAMES.get, disabled=add_role != "player")
            view = draw_detections(s.frame, s.dets, set(s.get("excluded_ids", [])))
            if s.ball:
                draw_point(view, s.ball, (255, 255, 255), 7, "top")
            pt = clickable(view, "det_click", disp_w)
            if pt and add_mode:
                sizes = np.array([[d.box[2]-d.box[0], d.box[3]-d.box[1]] for d in s.dets if d.role == "player"]
                                 or [[25, 70]])
                bw, bh = np.median(sizes, axis=0)
                new_id = max(d.id for d in s.dets) + 1
                s.dets.append(Detection(new_id, add_role, 1.0,
                                        (pt[0]-bw/2, pt[1]-bh, pt[0]+bw/2, pt[1]),
                                        add_team if add_role == "player" else None, manual=True))
                s.editor_ver += 1
                st.rerun()
            st.caption("Renkler: mavi = Takım A, kırmızı = Takım B, sarı = kaleci (K), siyah = hakem (H). "
                       "Kutunun altındaki nokta, ofsayt için kullanılan ayak noktasıdır.")
        with right:
            st.subheader("Düzeltme tablosu")
            st.caption("Takım ataması renk kümelemesiyle (KMeans) otomatik yapıldı. Yanlışsa buradan düzeltin; "
                       "yanlış tespitleri 'Yok say' ile çıkarın.")
            excl = set(s.get("excluded_ids", []))
            df = pd.DataFrame([{"ID": d.id, "Rol": d.role,
                                "Takım": TEAM_NAMES.get(d.team, "-") if d.role == "player" else "-",
                                "Güven": round(d.conf, 2), "Yok say": d.id in excl} for d in s.dets])
            edited = st.data_editor(
                df, key=f"editor_{s.editor_ver}", hide_index=True, width="stretch", height=520,
                disabled=["ID", "Güven"],
                column_config={
                    "Rol": st.column_config.SelectboxColumn(options=["player", "goalkeeper", "referee"]),
                    "Takım": st.column_config.SelectboxColumn(options=["Takım A", "Takım B", "-"]),
                })
            inv = {v: k for k, v in TEAM_NAMES.items()}
            changed = False
            by_id = {d.id: d for d in s.dets}
            for _, row in edited.iterrows():
                d = by_id[int(row["ID"])]
                team = inv.get(row["Takım"]) if row["Rol"] == "player" else None
                if d.role != row["Rol"] or d.team != team:
                    d.role, d.team = row["Rol"], team
                    changed = True
            new_excl = [int(r["ID"]) for _, r in edited.iterrows() if r["Yok say"]]
            if set(new_excl) != excl:
                changed = True
            s.excluded_ids = new_excl
            if changed:
                st.rerun()
            n = {r: sum(1 for d in s.dets if d.role == r and d.id not in s.excluded_ids)
                 for r in ("player", "goalkeeper", "referee")}
            st.write(f"Oyuncu: **{n['player']}** · Kaleci: **{n['goalkeeper']}** · Hakem: **{n['referee']}**")
            if n["goalkeeper"] == 0:
                st.warning("Kaleci tespit edilmedi. Savunan takımın kalecisi kadrajdaysa elle ekleyin; "
                           "kadraj dışındaysa sondan ikinci savunmacı yanlış belirlenebilir.")
    else:
        st.info("**Tespit et** butonuna basın.")

# ------------------------------------------------------------------------------------
# ADIM 4 — Kalibrasyon
# ------------------------------------------------------------------------------------
elif step == STEPS[3]:
    st.header("4 · Saha kalibrasyonu")
    st.caption("Her tıklamadan ÖNCE sağdaki şemada kırmızı halkayla gösterilen noktayı görüntüde bulun. "
               "Liste her tıklamadan sonra kendiliğinden bir sonraki noktaya geçer. "
               "**yakın** = kameraya yakın (ekranda alttaki) yan çizgi tarafı, **uzak** = karşı taraf. "
               "Birbirinden uzak 5-6 nokta önerilir.")
    g1, g2 = st.columns([3, 2])
    if "w_calib_side" not in st.session_state and s.get("calib_side"):
        st.session_state["w_calib_side"] = s.calib_side
    side_kw = {} if "w_calib_side" in st.session_state else {"index": None}   # ilk seferde seçimsiz
    side = g1.radio("Görüntüdeki kale hangi tarafta?", ["SAĞ", "SOL"], horizontal=True, key="w_calib_side",
                    format_func=lambda v: {"SAĞ": "Sağda →", "SOL": "← Solda"}[v], **side_kw)
    if side is None:
        st.info("👆 Önce görüntüde kalenin hangi tarafta olduğunu seçin. Nokta listesi buna göre hazırlanır.")
        st.stop()
    s.calib_side = side

    used = {c["name"] for c in s.calib}
    skipped = s.setdefault("calib_skipped", set())
    options = [n for n in ordered_keypoints(side) if n not in used]
    suggest = [n for n in options if n not in skipped] or options
    # Otomatik ilerleme: az önce kullanılan isim listeden çıkınca bir sonraki önerilen noktaya geç
    if st.session_state.get("calib_name") not in options:
        st.session_state["calib_name"] = suggest[0] if suggest else None

    def skip_current():
        # Geri çağırma (callback) widget'lar çizilmeden önce çalışır; seçimi burada değiştirmek güvenlidir
        cur = st.session_state.get("calib_name")
        skipped.add(cur)
        rest = [n for n in options if n not in skipped]
        if rest:
            st.session_state["calib_name"] = rest[0]

    name = g1.selectbox("Şimdi tıklayacağınız nokta", options, key="calib_name",
                        format_func=lambda n: n + ("  (atlandı)" if n in skipped else ""))
    if name and is_size_dependent(name):
        g1.warning("Bu nokta sahanın boyutuna bağlıdır (şablon 105×68 m varsayar). Mümkünse ceza sahası, "
                   "altıpas veya ceza yayı noktalarını tercih edin.")
    b0, b1, b2 = g1.columns(3)
    b0.button("⏭ Görünmüyor, atla", width="stretch", on_click=skip_current,
              help="Bu nokta görüntüde yoksa bir sonraki önerilen noktaya geç.")
    if b1.button("↩ Son noktayı sil", width="stretch") and s.calib:
        s.calib.pop(); st.rerun()
    if b2.button("🗑 Temizle", width="stretch"):
        s.calib = []; skipped.clear(); st.rerun()
    g2.image(to_rgb(keypoint_guide(side, name, used)), width="stretch",
             caption="Kırmızı halka: şimdi tıklanacak nokta · yeşil: tıklananlar")

    H = None
    if len(s.calib) >= 4:
        if not has_general_position([c["pitch"] for c in s.calib]):
            st.error("⚠️ Noktaların hiçbir 4'lüsü kullanılamıyor: her 4'lüde en az 3 nokta sahada AYNI DOĞRU "
                     "üzerinde (ör. ceza sahasının iki ön köşesi ve ceza yayı kesişimleri hepsi ceza sahası "
                     "ön çizgisinde). Bu doğrunun DIŞINDA bir nokta ekleyin: kale çizgisi üzerindeki bir "
                     "köşe veya altıpas köşesi.")
        else:
            try:
                H, inl = compute_homography([c["px"] for c in s.calib], [c["pitch"] for c in s.calib])
            except ValueError as e:
                st.error(str(e))

    show_tpl = st.toggle("Saha şablonunu üzerine çiz (doğrulama)", value=True, disabled=H is None)
    def calib_overlay(im, A):
        # A: orijinal -> gösterilen görüntü. Gösterilen görüntü için homography = H · A⁻¹
        if H is not None and show_tpl:
            im = draw_template(im, H @ np.linalg.inv(A))
        for i, c in enumerate(s.calib):
            draw_point(im, apply_affine(A, c["px"]), (0, 255, 0), 8, str(i + 1))
        return im

    pt = zoomable(s.frame, "calib", disp_w, overlay=calib_overlay, default_on=True)
    if pt and name:
        s.calib.append({"name": name, "px": [float(pt[0]), float(pt[1])], "pitch": list(KEYPOINTS[name])})
        s.zoom_centers.pop("calib", None)    # sonraki nokta için yeniden bölge seçilsin
        st.rerun()

    if s.calib:
        st.dataframe(pd.DataFrame([{"#": i + 1, "Nokta": c["name"], "piksel": f"({c['px'][0]:.0f}, {c['px'][1]:.0f})",
                                    "saha (m)": f"({c['pitch'][0]:.2f}, {c['pitch'][1]:.2f})"}
                                   for i, c in enumerate(s.calib)]), hide_index=True, width="stretch")
        d1, d2 = st.columns([3, 1])
        labels = [f"#{i + 1} · {c['name']}" for i, c in enumerate(s.calib)]
        del_idx = labels.index(d1.selectbox("Silinecek nokta", labels))
        if d2.button("✖ Seçili noktayı sil", width="stretch"):
            s.calib.pop(del_idx); st.rerun()
    if H is not None:
        err, per = reprojection_error([c["px"] for c in s.calib], [c["pitch"] for c in s.calib], H)
        m1, m2, m3 = st.columns(3)
        m1.metric("Yeniden izdüşüm hatası", f"{err:.2f} px" if len(s.calib) > 4 else "— (4 nokta)")
        m2.metric("Nokta sayısı", len(s.calib))
        # Ofsayt için önemli oyuncularda (görünen kaleye en yakın 8 kişi: son savunmacılar ve en
        # ilerideki hücumcular), ±2 px tıklama hatasının yarattığı konum belirsizliği
        feet = [d.foot for d in s.dets if d.role != "referee"]
        if feet:
            fx = image_to_pitch(feet, H)[:, 0]
            order = np.argsort(fx if side == "SOL" else -fx)
            probe = [feet[i] for i in order[:8]]
        else:
            probe = [c["px"] for c in s.calib]
        unc = position_uncertainty([c["px"] for c in s.calib], [c["pitch"] for c in s.calib], probe)
        m3.metric("Konum belirsizliği (oyuncular, %95)", f"±{unc:.2f} m",
                  help="Her tıklamada ±2 piksel hata olsaydı oyuncuların saha x'i ne kadar oynardı? "
                       "Sonuç adımındaki 'sınırda bandı' en az bu kadar olmalı.")
        if unc > 0.5:
            st.warning(f"Konum belirsizliği yüksek (±{unc:.2f} m): ofsayt farkları genelde bundan küçüktür. "
                       "Noktalar küçük bir alanda toplanmış olabilir (ör. sadece altıpas). Oyuncuların "
                       "bulunduğu bölgeyi çevreleyen noktalar ekleyin (ceza sahası köşeleri).")
        if len(s.calib) == 4:
            st.info("4 noktada matris bu noktalardan tam geçer, hata hep ~0 çıkar. Doğruluğu ölçmek için 5. ve 6. noktayı ekleyin.")
        elif err > 5:
            st.warning("Hata yüksek; en büyük hatalı noktayı silip yeniden tıklayın: "
                       + ", ".join(f"#{i+1}: {e:.1f}px" for i, e in enumerate(per)))
        for w in geometry_warnings([c["px"] for c in s.calib], s.frame.shape[1], s.frame.shape[0]):
            st.warning(w)
        st.caption("Sarı çizgiler gerçek saha çizgileriyle örtüşüyor mu? Örtüşmüyorsa noktaları yeniden seçin.")
        excl_ids = set(s.get("excluded_ids", []))
        players = [d for d in s.dets if d.id not in excl_ids and d.role == "player"]
        ratio, _ = outside_pitch_ratio(H, [d.foot for d in players]) if players else (0.0, None)
        pw = player_width_m(H, [d.box for d in players])
        problems = []
        if ratio > 0.2:
            problems.append(f"oyuncuların %{ratio*100:.0f}'i sahanın DIŞINA düşüyor")
        if pw is not None and not 0.25 <= pw <= 1.6:
            problems.append(f"bir oyuncu sahada {pw:.1f} m genişliğinde görünüyor (gerçekte ~1 m)")
        if problems:
            st.error("⚠️ Kalibrasyon tutarsız: " + "; ".join(problems) + ". Büyük ihtimalle tıklanan "
                     "yerler ile seçilen nokta isimleri uyuşmuyor (ör. ceza sahası köşesine tıklanıp "
                     "'köşe bayrağı' seçilmiş) ya da kale tarafı yanlış. Tablodaki isimleri kontrol edin.")
        if s.dets:
            st.subheader("Kuşbakışı saha (homography ile hesaplanan konumlar)")
            st.image(to_rgb(minimap(s.dets, H, set(s.get("excluded_ids", [])), ball_px=s.ball)), width=min(disp_w, 900))

# ------------------------------------------------------------------------------------
# ADIM 5 — Sonuç
# ------------------------------------------------------------------------------------
elif step == STEPS[4]:
    st.header("5 · Ofsayt kararı")
    if len(s.calib) < 4 or not s.dets:
        st.warning("Önce oyuncu tespiti (adım 3) ve en az 4 noktalı kalibrasyon (adım 4) gerekli.")
        st.stop()

    # Widget'lar "w_*" anahtarlarıyla tutulur ve değerler kalıcı alanlara kopyalanır.
    # (index= parametresini her çalıştırmada değiştirmek Streamlit'te seçimi sıfırlar; ayrıca
    # başka adıma geçince widget durumu silinir, kalıcı alandan yeniden doldurulur.)
    def bound(widget_key, value):
        if widget_key not in st.session_state:
            st.session_state[widget_key] = value
        return widget_key

    c1, c2, c3, c4 = st.columns(4)
    s.attack_right = c1.radio("Hücum yönü", [True, False], key=bound("w_attack_right", s.attack_right),
                              format_func=lambda v: "Sağ kaleye (→)" if v else "Sol kaleye (←)")
    s.attacking_team = c2.radio("Hücum eden takım", [0, 1], key=bound("w_attacking_team", s.attacking_team),
                                format_func=TEAM_NAMES.get)
    # Saha dışındakiler (top toplayıcı, yan hakem, yedekler...) otomatik hesap dışı
    try:
        H0, _ = compute_homography([c["px"] for c in s.calib], [c["pitch"] for c in s.calib])
    except ValueError as e:
        st.error(str(e)); st.stop()
    offp = offpitch_ids(s.dets, H0)
    if offp:
        st.info(f"Saha dışında oldukları için hesaba katılmayanlar: **{', '.join(map(str, sorted(offp)))}** "
                "(top toplayıcı, yan hakem, yedek vb.). Görüntüde bu ID'leri kontrol edin.")
        s.force_include_ids = st.multiselect("Yanlışlıkla çıkarılan varsa yine de dahil et", sorted(offp),
                                             key=bound("w_force_include", list(s.get("force_include_ids", []))))
    excl = set(s.get("excluded_ids", [])) | (offp - set(s.get("force_include_ids", [])))

    def adv(d):   # ilerilik: büyük = savunulan kaleye daha yakın
        x = image_to_pitch([d.foot], H0)[0, 0]
        return x if s.attack_right else PITCH_LENGTH - x
    att = sorted([d for d in s.dets if d.role == "player" and d.team == s.attacking_team and d.id not in excl],
                 key=adv, reverse=True)
    cands = [d.id for d in att]
    if not cands:
        st.error("Hücum eden takımda oyuncu yok."); st.stop()
    if s.attacker_id not in cands:
        s.attacker_id = cands[0]               # varsayılan: en ilerideki hücumcu
    if st.session_state.get("w_attacker_id") not in cands:
        st.session_state["w_attacker_id"] = s.attacker_id
    s.attacker_id = c3.selectbox("Kontrol edilecek hücumcu (ID)", cands, key="w_attacker_id",
                                 format_func=lambda i: f"{i}" + ("  (en ileride)" if i == cands[0] else ""))
    s.band = c4.slider("Sınırda bandı (± m)", 0.0, 1.5, step=0.05, key=bound("w_band", float(s.get("band", 0.5))))
    s.body_point_val = st.radio(
        "Vücut noktası", ["center", "nearest"], horizontal=True,
        key=bound("w_body_point", s.get("body_point_val", "center")),
        format_func=lambda v: {"center": "Kutunun alt-orta noktası (varsayılan)",
                               "nearest": "Alt kenarın kaleye en yakın ucu (kötümser)"}[v])

    scene = build_scene()
    try:
        res, H, info = decide(scene)
    except Exception as e:  # noqa
        st.error(f"Hesaplanamadı: {e}"); st.stop()

    color = {OFFSIDE: "red", ONSIDE: "green", BORDERLINE: "orange"}[res.decision]
    st.markdown(f"<h1 style='color:{color};margin:0'>{res.decision}</h1>", unsafe_allow_html=True)
    st.write(res.reason)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fark (hücumcu − çizgi)", f"{res.diff_m:+.2f} m")
    m2.metric("Ofsayt çizgisi x", f"{res.line_x:.2f} m", res.line_source, delta_color="off")
    m3.metric("Hücumcu x", f"{res.attacker_x:.2f} m")
    m4.metric("Top x", f"{res.ball_x:.2f} m" if res.ball_x is not None else "—")
    for w in res.warnings:
        st.warning(w)
    bad = [f"{n} x = {v:.1f} m" for n, v in (("hücumcu", res.attacker_x), ("top", res.ball_x))
           if v is not None and not -5 <= v <= PITCH_LENGTH + 5]
    if bad:
        st.error("⚠️ Bu karar GEÇERSİZ: " + ", ".join(bad) + " — saha 0-105 m arasındadır. "
                 "Kalibrasyon hatalı; 4. adımda nokta isimlerini ve kale tarafını kontrol edin.")

    out = draw_result(s.frame, s.dets, H, res, s.attacker_id, info["defender_ids"], s.ball, excl)
    st.image(to_rgb(out), width=disp_w,
             caption="Kırmızı çizgi: ofsayt çizgisi · Mavi çizgi: hücumcu · Kırmızı halka: savunmacılar. "
                     "Çizgiler perspektif nedeniyle eğik görünür; bu doğru geometridir.")
    st.image(to_rgb(minimap(s.dets, H, excl, res, s.ball, s.attacker_id)), width=min(disp_w, 900))
    st.info("Not: Kendi yarı sahasında olma, kale vuruşu / taç / köşe vuruşundan gelen top ve oyuna "
            "müdahale şartı bu sürümde otomatik kontrol edilmez (IFAB Kural 11). Kararı buna göre yorumlayın.")

    with st.expander("📋 Tüm hücumcular için karar", expanded=True):
        rows = []
        for i in cands:
            r_i, _, _ = decide({**scene, "attacker_id": i})
            rows.append({"ID": i, "Karar": r_i.decision, "Fark (m)": round(r_i.diff_m, 2),
                         "Hücumcu x (m)": round(r_i.attacker_x, 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Ofsayt çizgisi tüm hücumcular için aynıdır: x = {res.line_x:.2f} m ({res.line_source}). "
                   "Savunmacılar: " + ", ".join(map(str, info["defender_ids"])))

    ok, buf = cv2.imencode(".png", out)
    st.download_button("⬇ Sonuç görselini indir", buf.tobytes(), f"{s.frame_name}_sonuc.png", "image/png")

    with st.expander("💾 Sahneyi kaydet (değerlendirme ve duyarlılık analizi için)", expanded=False):
        g1, g2 = st.columns(2)
        gt = g1.selectbox("Gerçek karar (maçtaki resmi karar)", ["—", OFFSIDE, ONSIDE])
        scene_name = g2.text_input("Sahne adı", s.frame_name)
        if st.button("Kaydet"):
            SCENES.mkdir(exist_ok=True)
            scene["gt"] = None if gt == "—" else gt
            scene["image"] = f"{scene_name}.jpg"
            scene["result"] = res.to_dict()
            cv2.imencode(".jpg", s.frame)[1].tofile(str(SCENES / f"{scene_name}.jpg"))
            (SCENES / f"{scene_name}.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"Kaydedildi: scenes/{scene_name}.json")
