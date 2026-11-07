"""
Otomatik pas anı tespiti.

Ofsayt kararı, pasın verildiği İLK karede (topun ayaktan ayrıldığı an) verilir. Bu modül o kareyi
videodan otomatik bulur; kullanıcı yine de -1 / +1 düğmeleriyle düzeltebilir.

Yöntem:
  1) Top tespiti    : COCO ile eğitilmiş YOLOv8'in "sports ball" (32) sınıfı her karede çalıştırılır.
                      Aynı geçişte "person" (0) kutuları da alınır. Kendi modelimizde top sınıfı yoktur.
                      Çevresi çim olmayan adaylar (tribündeki ışık, reklam panosu, seyirci...) elenir.
  2) Kamera hareketi: Yayın kamerası sürekli döner; topun EKRANDAKİ hızı kamera hareketini de içerir.
                      Ardışık kareler arasındaki genel hareket (afin dönüşüm) arka plan köşe noktalarının
                      optik akışından kestirilir ve topun hızından çıkarılır.
  3) Top izi        : Adaylar kareler boyunca "iz parçalarına" bağlanır. Her parça; güven, çim oranı ve
                      sahadaki gerçek yer değiştirmesine göre puanlanır. Bütün video boyunca kıpırdamayan
                      bir nesne (saha kenarındaki yedek top gibi) oyun topu sayılmaz.
  4) Vuruş skoru    : Topun hız vektöründe ani değişim + sonrasında hızlı top + oyuncunun ayağına yakınlık.
                      Skorun tepe yaptığı bölgede topun hızlandığı İLK kare vuruş anıdır.
  5) Pas seçimi     : Ofsayt klibinde genellikle önce pas, sonra şut vardır. Birden çok güçlü vuruş
                      bulunursa SON vuruş şut kabul edilir ve ondan önceki vuruş pas olarak seçilir.
                      Bütün vuruşlar kullanıcıya listelenir.
  6) Donmuş görüntü : Maç özetlerinde yayıncı, ofsayt pozisyonunu göstermek için görüntüyü pas anında
                      1-2 saniye dondurur. Bu karelerde top hareketsiz göründüğü için hız hesabı bozulur;
                      donmuş kareler hesaptan çıkarılır ve donmanın başladığı kare pas adayı olarak seçilir.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

BALL_CLASS = 32      # COCO: sports ball
PERSON_CLASS = 0     # COCO: person
MIN_GREEN = 0.25     # top adayının çevresindeki en az çim oranı
FREEZE_DIFF = 0.6    # ardışık kareler arası ortalama gri fark bundan küçükse görüntü donmuş sayılır (640 px)
MIN_FREEZE_S = 0.3   # bundan kısa donmalar yok sayılır


@dataclass
class PassCandidate:
    frame: int
    time: float
    score: float          # 0..1, en güçlü vuruşa göre normalize
    near_player: bool     # vuruş anında top bir oyuncunun ayağına yakın mı
    ball_px: tuple | None # o karedeki top konumu (zemin teması: kutunun alt orta noktası)


@dataclass
class PassResult:
    candidates: list                 # seçilen pas ilk sırada, diğer vuruşlar zaman sırasıyla
    fps: float
    n_frames: int
    ball_ratio: float                # topun izlendiği karelerin oranı
    frames: np.ndarray = field(repr=False)   # analiz edilen kare indeksleri
    speed: np.ndarray = field(repr=False)    # kamera hareketi çıkarılmış top hızı (px/kare @1280)
    score: np.ndarray = field(repr=False)    # kare başına vuruş skoru
    ball: np.ndarray = field(repr=False)     # (N, 2) top merkezi, bilinmiyorsa nan
    reason: str = ""                          # seçimin açıklaması (arayüzde gösterilir)
    freeze: tuple | None = None               # yayında görüntünün donduğu (ilk, son) kare, yoksa None

    @property
    def best(self):
        return self.candidates[0] if self.candidates else None


# --------------------------------------------------------------------------------------
# 1) Tespit ve kamera hareketi
# --------------------------------------------------------------------------------------
def _camera_motion(prev_gray, gray):
    """prev -> cur genel hareketini 2x3 afin matris olarak döndürür (bulunamazsa birim)."""
    eye = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    if prev_gray is None:
        return eye
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 10:
        return eye
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    good = st.ravel() == 1
    if good.sum() < 10:
        return eye
    A, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return A if A is not None else eye


def green_ratio(hsv, cx, cy, w, h):
    """Adayın çevresindeki (kutunun ~4 katı) çim piksellerinin oranı. Gerçek top neredeyse hep çim üzerindedir."""
    H, W = hsv.shape[:2]
    r = max(w, h, 6)
    x1, x2 = int(max(0, cx - 2 * r)), int(min(W, cx + 2 * r))
    y1, y2 = int(max(0, cy - 2 * r)), int(min(H, cy + 2 * r))
    p = hsv[y1:y2, x1:x2]
    if p.size == 0:
        return 0.0
    g = (p[..., 0] >= 30) & (p[..., 0] <= 90) & (p[..., 1] > 40) & (p[..., 2] > 40)
    return float(g.mean())


def scan_video(video_path, model, stride=1, imgsz=960, conf=0.05, progress=None, frame_range=None):
    """Videoyu tarar. Dönüş: dict(frames, balls, people, motion, fps, n_frames, size).

    balls[i]  : o karedeki top adayları [(cx, cy, w, h, conf, çim_oranı), ...]
    people[i] : kişi kutuları [(x1, y1, x2, y2), ...]
    motion[i] : frames[i-1] -> frames[i] kamera hareketi (2x3, analiz edilen kareler arası)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Video açılamadı: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    lo, hi = frame_range or (0, n - 1)
    lo, hi = max(0, int(lo)), min(n - 1, int(hi))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)

    out = dict(frames=[], balls=[], people=[], motion=[], diff=[], fps=fps, n_frames=n, size=None)
    prev_gray, small_w = None, 640
    idx = lo
    total = (hi - lo) // stride + 1
    while idx <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - lo) % stride == 0:
            h, w = frame.shape[:2]
            out["size"] = (w, h)
            sc = small_w / w
            gray = cv2.cvtColor(cv2.resize(frame, (small_w, int(h * sc))), cv2.COLOR_BGR2GRAY)
            A = _camera_motion(prev_gray, gray)
            A = A.copy(); A[:, 2] /= sc          # ötelemeyi tam çözünürlüğe çevir
            diff = np.inf if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean())
            prev_gray = gray

            res = model.predict(frame, imgsz=imgsz, conf=conf, classes=[PERSON_CLASS, BALL_CLASS],
                                verbose=False)[0]
            hsv = None
            balls, people = [], []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                if int(b.cls) == BALL_CLASS:
                    if hsv is None:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    balls.append((cx, cy, bw, bh, float(b.conf), green_ratio(hsv, cx, cy, bw, bh)))
                elif float(b.conf) >= 0.25:
                    people.append((x1, y1, x2, y2))
            out["frames"].append(idx); out["balls"].append(balls)
            out["people"].append(people); out["motion"].append(A); out["diff"].append(diff)
            if progress:
                progress(len(out["frames"]) / max(total, 1))
        idx += 1
    cap.release()
    if not out["frames"]:
        raise IOError("Videodan kare okunamadı.")
    return out


def find_freezes(scan):
    """Donmuş görüntü aralıkları: [(ilk_i, son_i), ...] (scan indeksleri). ilk_i, donmuş görüntünün
    kendisidir (öncesinden farklı son kare); sonraki kareler onun kopyasıdır."""
    diff = np.asarray(scan.get("diff", []), float)
    if len(diff) == 0:
        return []
    frames = np.asarray(scan["frames"])
    stride = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
    min_len = max(2, int(round(MIN_FREEZE_S * scan["fps"] / stride)))
    still = diff < FREEZE_DIFF
    out, i = [], 0
    while i < len(still):
        if still[i]:
            j = i
            while j + 1 < len(still) and still[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((max(0, i - 1), j))
            i = j + 1
        else:
            i += 1
    return out


def drop_frozen(scan, freezes):
    """Donmuş kopyaları taramadan çıkarır (donmuş görüntünün ilk karesi kalır). Kopyalarda kamera hareketi
    birimdir; çıkarınca donma öncesi ve sonrası kareler hız hesabında ardışık olur."""
    if not freezes:
        return scan
    keep = np.ones(len(scan["frames"]), bool)
    for a, b in freezes:
        keep[a + 1:b + 1] = False
    out = dict(scan)
    for key in ("frames", "balls", "people", "motion", "diff"):
        if key in scan:
            out[key] = [x for x, k in zip(scan[key], keep) if k]
    return out


# --------------------------------------------------------------------------------------
# 2) Top izi: iz parçaları
# --------------------------------------------------------------------------------------
def _move(A, p):
    return A[:, :2] @ p + A[:, 2]


def build_tracklets(balls, motion, size, max_gap=8, step=1):
    """Adayları iz parçalarına bağlar. Konumlar her karede kamera hareketiyle taşınır; böylece hız,
    kameradan bağımsız (sahadaki) harekettir. Dönüş: [dict(idx, box, conf, green, spread), ...]"""
    w = size[0]
    done, active = [], []
    for i, cands in enumerate(balls):
        A = motion[i]
        for t in active:                       # kamera hareketi + sabit hız tahmini
            t["q"] = _move(A, t["q"]); t["gap"] += 1
            t["world"] = t["world"]            # sahadaki konum yalnızca eşleşmede güncellenir
        cands = [c for c in cands if (len(c) < 6 or c[5] >= MIN_GREEN)]
        pairs = []
        for ti, t in enumerate(active):
            pred = t["q"] + t["vel"] * t["gap"]
            limit = 0.04 * w * step + 10 * t["gap"]
            for ci, c in enumerate(cands):
                d = float(np.hypot(c[0] - pred[0], c[1] - pred[1]))
                if d < limit:
                    pairs.append((d / limit - 0.3 * c[4], ti, ci))
        used_t, used_c = set(), set()
        for _, ti, ci in sorted(pairs):
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti); used_c.add(ci)
            t, c = active[ti], cands[ci]
            vstep = (np.array(c[:2]) - t["q"]) / t["gap"]         # kamerasız hız (px/kare)
            t["vel"] = vstep
            t["world"] = t["world"] + vstep * t["gap"]
            t["wpts"].append(t["world"].copy())
            t["q"] = np.array(c[:2], float); t["gap"] = 0
            t["idx"].append(i); t["box"].append(c[:4]); t["conf"].append(c[4])
            t["green"].append(c[5] if len(c) > 5 else 1.0)
        for ci, c in enumerate(cands):
            if ci not in used_c:
                active.append(dict(q=np.array(c[:2], float), vel=np.zeros(2), gap=0, world=np.zeros(2),
                                   wpts=[np.zeros(2)], idx=[i], box=[c[:4]], conf=[c[4]],
                                   green=[c[5] if len(c) > 5 else 1.0]))
        still = []
        for t in active:
            (done if t["gap"] > max_gap else still).append(t)
        active = still
    done += active
    for t in done:
        wp = np.array(t["wpts"])
        t["spread"] = float(np.max(np.hypot(*(wp - wp[0]).T))) * (1280.0 / w)   # px @1280
        moving = float(np.clip(t["spread"] / 120.0, 0.1, 1.0))                  # hiç kıpırdamayan = yedek top / sabit nesne
        t["score"] = float(np.sum(np.array(t["conf"]) * np.array(t["green"]))) * moving
    return done


def build_track(balls, size, motion=None, max_gap=8, step=1):
    """Her kare için tek top konumu (N, 4), iz parçası numarası (N,) ve doldurulmuş kare maskesi (N,) döndürür.

    Bir karede birden çok iz parçası varsa en yüksek puanlı olan seçilir. En iyi parçanın %25'inden
    düşük puanlı parçalar (kısa ömürlü yanlış tespitler) kullanılmaz. Parça içindeki kısa boşluklar
    doğrusal doldurulur; farklı parçalar arasındaki geçişte hız hesaplanmaz.
    """
    n = len(balls)
    if motion is None:
        motion = [np.array([[1, 0, 0], [0, 1, 0]], float)] * n
    tracks = sorted(build_tracklets(balls, motion, size, max_gap, step), key=lambda t: -t["score"])
    track = np.full((n, 4), np.nan)
    seg = np.full(n, -1)
    interp = np.zeros(n, bool)
    if not tracks:
        return track, seg, interp
    best = tracks[0]["score"]
    for k, t in enumerate(tracks):
        if t["score"] < 0.25 * best or len(t["idx"]) < 3:
            continue
        idx, box = t["idx"], np.array(t["box"], float)
        lo, hi = idx[0], idx[-1]
        if (seg[lo:hi + 1] >= 0).any():        # daha iyi bir parçanın kapsadığı kareleri ezme
            free = seg[lo:hi + 1] < 0
            if free.mean() < 0.5:
                continue
        for j, (a, b) in enumerate(zip(idx[:-1], idx[1:])):
            for f in range(a, b + 1):
                if seg[f] >= 0:
                    continue
                u = 0.0 if b == a else (f - a) / (b - a)
                track[f] = (1 - u) * box[j] + u * box[j + 1]
                seg[f] = k
                interp[f] = f not in (a, b)
        if len(idx) == 1 and seg[lo] < 0:
            track[lo] = box[0]; seg[lo] = k
    return track, seg, interp


# --------------------------------------------------------------------------------------
# 3-4) Hız ve vuruş skoru
# --------------------------------------------------------------------------------------
def ball_velocity(track, motion, seg=None):
    """Kamera hareketi çıkarılmış top hızı (N, 2). v[i] = i -> i+1 yer değiştirmesi; bilinmiyorsa nan."""
    n = len(track)
    v = np.full((n, 2), np.nan)
    for i in range(n - 1):
        if seg is not None and seg[i] != seg[i + 1]:
            continue
        p, q = track[i, :2], track[i + 1, :2]
        if np.isnan(p).any() or np.isnan(q).any():
            continue
        v[i] = q - _move(motion[i + 1], p)
    return v


def _near_player(ball_box, people, factor=0.35):
    """Top, bir kişinin ayak bölgesine (kutunun alt kısmı) yakın mı?"""
    if ball_box is None or np.isnan(ball_box).any():
        return False
    cx, cy = ball_box[0], ball_box[1] + ball_box[3] / 2
    for x1, y1, x2, y2 in people:
        bh = y2 - y1
        pad = factor * bh
        if x1 - pad <= cx <= x2 + pad and y1 + 0.5 * bh <= cy <= y2 + pad:
            return True
    return False


def pass_scores(v, k=4):
    """Kare başına vuruş skoru.

    v1, v2: önceki / sonraki k karedeki MEDYAN hız vektörü.
    skor = |v2 - v1| * |v2| / (|v2| + |v1|)  ->  hız vektöründe ani değişim VE sonrasında hızlı top.
    Top durdurulduğunda (|v2| küçük) skor düşer; yalnızca pas / şut / sektirme yüksek skor alır.
    Medyan, izin tek bir karede yanlış adaya atlamasını etkisiz bırakır.
    """
    n = len(v)
    s = np.zeros(n)
    need = max(2, k - 1)                  # pencerede en az bu kadar geçerli hız olmalı
    for t in range(k, n - k):
        before, after = v[t - k:t], v[t:t + k]
        before = before[~np.isnan(before[:, 0])]
        after = after[~np.isnan(after[:, 0])]
        if len(before) < need or len(after) < need:
            continue
        vb, va = np.median(before, axis=0), np.median(after, axis=0)
        sa, sb = np.hypot(*va), np.hypot(*vb)
        s[t] = np.hypot(*(va - vb)) * sa / (sa + sb + 1e-6)
    return s


def _kick_in_gap(t, track, seg, interp, v, motion, k):
    """Vuruş, topun görülmediği bir boşluğa denk geldiyse doğrusal doldurma vuruşu boşluğun başına çeker.
    Bunun yerine, boşluktan sonraki gerçek hızla geriye gidilerek topun hareket etmeye başladığı kare
    kestirilir: boşluk öncesi son konum (kamerayla taşınmış) ile boşluk sonrası ilk konum arasındaki
    mesafe / vuruş sonrası hız = topun boşlukta hareket ettiği kare sayısı."""
    n = len(track)
    if not (0 <= t < n):
        return t
    if interp[t]:
        b = t
        while b > 0 and interp[b]:
            b -= 1
        a = t + 1
        while a < n and interp[a]:
            a += 1
    elif t > 0 and interp[t - 1]:          # hızlanma, t'den hemen önceki boşlukta başlamış
        a, b = t, t - 1
        while b > 0 and interp[b]:
            b -= 1
    else:
        return t
    if a >= n or seg[a] != seg[b] or np.isnan(track[b, 0]) or np.isnan(track[a, 0]):
        return t
    after = [v[j] for j in range(a, min(n - 1, a + k)) if not interp[j] and not interp[j + 1] and not np.isnan(v[j, 0])]
    if not after:
        return t
    va = float(np.hypot(*np.median(after, axis=0)))
    before = [v[j] for j in range(max(0, b - k), b) if not interp[j] and not interp[j + 1] and not np.isnan(v[j, 0])]
    vb = float(np.hypot(*np.median(before, axis=0))) if before else 0.0   # top sürme hızı (boşluktan önce)
    if va - vb < 1e-6:
        return t
    p = track[b, :2].copy()
    for f in range(b + 1, a + 1):
        p = _move(motion[f], p)
    d = float(np.hypot(*(track[a, :2] - p)))
    # d = vb * (tk - b) + va * (a - tk)  ->  tk
    tk = b + (va * (a - b) - d) / (va - vb)
    return int(np.clip(round(tk), b, a))


MIN_KICK = 1.5      # bundan düşük skor (px/kare @1280) vuruş sayılmaz: titreşim / yavaş top sürme
STRONG = 0.3        # en güçlü vuruşun bu oranından güçlü olanlar "belirgin vuruş" sayılır


def find_pass(scan, k=4, top_n=4, min_sep_s=0.6):
    """scan_video çıktısından vuruşları bulur ve pası seçer."""
    raw_freezes = find_freezes(scan)
    freezes = [(int(scan["frames"][a]), int(scan["frames"][b])) for a, b in raw_freezes]
    scan = drop_frozen(scan, raw_freezes)
    frames = np.asarray(scan["frames"])
    w, h = scan["size"]
    stride = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
    track, seg, interp = build_track(scan["balls"], scan["size"], scan["motion"], max_gap=max(2, 8 // stride),
                                     step=stride)
    v = ball_velocity(track, scan["motion"], seg)
    v_norm = v * (1280.0 / w) / stride                  # çözünürlük ve adımdan bağımsız: px/kare @1280
    speed = np.hypot(v_norm[:, 0], v_norm[:, 1])
    kk = max(2, k // stride)
    score = pass_scores(v_norm, kk)
    near = np.array([_near_player(track[i], scan["people"][i]) for i in range(len(frames))])
    near_any = np.array([near[max(0, i - kk):i + 1].any() for i in range(len(frames))])
    score = score * np.where(near_any, 1.0, 0.4)        # topun yakınında oyuncu yoksa skoru azalt
    ball_ratio = float(np.mean(~np.isnan(track[:, 0])))

    # Vuruş olayları (tepe noktaları, aralarında en az min_sep_s)
    events, sep = [], max(1, int(min_sep_s * scan["fps"] / stride))
    for i in np.argsort(-score):
        if score[i] < MIN_KICK or len(events) >= top_n:
            break
        if any(abs(i - j) < sep for j, _ in events):
            continue
        # Vuruş karesi: olay penceresinde topun vuruş sonrası hızına (S) ulaştığı İLK kare. Top sürme
        # (oyuncuyla birlikte yavaş ilerleme) bu hıza ulaşmaz; böylece sürmenin başlangıcı değil,
        # asıl vuruş seçilir. rm: 3 karelik ileriye dönük medyan hız (tek karelik sıçramaya dayanıklı).
        lo_w, hi_w = max(0, i - kk), min(len(speed) - 1, i + sep)
        rm = np.array([np.nanmedian(speed[j:j + 3]) if not np.isnan(speed[j:j + 3]).all() else np.nan
                       for j in range(lo_w, hi_w + 1)])
        t = i
        if not np.isnan(rm).all():
            S = np.nanmax(rm)
            ok = (np.nan_to_num(rm) >= 0.7 * S) & (np.nan_to_num(speed[lo_w:hi_w + 1]) >= 0.7 * S)
            t = lo_w + int(np.argmax(ok)) if ok.any() else i
        t = _kick_in_gap(t, track, seg, interp, v, scan["motion"], kk)
        events.append((t, float(score[i])))

    def cand(t, s, top):
        bb = track[t]
        ball_px = None if np.isnan(bb).any() else (float(bb[0]), float(bb[1] + bb[3] / 2))
        f = int(frames[t])
        return PassCandidate(frame=f, time=f / scan["fps"], score=s / top if top > 0 else 0.0,
                             near_player=bool(near_any[t]), ball_px=ball_px)

    reason, cands = "", []
    if freezes:
        # Yayıncı görüntüyü dondurduysa: en uzun donmanın ilk karesi pas anıdır. Donmanın hemen
        # ardından (0,6 sn içinde) bulunan vuruş aynı pasın kendisidir; listede tekrar gösterilmez.
        fa, fb = max(freezes, key=lambda ab: ab[1] - ab[0])
        ti = int(np.searchsorted(frames, fa))
        top = max([s for _, s in events], default=1.0)
        win = int(0.6 * scan["fps"])
        same = [e for e in events if fa - win <= int(frames[e[0]]) <= fb + win]
        rest = [e for e in sorted(events) if e not in same]
        first = cand(ti, top, top)
        # Donmuş karede top çoğu zaman pasörün ayağında, oyuncuyla iç içedir ve model onu güvenilir
        # bulamaz (denemelerde krampon / saha kenarı nesneleri seçildi). Yanlış işaret yerine boş bırakılır;
        # top 2. adımda elle işaretlenir.
        first.ball_px = None
        cands = [first] + [cand(*e, top) for e in rest]
        reason = (f"Yayında görüntü {fa / scan['fps']:.1f}–{fb / scan['fps']:.1f}. saniyeler arasında "
                  f"donduruluyor ({(fb - fa) / scan['fps']:.1f} sn). "
                  "Maç özetlerinde ofsayt pozisyonu genellikle pas anında dondurularak gösterilir; "
                  "donmanın başladığı kare pas anı olarak seçildi.")
        return PassResult(candidates=cands, fps=scan["fps"], n_frames=scan["n_frames"], ball_ratio=ball_ratio,
                          frames=frames, speed=speed, score=score, ball=track[:, :2], reason=reason,
                          freeze=(fa, fb))
    if events:
        top = max(s for _, s in events)
        chrono = sorted(events)
        strong = [e for e in chrono if e[1] >= STRONG * top]
        if len(strong) >= 2:
            chosen = strong[-2]
            reason = (f"Videoda {len(strong)} belirgin vuruş bulundu. Son vuruş (kare {int(frames[strong[-1][0]])}) "
                      "şut / bitiriş kabul edildi; ondan önceki vuruş pas olarak seçildi.")
        else:
            chosen = strong[0]
            reason = "Videoda tek belirgin vuruş bulundu."
        cands = [cand(*chosen, top)] + [cand(*e, top) for e in chrono if e != chosen]
    return PassResult(candidates=cands, fps=scan["fps"], n_frames=scan["n_frames"], ball_ratio=ball_ratio,
                      frames=frames, speed=speed, score=score, ball=track[:, :2], reason=reason)


def detect_pass(video_path, model, stride=1, imgsz=960, conf=0.05, progress=None, refine=True):
    """Tam akış. stride > 1 ise önce seyrek tarar, sonra seçilen pasın çevresini her karede yeniden tarar."""
    p1 = (lambda f: progress(0.7 * f)) if (progress and stride > 1 and refine) else progress
    scan = scan_video(video_path, model, stride=stride, imgsz=imgsz, conf=conf, progress=p1)
    res = find_pass(scan)
    # Donmuş görüntüden seçilen kare zaten kesindir (donmuş kopyalar özdeştir); yeniden taramaya gerek yok.
    if stride > 1 and refine and res.best is not None and res.freeze is None:
        c = res.best.frame
        win = int(1.0 * scan["fps"])
        p2 = (lambda f: progress(0.7 + 0.3 * f)) if progress else None
        fine = scan_video(video_path, model, stride=1, imgsz=imgsz, conf=conf, progress=p2,
                          frame_range=(c - win, c + win))
        fres = find_pass(fine)
        if fres.candidates:
            near = min(fres.candidates, key=lambda x: abs(x.frame - c))
            if abs(near.frame - c) <= 2 * stride:
                near.score = res.best.score
                res.candidates[0] = near
    return res
