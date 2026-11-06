"""
Otomatik pas anı tespiti.

Ofsayt kararı, pasın verildiği İLK karede (topun ayaktan ayrıldığı an) verilir. Bu modül o kareyi
videodan otomatik bulur; kullanıcı yine de -1 / +1 düğmeleriyle düzeltebilir.

Yöntem (dört adım):
  1) Top tespiti   : COCO ile eğitilmiş YOLOv8'in "sports ball" (32) sınıfı her karede çalıştırılır.
                     Aynı geçişte "person" (0) kutuları da alınır (topun bir oyuncunun ayağında olup
                     olmadığını anlamak için). Kendi modelimizde top sınıfı yoktur.
  2) Top izi       : Her karede birden çok aday olabilir (seyircideki top, beyaz ayakkabı...).
                     Bir önceki konuma en yakın aday seçilir; kısa boşluklar doğrusal doldurulur.
  3) Kamera hareketi: Yayın kamerası sürekli döner; topun EKRANDAKİ hızı kamera hareketini de içerir.
                     Ardışık kareler arasındaki genel hareket (afin dönüşüm) arka plan köşe noktalarının
                     optik akışından kestirilir ve topun hızından çıkarılır.
  4) Pas skoru     : Pas = topun hız vektöründe ani değişim + sonrasında hızlanma + topun bir oyuncunun
                     ayağına yakın olması. Top kontrolü (top yavaşlıyor) düşük skor alır.
                     Skorun tepe yaptığı bölgede, topun hızlandığı İLK kare pas anı kabul edilir.

Çıktı: en yüksek skorlu birkaç aday kare (en iyisi ilk sırada) ve grafik için kare başına hız/skor.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

BALL_CLASS = 32      # COCO: sports ball
PERSON_CLASS = 0     # COCO: person


@dataclass
class PassCandidate:
    frame: int
    time: float
    score: float          # 0..1, en iyi adaya göre normalize
    near_player: bool     # pas anında top bir oyuncunun ayağına yakın mı
    ball_px: tuple | None # o karedeki top konumu (zemin teması: kutunun alt orta noktası)


@dataclass
class PassResult:
    candidates: list                 # PassCandidate listesi, en iyisi ilk
    fps: float
    n_frames: int
    ball_ratio: float                # topun bulunduğu analiz edilen karelerin oranı
    frames: np.ndarray = field(repr=False)   # analiz edilen kare indeksleri
    speed: np.ndarray = field(repr=False)    # kamera hareketi çıkarılmış top hızı (px/kare)
    score: np.ndarray = field(repr=False)    # kare başına pas skoru
    ball: np.ndarray = field(repr=False)     # (N, 2) top merkezi, bilinmiyorsa nan

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


def scan_video(video_path, model, stride=1, imgsz=960, conf=0.05, progress=None, frame_range=None):
    """Videoyu tarar. Dönüş: dict(frames, balls, people, motion, fps, n_frames, size).

    balls[i]  : o karedeki top adayları [(cx, cy, w, h, conf), ...]
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

    out = dict(frames=[], balls=[], people=[], motion=[], fps=fps, n_frames=n, size=None)
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
            prev_gray = gray

            res = model.predict(frame, imgsz=imgsz, conf=conf, classes=[PERSON_CLASS, BALL_CLASS],
                                verbose=False)[0]
            balls, people = [], []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                if int(b.cls) == BALL_CLASS:
                    balls.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(b.conf)))
                elif float(b.conf) >= 0.25:
                    people.append((x1, y1, x2, y2))
            out["frames"].append(idx); out["balls"].append(balls)
            out["people"].append(people); out["motion"].append(A)
            if progress:
                progress(len(out["frames"]) / max(total, 1))
        idx += 1
    cap.release()
    if not out["frames"]:
        raise IOError("Videodan kare okunamadı.")
    return out


# --------------------------------------------------------------------------------------
# 2) Top izi
# --------------------------------------------------------------------------------------
def build_track(balls, size, max_gap=8):
    """Her kare için tek top konumu seçer (N, 4: cx, cy, w, h; yoksa nan) ve kısa boşlukları doldurur."""
    w, h = size
    n = len(balls)
    track = np.full((n, 4), np.nan)
    last, last_i, vel = None, None, np.zeros(2)
    jumps = set()                                     # izin yeniden başladığı kareler (boşluk doldurulmaz)
    for i, cands in enumerate(balls):
        if not cands:
            continue
        c = np.array(cands)
        if last is not None and i - last_i <= max_gap:
            gap = i - last_i
            pred = last[:2] + vel * gap               # sabit hız varsayımıyla beklenen konum
            d = np.hypot(c[:, 0] - pred[0], c[:, 1] - pred[1])
            limit = 0.04 * w * gap + 20               # beklenenden kare başına en fazla ~%4 ekran sapma
            ok = d < limit
            if ok.any():
                # Mesafe ve güveni birlikte değerlendir
                cost = d / limit - 0.5 * c[:, 4]
                j = int(np.argmin(np.where(ok, cost, np.inf)))
            else:
                j = int(np.argmax(c[:, 4]))
                if c[j, 4] < 0.3:                  # zayıf ve uzak aday: büyük olasılıkla yanlış
                    continue
                jumps.add(i)
        else:
            j = int(np.argmax(c[:, 4]))
            jumps.add(i)
        if i in jumps:
            vel = np.zeros(2)
        else:
            vel = (c[j, :2] - last[:2]) / (i - last_i)
        track[i] = c[j, :4]
        last, last_i = c[j], i

    # Kısa boşlukları doğrusal doldur
    known = np.where(~np.isnan(track[:, 0]))[0]
    for a, b in zip(known[:-1], known[1:]):
        if 1 < b - a <= max_gap and b not in jumps:
            t = np.linspace(0, 1, b - a + 1)[1:-1, None]
            track[a + 1:b] = (1 - t) * track[a] + t * track[b]
    return track


# --------------------------------------------------------------------------------------
# 3-4) Hız ve pas skoru
# --------------------------------------------------------------------------------------
def ball_velocity(track, motion):
    """Kamera hareketi çıkarılmış top hızı (N, 2). v[i] = i -> i+1 yer değiştirmesi; bilinmiyorsa nan."""
    n = len(track)
    v = np.full((n, 2), np.nan)
    for i in range(n - 1):
        p, q = track[i, :2], track[i + 1, :2]
        if np.isnan(p).any() or np.isnan(q).any():
            continue
        A = motion[i + 1]
        p_moved = A[:, :2] @ p + A[:, 2]     # kamera hareketiyle p'nin yeni kareye taşınmış hali
        v[i] = q - p_moved
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
    """Kare başına pas skoru.

    vb, va: önceki / sonraki k karedeki MEDYAN hız vektörü.
    skor = |va - vb| * |va| / (|va| + |vb|)  ->  hız vektöründe ani değişim VE sonrasında hızlı top.
    Top durdurulduğunda (|va| küçük) skor düşer; yalnızca pas / şut / sektirme yüksek skor alır.
    Ortalama yerine medyan: iz bir karede yanlış adaya atlarsa (top ekrandan çıkarken beyaz bir
    ayakkabıya geçmek gibi) tek bir dev hız oluşur; medyan bunu yok sayar.
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


def find_pass(scan, k=4, top_n=3, min_sep_s=0.6):
    """scan_video çıktısından pas adaylarını bulur."""
    frames = np.asarray(scan["frames"])
    w, h = scan["size"]
    stride = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
    track = build_track(scan["balls"], scan["size"], max_gap=max(2, 8 // stride))
    v = ball_velocity(track, scan["motion"])
    v_norm = v * (1280.0 / w) / stride                  # çözünürlük ve adımdan bağımsız: px/kare @1280
    speed = np.hypot(v_norm[:, 0], v_norm[:, 1])
    kk = max(2, k // stride)
    score = pass_scores(v_norm, kk)
    near = np.array([_near_player(track[i], scan["people"][i]) for i in range(len(frames))])
    # Topun yakınında oyuncu yoksa (ör. sektirme, yanlış iz) skoru azalt
    near_any = np.array([near[max(0, i - kk):i + 1].any() for i in range(len(frames))])
    score = score * np.where(near_any, 1.0, 0.4)
    ball_ratio = float(np.mean([len(b) > 0 for b in scan["balls"]]))

    cands, order = [], np.argsort(-score)
    sep = max(1, int(min_sep_s * scan["fps"] / stride))
    taken = []
    for i in order:
        if score[i] <= 0 or len(cands) >= top_n:
            break
        if any(abs(i - j) < sep for j in taken):
            continue
        taken.append(i)
        # Tepe noktasından geriye doğru: topun hızlanmaya başladığı ilk kare = son temas karesi
        after = np.nanmean(speed[i:i + kk]) if not np.isnan(speed[i:i + kk]).all() else 0
        t = i
        for j in range(i, max(-1, i - kk - 1), -1):
            if not np.isnan(speed[j]) and speed[j] >= 0.5 * after:
                t = j
            else:
                break
        bb = track[t]
        ball_px = None if np.isnan(bb).any() else (float(bb[0]), float(bb[1] + bb[3] / 2))
        f = int(frames[t])
        cands.append(PassCandidate(frame=f, time=f / scan["fps"], score=float(score[i]),
                                   near_player=bool(near_any[t]), ball_px=ball_px))
    if cands:
        top = cands[0].score
        for c in cands:
            c.score = c.score / top if top > 0 else 0.0
    return PassResult(candidates=cands, fps=scan["fps"], n_frames=scan["n_frames"], ball_ratio=ball_ratio,
                      frames=frames, speed=speed, score=score, ball=track[:, :2])


def detect_pass(video_path, model, stride=1, imgsz=960, conf=0.05, progress=None, refine=True):
    """Tam akış. stride > 1 ise önce seyrek tarar, sonra en iyi adayın çevresini her karede yeniden tarar."""
    p1 = (lambda f: progress(0.7 * f)) if (progress and stride > 1 and refine) else progress
    scan = scan_video(video_path, model, stride=stride, imgsz=imgsz, conf=conf, progress=p1)
    res = find_pass(scan)
    if stride > 1 and refine and res.best is not None:
        c = res.best.frame
        win = int(1.0 * scan["fps"])
        p2 = (lambda f: progress(0.7 + 0.3 * f)) if progress else None
        fine = scan_video(video_path, model, stride=1, imgsz=imgsz, conf=conf, progress=p2,
                          frame_range=(c - win, c + win))
        fres = find_pass(fine, top_n=1)
        if fres.best is not None:
            fres.best.score = 1.0
            res.candidates[0] = fres.best
    return res
