"""
Oyuncu tespiti (YOLOv8) ve forma rengine göre takım ayrımı (KMeans) — rapor Bölüm 7 ve 13.

Modelin sınıfları: player, goalkeeper, referee.
Takımlar modelde ayrı sınıf DEĞİL; forma rengiyle otomatik ayrılır. Böylece farklı forma
renkli maçlarda modeli yeniden eğitmek gerekmez.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

ROLE_ALIASES = {
    "player": "player", "oyuncu": "player", "team_a": "player", "team_b": "player",
    "goalkeeper": "goalkeeper", "kaleci": "goalkeeper", "gk": "goalkeeper",
    "referee": "referee", "hakem": "referee",
}


@dataclass
class Detection:
    id: int
    role: str                 # player / goalkeeper / referee
    conf: float
    box: tuple                # (x1, y1, x2, y2) piksel
    team: int | None = None   # 0 = Takım A, 1 = Takım B, None = atanmadı
    manual: bool = False      # kullanıcı elle eklediyse True

    @property
    def foot(self):
        """Ayak noktası: kutunun alt orta noktası (rapor Bölüm 4'teki sadeleştirme)."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, float(y2))

    def foot_corners(self):
        """Alt kenarın sol ve sağ uçları (kaleye en yakın uç seçeneği için)."""
        x1, _, x2, y2 = self.box
        return (float(x1), float(y2)), (float(x2), float(y2))

    def to_dict(self):
        d = asdict(self)
        d["box"] = [float(v) for v in self.box]
        return d

    @staticmethod
    def from_dict(d):
        return Detection(id=d["id"], role=d["role"], conf=d.get("conf", 1.0),
                         box=tuple(d["box"]), team=d.get("team"), manual=d.get("manual", False))


class PlayerDetector:
    def __init__(self, weights_path, device=None):
        from ultralytics import YOLO   # ağır import; yalnızca gerektiğinde
        self.model = YOLO(weights_path)
        self.device = device
        self.names = self.model.names

    def detect(self, img_bgr, conf=0.25, imgsz=960):
        # agnostic_nms=True: NMS'yi sınıftan bağımsız yapar. Aksi halde YOLO aynı kişiye hem
        # "player" hem "referee" kutusu verebiliyor (test setinde 125 karede 43 kez); bu da bir
        # hakemin fazladan savunmacı sayılmasına yol açar.
        res = self.model.predict(img_bgr, conf=conf, imgsz=imgsz, device=self.device,
                                 agnostic_nms=True, verbose=False)[0]
        dets = []
        for i, b in enumerate(res.boxes):
            cls_name = str(self.names[int(b.cls)]).lower()
            role = ROLE_ALIASES.get(cls_name, cls_name)
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append(Detection(id=i, role=role, conf=float(b.conf), box=(x1, y1, x2, y2)))
        return dets


# --------------------------------------------------------------------------------------
# Takım ayrımı
# --------------------------------------------------------------------------------------
def jersey_color(img_bgr, box):
    """Oyuncunun gövde (forma) bölgesindeki çim dışı piksellerin medyan rengini (Lab) döndürür."""
    h_img, w_img = img_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    # Gövde: yüksekliğin %15-%50'si, genişliğin ortadaki %60'ı (baş, şort ve bacaklar hariç)
    cx1 = int(max(0, x1 + 0.2 * bw)); cx2 = int(min(w_img, x2 - 0.2 * bw))
    cy1 = int(max(0, y1 + 0.15 * bh)); cy2 = int(min(h_img, y1 + 0.5 * bh))
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] > 50)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    keep = lab[~grass.ravel()]
    if len(keep) < 5:
        keep = lab
    # Yalnızca renk tonu (a, b) kullanılır; parlaklık (L) atılır. Aksi halde KMeans oyuncuları
    # forma yerine GÜNEŞ / GÖLGE'ye göre ayırır (aynı forma gölgede çok daha koyu görünür).
    return np.median(keep[:, 1:], axis=0).astype(np.float32)


def assign_teams(img_bgr, detections, seed=0):
    """'player' rolündeki tespitleri forma rengine göre 2 kümeye ayırır (detections yerinde güncellenir).

    Küme 0 / 1 etiketlerinin hangi takım olduğu keyfidir; arayüzde kullanıcı gerekirse
    takım isimlerini değiştirebilir. Kaleciler kümelenmez (forma rengi farklıdır);
    takımları ofsayt motorunda yarı sahaya göre belirlenir.
    """
    players = [d for d in detections if d.role == "player"]
    feats, owners = [], []
    for d in players:
        c = jersey_color(img_bgr, d.box)
        if c is not None:
            feats.append(c); owners.append(d)
    if len(feats) < 2:
        return detections
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(np.array(feats))
    # Etiketleri kararlı yapmak için: kırmızıya daha yakın (a kanalı yüksek) küme = Takım A (0)
    order = np.argsort(-km.cluster_centers_[:, 0])
    remap = {int(order[0]): 0, int(order[1]): 1}
    for d, lab in zip(owners, km.labels_):
        d.team = remap[int(lab)]
    return detections
