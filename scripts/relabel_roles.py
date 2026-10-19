"""
ADIM 1b — Yarı otomatik rol düzeltme (kaleci / hakem).

Sorun: CVAT'tan alınan son etiketlerde tüm kutular "player" (sınıf 0) olarak kaydedilmiş;
kaleci ve hakem sınıfları kaybolmuş. Kutuların KONUMLARI doğru, yalnızca SINIFLARI eksik.

Çözüm (yarı otomatik etiketleme / pseudo-labeling):
  1) Daha önce kaleci ve hakemi de öğrenmiş eski model (../models/best.pt) her görselde çalıştırılır.
  2) Her etiket kutusu, modelin tahmin kutularıyla IoU (kesişim/birleşim) ile eşleştirilir.
  3) Eşleşen tahmin "goalkeeper" veya "referee" ise ve güveni eşiğin üzerindeyse etiketin
     sınıfı değiştirilir. Kutunun konumuna DOKUNULMAZ (insan etiketinin konumu korunur).
  4) Mantık sınırı: bir karede en fazla 2 kaleci ve 4 hakem (orta hakem + yan hakemler + 4. hakem).
     Fazlası varsa en yüksek güvenliler alınır.
  5) Değişen her kutu review/ klasöründe kırpılmış olarak gösterilir: insan kontrolü için.

Orijinal etiketler labels_orig/ klasörüne yedeklenir; betik tekrar çalıştırılırsa yedekten başlar.

Kullanım:
  python scripts/relabel_roles.py [--weights ../models/best.pt] [--conf 0.4] [--iou 0.5]
"""
import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROLE_IDS = {"player": 0, "goalkeeper": 1, "referee": 2}
MAX_PER_FRAME = {1: 2, 2: 4}


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def contact_sheet(crops, title, path, cell=(80, 160), cols=16):
    if not crops:
        return
    rows = (len(crops) + cols - 1) // cols
    sheet = np.full((rows * (cell[1] + 18) + 30, cols * cell[0], 3), 255, np.uint8)
    cv2.putText(sheet, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    for k, (crop, label) in enumerate(crops):
        r, c = divmod(k, cols)
        y, x = 30 + r * (cell[1] + 18), c * cell[0]
        h, w = crop.shape[:2]
        s = min(cell[0] / w, cell[1] / h)
        im = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
        sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(sheet, label, (x + 2, y + cell[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
    cv2.imwrite(str(path), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--weights", default=str(ROOT.parent / "models" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.4, help="kaleci/hakem tahmini için en düşük güven")
    ap.add_argument("--iou", type=float, default=0.5, help="etiket-tahmin eşleşmesi için en düşük IoU")
    args = ap.parse_args()

    # İnceleme sonrası elle düzeltmeler: data/manual_fixes.csv (dosya, kutu sırası, doğru sınıf)
    fixes = {}
    fix_path = ROOT / "data" / "manual_fixes.csv"
    if fix_path.exists():
        with open(fix_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                fixes[(r["dosya"], int(r["kutu"]))] = int(r["sinif"])

    from ultralytics import YOLO
    model = YOLO(args.weights)
    name_to_id = {n: ROLE_IDS.get(str(n).lower()) for n in model.names.values()}

    ds = Path(args.dataset)
    review = ROOT / "results" / "relabel_review"
    review.mkdir(parents=True, exist_ok=True)
    log_rows, totals = [], Counter()
    crops = {1: [], 2: []}

    for split in ("train", "val", "test"):
        lab_dir, orig_dir = ds / "labels" / split, ds / "labels_orig" / split
        if not orig_dir.exists():
            shutil.copytree(lab_dir, orig_dir)          # ilk çalıştırmada yedekle
        for lab_path in sorted(orig_dir.glob("*.txt")):
            img_path = next((ds / "images" / split).glob(lab_path.stem + ".*"))
            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]
            rows = [l.split() for l in lab_path.read_text().splitlines() if l.strip()]
            if not rows:
                shutil.copy2(lab_path, lab_dir / lab_path.name); continue
            arr = np.array(rows, dtype=float)
            cls = arr[:, 0].astype(int)
            cx, cy, bw, bh = arr[:, 1] * W, arr[:, 2] * H, arr[:, 3] * W, arr[:, 4] * H
            gt = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)

            res = model.predict(img, conf=0.1, imgsz=960, verbose=False, device=0)[0]
            pb = res.boxes.xyxy.cpu().numpy()
            pc = np.array([name_to_id[model.names[int(c)]] for c in res.boxes.cls.cpu().numpy()])
            pconf = res.boxes.conf.cpu().numpy()

            iou = iou_matrix(gt, pb)
            new_cls = cls.copy()
            for role in (1, 2):
                cand = []
                for i in range(len(gt)):
                    m = (pc == role) & (iou[i] >= args.iou) & (pconf >= args.conf)
                    if m.any():
                        cand.append((float(pconf[m].max()), i))
                for conf, i in sorted(cand, reverse=True)[:MAX_PER_FRAME[role]]:
                    if new_cls[i] == 0:
                        new_cls[i] = role
                        x1, y1, x2, y2 = np.clip(gt[i], 0, [W, H, W, H]).astype(int)
                        crops[role].append((img[y1:y2, x1:x2].copy(), f"{lab_path.stem[-11:]} {conf:.2f}"))
                        log_rows.append({"split": split, "dosya": lab_path.stem, "kutu": i,
                                         "yeni_sınıf": role, "güven": round(conf, 3)})

            for i in range(len(new_cls)):
                if (lab_path.stem, i) in fixes:
                    new_cls[i] = fixes[(lab_path.stem, i)]
                    totals["elle_düzeltme"] += 1
            lines = [" ".join([str(c)] + r[1:]) for c, r in zip(new_cls, rows)]
            (lab_dir / lab_path.name).write_text("\n".join(lines) + "\n")
            for c in new_cls:
                totals[(split, int(c))] += 1

    with open(review / "degisiklikler.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["split", "dosya", "kutu", "yeni_sınıf", "güven"])
        w.writeheader(); w.writerows(log_rows)
    for role, name in ((1, "kaleci"), (2, "hakem")):
        cs = crops[role]
        for p in range(0, len(cs), 160):
            contact_sheet(cs[p:p + 160], f"{name} olarak degistirilen kutular ({p+1}-{min(p+160, len(cs))})",
                          review / f"{name}_{p // 160 + 1}.jpg")

    print("\n=== YENİ SINIF DAĞILIMI ===")
    print(f"{'bölüm':<7}{'player':>9}{'goalkeeper':>12}{'referee':>9}")
    for split in ("train", "val", "test"):
        print(f"{split:<7}" + "".join(f"{totals[(split, c)]:>{w}}" for c, w in ((0, 9), (1, 12), (2, 9))))
    print(f"\nModelin değiştirdiği kutu: {len(log_rows)}, elle düzeltme: {totals['elle_düzeltme']}"
          f"  -> kontrol için: {review}")


if __name__ == "__main__":
    main()
