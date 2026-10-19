"""
Uygulamanın kullandığı güven eşiğinde (varsayılan 0.25) karışıklık matrisi.

Neden ayrı bir betik? Ultralytics'in val() çıktısındaki confusion_matrix.png, mAP hesabı için
çok düşük güven eşiğiyle (0.001) üretilir; binlerce zayıf tahmin "background" yanlış pozitifi
olarak görünür ve modelin gerçek kullanımını yansıtmaz. Bu betik, her etiket kutusunu
IoU >= 0.5 ile en iyi tahminle eşleştirip arayüzdeki eşikte sayım yapar.

Çıktı: results/model_test/confusion_conf<eşik>.csv / .png
Kullanım: python scripts/confusion_at_conf.py [--conf 0.25] [--split test]
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--dataset", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = [model.names[i] for i in range(len(model.names))]
    K = len(names)
    cm = np.zeros((K + 1, K + 1), int)       # satır = tahmin, sütun = gerçek; son indeks = arka plan

    ds = Path(args.dataset)
    for img_path in sorted((ds / "images" / args.split).glob("*.*")):
        img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        rows = [l.split() for l in (ds / "labels" / args.split / (img_path.stem + ".txt")).read_text().splitlines() if l.strip()]
        g = np.array(rows, float).reshape(-1, 5)
        gc = g[:, 0].astype(int)
        gb = np.stack([(g[:, 1] - g[:, 3] / 2) * W, (g[:, 2] - g[:, 4] / 2) * H,
                       (g[:, 1] + g[:, 3] / 2) * W, (g[:, 2] + g[:, 4] / 2) * H], 1)
        r = model.predict(img, conf=args.conf, imgsz=960, verbose=False)[0]
        pb = r.boxes.xyxy.cpu().numpy(); pc = r.boxes.cls.cpu().numpy().astype(int)
        iou = iou_matrix(gb, pb)
        used_p = set()
        # Açgözlü eşleştirme: en yüksek IoU'dan başla
        for gi, pi in sorted(((i, j) for i in range(len(gb)) for j in range(len(pb)) if iou[i, j] >= args.iou),
                             key=lambda t: -iou[t]):
            if gi in (x[0] for x in used_p) or pi in (x[1] for x in used_p):
                continue
            used_p.add((gi, pi))
            cm[pc[pi], gc[gi]] += 1
        matched_g = {x[0] for x in used_p}; matched_p = {x[1] for x in used_p}
        for gi in range(len(gb)):
            if gi not in matched_g:
                cm[K, gc[gi]] += 1                # kaçırılan (tahmin = arka plan)
        for pi in range(len(pb)):
            if pi not in matched_p:
                cm[pc[pi], K] += 1                # yanlış alarm (gerçek = arka plan)

    labels = names + ["arka plan"]
    out = ROOT / "results" / f"model_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"confusion_conf{args.conf:.2f}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["tahmin \\ gerçek"] + labels)
        for i, l in enumerate(labels):
            w.writerow([l] + cm[i].tolist())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(0, keepdims=True), 1)
    norm[:, K] = 0
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(K + 1):
        for j in range(K + 1):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=11)
    ax.set_xticks(range(K + 1), labels, rotation=30); ax.set_yticks(range(K + 1), labels)
    ax.set_xlabel("Gerçek"); ax.set_ylabel("Tahmin")
    ax.set_title(f"Karışıklık matrisi ({args.split}, güven ≥ {args.conf}, IoU ≥ {args.iou})")
    fig.tight_layout(); fig.savefig(out / f"confusion_conf{args.conf:.2f}.png", dpi=150); plt.close(fig)

    print("tahmin / gerçek".ljust(16) +"".join(f"{l:>12}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:<16}" + "".join(f"{v:>12}" for v in cm[i]))
    print("\nSınıf bazında (bu eşikte):")
    for k, n in enumerate(names):
        tp = cm[k, k]; p = tp / max(cm[k].sum(), 1); r = tp / max(cm[:, k].sum(), 1)
        print(f"  {n:<11} precision={p:.3f}  recall={r:.3f}")
    print(f"\nKaydedildi -> {out}")


if __name__ == "__main__":
    main()
