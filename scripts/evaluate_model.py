"""
ADIM 3 — Eğitilmiş modeli TEST bölümünde değerlendirir (rapor Bölüm 7.3 ve 11.2).

Eğitim sırasında görülen doğrulama (val) sonuçları model seçiminde kullanıldığı için
iyimserdir. Rapora konacak asıl sayılar, modelin HİÇ görmediği test bölümünden gelir.

Çıktılar (results/model_test/):
  * metrics.csv          : sınıf bazında precision, recall, mAP@0.5, mAP@0.5:0.95
  * confusion_matrix.png : karışıklık matrisi (ultralytics üretir)
  * PR_curve.png vb.

Kullanım:  python scripts/evaluate_model.py [--weights models/best.pt]
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset" / "data.yaml"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--split", default="test")
    ap.add_argument("--class-nms", action="store_true",
                    help="sınıf bazlı NMS (ultralytics varsayılanı). Verilmezse uygulamadaki gibi "
                         "sınıftan bağımsız (agnostic) NMS kullanılır.")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    out_dir = ROOT / "results"
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=4, plots=True,
                  agnostic_nms=not args.class_nms, workers=0,
                  project=str(out_dir), name=f"model_{args.split}", exist_ok=True, verbose=False)

    rows = [("sınıf", "precision", "recall", "mAP@0.5", "mAP@0.5:0.95")]
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rows.append((model.names[int(c)], p, r, ap50, ap))
    rows.append(("TÜMÜ", m.box.mp, m.box.mr, m.box.map50, m.box.map))

    csv_path = out_dir / f"model_{args.split}" / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([(r[0], *[f"{v:.3f}" for v in r[1:]]) if i else r for i, r in enumerate(rows)])

    print(f"\n=== {args.split.upper()} SONUÇLARI ===")
    print(f"{'sınıf':<12}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for r in rows[1:]:
        print(f"{r[0]:<12}{r[1]:>8.3f}{r[2]:>8.3f}{r[3]:>8.3f}{r[4]:>10.3f}")
    print(f"\nKaydedildi -> {csv_path.parent}")


if __name__ == "__main__":
    main()
