"""
ADIM 2 — YOLOv8 eğitimi (rapor Bölüm 7.3).

Neden bu ayarlar?
  * yolov8s   : küçük ama yeterli; 4 GB VRAM'li RTX 3050 Ti'ye sığar. (yolov8m daha iyi
                olabilir ama batch'i çok küçültmek gerekir.)
  * imgsz=960 : geniş açılı yayın görüntüsünde oyuncular küçük (~20x60 px). 640'ta
                küçülünce kayboluyorlar; 960 küçük nesne başarımını belirgin artırır.
  * batch=4   : 960 çözünürlükte 4 GB VRAM için güvenli değer. "CUDA out of memory"
                alırsanız --batch 2 deneyin.
  * fliplr=0.5: futbol sahası yatayda simetrik, aynalama gerçekçi bir artırım.
  * pretrained: COCO'da "person" öğrenmiş ağırlıklardan başlamak (transfer learning)
                600 görsellik küçük veri setinde çok önemli.
  * patience  : doğrulama mAP'i 25 epoch iyileşmezse erken durdurur (aşırı öğrenme önlemi).

Kullanım:
  python scripts/train.py                       (varsayılanlar)
  python scripts/train.py --model yolov8m.pt --batch 2 --epochs 150
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset" / "data.yaml"))
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--name", default="ofsayt_yolov8s")
    ap.add_argument("--device", default="0", help="'0' = ilk GPU, 'cpu' = işlemci")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    if args.device != "cpu" and not torch.cuda.is_available():
        print("UYARI: CUDA bulunamadı, CPU ile eğitilecek (çok yavaş).")
        args.device = "cpu"
    else:
        print(f"Cihaz: {torch.cuda.get_device_name(0) if args.device != 'cpu' else 'CPU'}")

    model = YOLO(args.model)
    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        patience=args.patience, device=args.device,
        project=str(ROOT / "runs"), name=args.name, exist_ok=True,
        workers=2,              # Windows'ta yüksek değer bellek/kilitlenme sorunu çıkarabilir
        seed=42, deterministic=False,
        # Veri artırımı
        fliplr=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, scale=0.5, translate=0.1,
        mosaic=1.0, close_mosaic=10,
        plots=True,
    )

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if best.exists():
        dst = ROOT / "models" / "best.pt"
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(best, dst)
        print(f"\nEn iyi model kopyalandı -> {dst}")
        print(f"Eğitim grafikleri ve metrikler -> {best.parents[1]}")


if __name__ == "__main__":
    main()
