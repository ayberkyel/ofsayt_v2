"""
ADIM 1 — CVAT'tan indirilen "YOLO 1.1" zip'ini Ultralytics YOLOv8 eğitim formatına çevirir.

CVAT çıktısı:                         Bu betiğin ürettiği:
  obj.names                            data/dataset/
  obj.data                               images/{train,val,test}/*.jpg
  train.txt                              labels/{train,val,test}/*.txt
  obj_train_data/*.jpg + *.txt          data/dataset/data.yaml

VERİ SIZINTISI (rapor Bölüm 7.2):
  Aynı videonun art arda kareleri birbirine çok benzer. Rastgele bölersek komşu kareler
  hem eğitime hem teste düşer ve sonuçlar olduğundan iyi görünür. Bu yüzden:
    * Dosya adları "mac1_000518.jpg" gibi "<maç>_<kare>" biçimindeyse ve en az 3 farklı
      maç varsa -> bölme MAÇ BAZINDA yapılır (bir maçın tüm kareleri aynı bölümde).
    * Daha az maç varsa -> her maç kendi içinde zamana göre sıralanır ve ARDIŞIK BLOKLARA
      bölünür (ilk %70 eğitim, sonraki %15 doğrulama, son %15 test). Bloklar arasına
      "boşluk" (gap) bırakılarak sınırdaki komşu karelerin sızması da önlenir.

Kullanım:
  python scripts/prepare_dataset.py --zip "C:/.../task_xxx_yolo 1.1.zip"
  python scripts/prepare_dataset.py --zip a.zip --zip b.zip     (birden fazla CVAT görevi)
"""
import argparse
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
NAME_RE = re.compile(r"^(?P<group>.+?)[_\-](?P<frame>\d+)$")


NOT_LABELS = {"train.txt", "valid.txt", "test.txt", "obj.names", "obj.data"}


def read_zips(zip_paths, extract_dir):
    """Tüm zip'leri açar ve görselleri etiketlerle DOSYA ADINA göre eşleştirir.

    Görseller ve etiketler aynı zip'te de olabilir (CVAT 'Save images'), ayrı zip'lerde de
    (ör. gorsel_mac1.zip + etiket_mac1.zip). Dönüş: [(görsel, etiket|None)], sınıf isimleri
    """
    names = None
    for i, zp in enumerate(zip_paths):
        with zipfile.ZipFile(zp) as z:
            z.extractall(extract_dir / f"z{i}")
        nf = next((extract_dir / f"z{i}").rglob("obj.names"), None)
        if nf:
            nm = [l.strip() for l in nf.read_text(encoding="utf-8").splitlines() if l.strip()]
            if names and nm != names:
                raise SystemExit(f"Zip'lerin sınıf listeleri farklı: {names} vs {nm} ({zp})")
            names = nm
    images, labels, dup = {}, {}, 0
    for f in sorted(extract_dir.rglob("*")):
        if f.suffix.lower() in IMG_EXT:
            if f.stem in images:      # ör. etiket zip'i de görselleri içeriyorsa
                dup += 1
                continue
            images[f.stem] = f
        elif f.suffix == ".txt" and f.name not in NOT_LABELS:
            labels[f.stem] = f
    if dup:
        print(f"[bilgi] {dup} görsel birden fazla zip'te vardı; her birinin bir kopyası kullanıldı.")
    orphan = sorted(set(labels) - set(images))
    if orphan:
        print(f"[uyarı] görseli olmayan {len(orphan)} etiket atlandı (ör. {orphan[:3]})")
    return [(img, labels.get(stem)) for stem, img in sorted(images.items())], names


def group_and_frame(stem):
    m = NAME_RE.match(stem)
    if m:
        return m.group("group"), int(m.group("frame"))
    return "grup0", 0


def split_items(items, ratios=(0.7, 0.15, 0.15), gap=2, seed=42):
    groups = defaultdict(list)
    for it in items:
        g, f = group_and_frame(it[0].stem)
        groups[g].append((f, it))
    for g in groups:
        groups[g].sort(key=lambda t: t[0])

    splits = {"train": [], "val": [], "test": []}
    if len(groups) >= 3:
        print(f"[bölme] {len(groups)} farklı maç/video bulundu -> MAÇ BAZINDA bölme yapılıyor.")
        order = sorted(groups, key=lambda g: len(groups[g]), reverse=True)
        random.Random(seed).shuffle(order)
        total = len(items)
        targets = {"val": ratios[1] * total, "test": ratios[2] * total}
        for g in order:
            n = len(groups[g])
            # Hedefe en çok ihtiyacı olan bölüme ata; doğrulama/test dolduysa eğitime
            need = {k: targets[k] - len(splits[k]) for k in ("val", "test")}
            k = max(need, key=need.get)
            dest = k if need[k] >= n / 2 else "train"
            splits[dest].extend(it for _, it in groups[g])
        for k, v in splits.items():
            print(f"   {k}: {sorted({group_and_frame(i[0].stem)[0] for i in v})}")
    else:
        print(f"[bölme] Yalnızca {len(groups)} maç/video var -> her video ZAMANA GÖRE ardışık bloklara "
              f"bölünüyor (blok arası {gap} kare atlanıyor).")
        print("   UYARI: Raporda bunu belirtin; mümkünse farklı maçlardan veri ekleyin.")
        for g, lst in groups.items():
            n = len(lst)
            a = int(n * ratios[0]); b = int(n * (ratios[0] + ratios[1]))
            splits["train"].extend(it for _, it in lst[:a])
            splits["val"].extend(it for _, it in lst[a + gap:b])
            splits["test"].extend(it for _, it in lst[b + gap:])
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="append", default=[], help="zip dosyası (birden fazla verilebilir)")
    ap.add_argument("--dir", help="içindeki TÜM zip'ler kullanılır (ör. gorsel_mac*.zip + etiket_mac*.zip)")
    ap.add_argument("--out", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--gap", type=int, default=2, help="ardışık bloklar arasında atlanacak kare sayısı")
    args = ap.parse_args()

    zips = [Path(z) for z in args.zip] + (sorted(Path(args.dir).glob("*.zip")) if args.dir else [])
    if not zips:
        raise SystemExit("--zip veya --dir verin.")
    out = Path(args.out)
    tmp = ROOT / "data" / "_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    if out.exists():
        shutil.rmtree(out)

    print(f"[okuma] {len(zips)} zip açılıyor: {', '.join(z.name for z in zips)}")
    all_items, names = read_zips(zips, tmp)
    print(f"[okuma] {len(all_items)} görsel, {sum(1 for _, l in all_items if l)} tanesinin etiketi var")
    if not all_items:
        raise SystemExit("Zip içinde görsel bulunamadı! CVAT'ta export ederken 'Save images' "
                         "kutusunu işaretlediğinizden emin olun.")
    names = names or ["player", "goalkeeper", "referee"]

    no_label = sum(1 for _, l in all_items if l is None)
    if no_label:
        print(f"[uyarı] {no_label} görselin etiket dosyası yok -> boş (arka plan) görsel olarak eklenecek.")

    splits = split_items(all_items, gap=args.gap)
    stats = {}
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts = defaultdict(int)
        for img, lab in items:
            shutil.copy2(img, out / "images" / split / img.name)
            dst = out / "labels" / split / (img.stem + ".txt")
            text = lab.read_text(encoding="utf-8") if lab else ""
            dst.write_text(text, encoding="utf-8")
            for line in text.splitlines():
                if line.strip():
                    counts[names[int(line.split()[0])]] += 1
        stats[split] = (len(items), dict(counts))

    yaml = (f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    shutil.rmtree(tmp)

    print("\n=== VERİ SETİ ÖZETİ (rapora koyabilirsiniz) ===")
    print(f"{'bölüm':<7}{'görsel':>8}  " + "  ".join(f"{n:>11}" for n in names))
    for split, (n, c) in stats.items():
        print(f"{split:<7}{n:>8}  " + "  ".join(f"{c.get(nm, 0):>11}" for nm in names))
    print(f"\ndata.yaml -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
