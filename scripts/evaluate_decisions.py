"""
ADIM 4 — Sistem düzeyinde değerlendirme: karar doğruluğu (rapor Bölüm 11.1-11.2).

Arayüzde her test klibi için sahne kaydedip "Gerçek karar" alanını doldurun
(resmi maç kararı / tekrar görüntüsü). Bu betik tüm sahneleri yeniden hesaplar ve:
  * karışıklık matrisi (gerçek: OFSAYT/DEĞİL  x  tahmin: OFSAYT/DEĞİL/SINIRDA)
  * doğruluk (accuracy) — SINIRDA kararları hariç ve dahil (yanlış sayılarak)
  * sınırda karar oranı
  * farklı bant genişlikleri için bu değerlerin nasıl değiştiği (bant seçimi için)
üretir.

Kullanım:  python scripts/evaluate_decisions.py [--bands 0 0.25 0.5 0.75 1.0]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.offside import OFFSIDE, ONSIDE, BORDERLINE  # noqa: E402
from src.pipeline import decide  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()

    scenes = []
    for p in sorted((ROOT / "scenes").glob("*.json")):
        sc = json.loads(p.read_text(encoding="utf-8"))
        if sc.get("gt") in (OFFSIDE, ONSIDE):
            scenes.append((p.stem, sc))
    if not scenes:
        raise SystemExit("'Gerçek karar' alanı doldurulmuş sahne yok (scenes/*.json).")
    out = ROOT / "results" / "decisions"
    out.mkdir(parents=True, exist_ok=True)

    # 1) Kaydedilmiş bant ile sahne bazında sonuçlar
    rows = []
    for name, sc in scenes:
        r, _, _ = decide(sc)
        rows.append({"sahne": name, "gerçek": sc["gt"], "tahmin": r.decision,
                     "fark_m": round(r.diff_m, 3), "çizgi": r.line_source,
                     "doğru": "" if r.decision == BORDERLINE else int(r.decision == sc["gt"])})
    with open(out / "per_scene.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"{'sahne':<28}{'gerçek':<14}{'tahmin':<14}{'fark (m)':>9}")
    for r in rows:
        print(f"{r['sahne']:<28}{r['gerçek']:<14}{r['tahmin']:<14}{r['fark_m']:>+9.2f}")

    # 2) Karışıklık matrisi
    preds = [OFFSIDE, ONSIDE, BORDERLINE]
    cm = {g: {p: 0 for p in preds} for g in (OFFSIDE, ONSIDE)}
    for r in rows:
        cm[r["gerçek"]][r["tahmin"]] += 1
    print("\nKarışıklık matrisi (satır = gerçek, sütun = tahmin)")
    print(f"{'':<14}" + "".join(f"{p:>14}" for p in preds))
    for g in cm:
        print(f"{g:<14}" + "".join(f"{cm[g][p]:>14}" for p in preds))

    # 3) Bant genişliği taraması: diff sabit, sadece bant değişir
    print("\nBant   kesin karar   doğruluk(kesin)   doğruluk(sınırda=yanlış)   sınırda oranı")
    sweep = []
    for band in args.bands:
        n = len(rows); dec = correct = bord = 0
        for r in rows:
            d = r["fark_m"]
            p = OFFSIDE if d > band else ONSIDE if d < -band else BORDERLINE
            if p == BORDERLINE:
                bord += 1
            else:
                dec += 1; correct += p == r["gerçek"]
        acc_dec = correct / dec if dec else float("nan")
        sweep.append({"bant_m": band, "kesin": dec, "doğruluk_kesin": round(acc_dec, 3),
                      "doğruluk_tümü": round(correct / n, 3), "sınırda_oranı": round(bord / n, 3)})
        print(f"{band:>4.2f}   {dec:>5}/{n:<5}   {acc_dec:>14.3f}   {correct/n:>24.3f}   {bord/n:>12.3f}")
    with open(out / "band_sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)
    print(f"\nKaydedildi -> {out}")


if __name__ == "__main__":
    main()
