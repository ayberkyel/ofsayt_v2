"""
ADIM 5 — Duyarlılık analizi (rapor Bölüm 11.3).

Soru: "Kullanıcı kalibrasyon noktalarına birkaç piksel yanlış tıklarsa karar değişir mi?"

Yöntem (Monte Carlo):
  Her kaydedilmiş sahne için, her gürültü seviyesinde (±3, ±5, ±10 px) N kez:
    1) Kalibrasyon tıklamalarının her birine [-n, +n] aralığında rastgele piksel gürültüsü ekle
       (isteğe bağlı: --foot-noise ile oyuncu ayak noktalarına da)
    2) Homography'yi yeniden hesapla, kararı yeniden üret
  Sonra: farkın (metre) standart sapması ve kararın gürültüsüz karara göre değişme oranı.

Çıktılar (results/sensitivity/):
  sensitivity.csv, sensitivity_<sahne>.png

Kullanım:
  python scripts/sensitivity.py                       (scenes/ altındaki tüm sahneler)
  python scripts/sensitivity.py --scene scenes/x.json --trials 1000 --foot-noise 3
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import decide  # noqa: E402


def analyse(scene, levels, trials, foot_noise, rng):
    base, _, _ = decide(scene)
    px = np.array([c["px"] for c in scene["calib"]], dtype=float)
    rows, samples = [], {}
    for n in levels:
        diffs, same = [], 0
        for _ in range(trials):
            noisy = px + rng.uniform(-n, n, size=px.shape)
            try:
                r, _, _ = decide(scene, calib_px=noisy.tolist(), foot_noise=foot_noise, rng=rng)
            except Exception:  # noqa: dejenere matris
                continue
            diffs.append(r.diff_m)
            same += r.decision == base.decision
        d = np.array(diffs)
        samples[n] = d
        rows.append({
            "gürültü_px": n, "deneme": len(d), "temel_karar": base.decision,
            "temel_fark_m": round(base.diff_m, 3),
            "fark_ort_m": round(float(d.mean()), 3), "fark_std_m": round(float(d.std()), 3),
            "fark_p5_m": round(float(np.percentile(d, 5)), 3), "fark_p95_m": round(float(np.percentile(d, 95)), 3),
            "aynı_karar_%": round(100 * same / max(len(d), 1), 1),
        })
    return base, rows, samples


def plot(name, base, samples, band, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, d in samples.items():
        ax.hist(d, bins=40, alpha=0.5, label=f"±{n} px (σ={d.std():.2f} m)")
    ax.axvline(base.diff_m, color="k", lw=1.5, label=f"gürültüsüz: {base.diff_m:+.2f} m")
    ax.axvspan(-band, band, color="orange", alpha=0.15, label=f"sınırda bandı ±{band} m")
    ax.axvline(0, color="r", lw=1, ls="--")
    ax.set_xlabel("ofsayt farkı (hücumcu − çizgi) [m]"); ax.set_ylabel("deneme sayısı")
    ax.set_title(f"Duyarlılık analizi — {name}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", action="append", help="sahne JSON (varsayılan: scenes/*.json)")
    ap.add_argument("--levels", type=float, nargs="+", default=[3, 5, 10])
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--foot-noise", type=float, default=0.0, help="ayak noktalarına ±px gürültü")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = [Path(p) for p in args.scene] if args.scene else sorted((ROOT / "scenes").glob("*.json"))
    if not paths:
        raise SystemExit("Sahne bulunamadı. Önce arayüzde 'Sahneyi kaydet' ile sahne kaydedin.")
    out = ROOT / "results" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_rows = []
    for p in paths:
        scene = json.loads(p.read_text(encoding="utf-8"))
        base, rows, samples = analyse(scene, args.levels, args.trials, args.foot_noise, rng)
        plot(p.stem, base, samples, scene.get("band", 0.5), out / f"sensitivity_{p.stem}.png")
        print(f"\n{p.stem}: temel karar {base.decision} ({base.diff_m:+.2f} m)")
        for r in rows:
            print(f"  ±{r['gürültü_px']:>4} px  σ={r['fark_std_m']:.3f} m  "
                  f"[%5..%95: {r['fark_p5_m']:+.2f} .. {r['fark_p95_m']:+.2f}]  aynı karar: %{r['aynı_karar_%']}")
            all_rows.append({"sahne": p.stem, **r})

    with open(out / "sensitivity.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)

    print("\n=== ÖZET (tüm sahneler) ===")
    for n in args.levels:
        rs = [r for r in all_rows if r["gürültü_px"] == n]
        print(f"  ±{n:>4} px: ortalama σ = {np.mean([r['fark_std_m'] for r in rs]):.3f} m, "
              f"ortalama aynı karar = %{np.mean([r['aynı_karar_%'] for r in rs]):.1f}")
    print(f"\nKaydedildi -> {out}")


if __name__ == "__main__":
    main()
