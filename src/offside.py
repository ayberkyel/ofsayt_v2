"""
Ofsayt karar motoru — rapor Bölüm 4 ve 9.

Tüm hesap SAHA koordinatında (metre) yapılır. Hücum yönünden bağımsız çalışmak için
her x değeri "ilerilik" u'ya çevrilir:

    hücum sağ kaleye (x=105) ise:  u = x
    hücum sol kaleye (x=0)   ise:  u = 105 - x

Böylece u büyüdükçe oyuncu savunan takımın kale çizgisine yaklaşır.

Kural (sadeleştirilmiş):
    Hücum oyuncusu pas anında
      1) rakip yarı sahadaysa              (u > 52.5)
      2) toptan daha ilerideyse             (u > top_u)
      3) sondan ikinci savunmacıdan daha ilerideyse
    ofsayt pozisyonundadır. Aynı hizada olmak ofsayt değildir.

    Ofsayt çizgisi = max(sondan ikinci savunmacı, top, orta çizgi)
    fark = hücumcu_u - çizgi_u   (pozitif = çizginin önünde)

Sınırda karar bandı (Bölüm 9.2):
    fark >  +band  -> OFSAYT
    fark <  -band  -> OFSAYT DEĞİL
    aksi halde      -> SINIRDA
"""
from dataclasses import dataclass, field, asdict

from .pitch import PITCH_LENGTH, HALF_X

OFFSIDE = "OFSAYT"
ONSIDE = "OFSAYT DEĞİL"
BORDERLINE = "SINIRDA"


@dataclass
class OffsideResult:
    decision: str
    diff_m: float                 # hücumcu - çizgi (metre), pozitif = ileride
    line_x: float                 # ofsayt çizgisinin saha x koordinatı (metre)
    attacker_x: float
    second_last_x: float | None
    ball_x: float | None
    line_source: str              # çizgiyi belirleyen: "savunmacı", "top" veya "orta çizgi"
    reason: str
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _u(x, attack_right):
    return x if attack_right else PITCH_LENGTH - x


def _x(u, attack_right):
    return u if attack_right else PITCH_LENGTH - u


def evaluate(attacker_x, defender_xs, ball_x, attack_right, band=0.5):
    """
    attacker_x   : hücum oyuncusunun saha x'i (metre)
    defender_xs  : savunan takımın TÜM oyuncularının x'leri (kaleci dahil)
    ball_x       : topun x'i (None ise top kuralı uygulanmaz)
    attack_right : True -> hücum x=105 kalesine doğru
    band         : sınırda karar bandı (metre)
    """
    warnings = []
    a_u = _u(attacker_x, attack_right)
    d_us = sorted((_u(x, attack_right) for x in defender_xs), reverse=True)

    if len(d_us) < 2:
        warnings.append("Kadrajda 2'den az savunmacı var (kaleci görünmüyor olabilir). "
                        "Sondan ikinci savunmacı belirlenemedi; sonuç güvenilir değil.")
    second_u = d_us[1] if len(d_us) >= 2 else (d_us[0] if d_us else None)
    ball_u = _u(ball_x, attack_right) if ball_x is not None else None

    candidates = [(HALF_X, "orta çizgi")]
    if second_u is not None:
        candidates.append((second_u, "savunmacı"))
    if ball_u is not None:
        candidates.append((ball_u, "top"))
    line_u, source = max(candidates, key=lambda c: c[0])
    diff = a_u - line_u

    if a_u <= HALF_X:
        decision, reason = ONSIDE, "Hücum oyuncusu kendi yarı sahasında."
    elif diff > band:
        decision = OFFSIDE
        reason = f"Hücum oyuncusu ofsayt çizgisinin {diff:.2f} m önünde (çizgi: {source})."
    elif diff < -band:
        decision = ONSIDE
        reason = f"Hücum oyuncusu ofsayt çizgisinin {-diff:.2f} m gerisinde (çizgi: {source})."
    else:
        decision = BORDERLINE
        reason = (f"Fark {diff:+.2f} m, ±{band:.2f} m belirsizlik bandının içinde. "
                  "Ölçüm hatası nedeniyle kesin karar verilemez.")

    if ball_u is None:
        warnings.append("Top konumu verilmedi; 'toptan ileride olma' şartı kontrol edilmedi.")

    return OffsideResult(
        decision=decision, diff_m=float(diff), line_x=float(_x(line_u, attack_right)),
        attacker_x=float(attacker_x),
        second_last_x=None if second_u is None else float(_x(second_u, attack_right)),
        ball_x=None if ball_x is None else float(ball_x),
        line_source=source, reason=reason, warnings=warnings,
    )
