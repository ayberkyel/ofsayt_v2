"""
Arayüz süslemeleri: renk paleti, CSS, logo, adım göstergesi, karşılama kartı, VAR tarzı karar kartı
ve yükleme animasyonu. Hesaplama yapmaz; yalnızca HTML/CSS üretir (app.py st.markdown ile basar).

Palet: marshmallow beyazı (zemin), galactic violet (yazı, yan menü), bubblegum pink (vurgu).
Karar renkleri anlam taşıdığı için (ofsayt / değil / sınırda) paletle uyumlu tonlarda ayrı tutulur.
"""
import html
import urllib.parse

import numpy as np
from PIL import Image, ImageDraw

MARSHMALLOW = "#FDF8F3"
LILAC = "#F3ECF8"
VIOLET = "#4A2A7A"
VIOLET_2 = "#6B3FA0"
VIOLET_DARK = "#2E1A4F"
PINK = "#FF7EB6"
PINK_LIGHT = "#FFD3E6"
PINK_DARK = "#C2367F"

# Karar renkleri (HTML); görüntü üzerindeki karşılıkları drawing.py içinde BGR olarak
VERDICT = {"OFSAYT": "#E8436F", "OFSAYT DEĞİL": "#2FA36B", "SINIRDA": "#F2A531"}


def _bgr(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# --------------------------------------------------------------------------------------
# Logo
# --------------------------------------------------------------------------------------
LOGO_SVG = f"""<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{VIOLET_2}"/><stop offset="1" stop-color="{PINK}"/></linearGradient></defs>
  <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#lg)"/>
  <path d="M14 46 L50 18" stroke="{MARSHMALLOW}" stroke-width="3" stroke-dasharray="5 4" stroke-linecap="round"/>
  <line x1="40" y1="10" x2="26" y2="54" stroke="{PINK_LIGHT}" stroke-width="4" stroke-linecap="round"/>
  <circle cx="22" cy="26" r="8" fill="{MARSHMALLOW}"/>
  <path d="M22 20 l5 3.5 -2 6 h-6 l-2 -6 z" fill="{VIOLET}"/>
</svg>"""


def logo_image(size=64):
    """Sekme simgesi (favicon) için PNG: degrade kare, beyaz top, pembe ofsayt çizgisi."""
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    top = np.array([int(VIOLET_2[i:i + 2], 16) for i in (1, 3, 5)], float)
    bot = np.array([int(PINK[i:i + 2], 16) for i in (1, 3, 5)], float)
    grad = Image.new("RGBA", (s, s))
    px = grad.load()
    for y in range(s):
        for x in range(s):
            t = (x + y) / (2 * (s - 1))
            c = (1 - t) * top + t * bot
            px[x, y] = (int(c[0]), int(c[1]), int(c[2]), 255)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=s // 4, fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    d.line((0.62 * s, 0.14 * s, 0.40 * s, 0.86 * s), fill=PINK_LIGHT, width=max(2, s // 14))
    r = 0.15 * s
    d.ellipse((0.34 * s - r, 0.42 * s - r, 0.34 * s + r, 0.42 * s + r), fill=MARSHMALLOW)
    return img


def _ball_texture():
    """Dönen 3B top için yatayda kesintisiz tekrarlanan desen (200x100): beşgenler ve dikiş çizgileri."""
    def pent(cx, cy, r=13):
        pts = [(cx + r * np.sin(a), cy - r * np.cos(a)) for a in np.linspace(0, 2 * np.pi, 6)[:-1]]
        return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + " Z"
    centers = [(50, 18), (150, 18), (0, 52), (100, 52), (200, 52), (50, 86), (150, 86)]
    seams = [((0, 52), (50, 18)), ((50, 18), (100, 52)), ((100, 52), (150, 18)), ((150, 18), (200, 52)),
             ((0, 52), (50, 86)), ((50, 86), (100, 52)), ((100, 52), (150, 86)), ((150, 86), (200, 52)),
             ((50, 18), (50, -20)), ((150, 18), (150, -20)), ((50, 86), (50, 120)), ((150, 86), (150, 120))]
    lines = "".join(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"/>' for a, b in seams)
    shapes = "".join(f'<path d="{pent(x, y)}"/>' for x, y in centers)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
           f'<rect width="200" height="100" fill="{MARSHMALLOW}"/>'
           f'<g stroke="#CDBFE0" stroke-width="2">{lines}</g><g fill="{VIOLET_DARK}">{shapes}</g></svg>')
    return "data:image/svg+xml," + urllib.parse.quote(svg)


def ball3d(size=110, cls=""):
    """Saf CSS ile dönen 3B futbol topu (JS yok). Gölgelendirme sabit kalır, yalnızca desen döner."""
    return f'<div class="ot-ball3d {cls}" style="--s:{size}px"><i></i></div>'


def sidebar_brand():
    return f"""<div class="ot-brand"><div class="ot-logo3d">{LOGO_SVG}</div><div><div class="ot-brand-title">Ofsayt Tespit</div>
<div class="ot-brand-sub">yarı otomatik VAR</div></div></div>"""


# --------------------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------------------
CSS = f"""
<style>
/* ---------- Başlıklar ---------- */
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3 {{
  color: {VIOLET} !important; letter-spacing: -0.01em;
}}
[data-testid="stMain"] h2 {{ border-bottom: 3px solid {PINK}; padding-bottom: .35rem; display: inline-block; }}

/* ---------- Düğmeler ---------- */
[data-testid="stBaseButton-primary"] {{
  background: linear-gradient(135deg, {VIOLET} 0%, {VIOLET_2} 45%, {PINK} 100%) !important;
  color: {MARSHMALLOW} !important; border: none !important; font-weight: 600 !important;
  box-shadow: 0 4px 14px rgba(74, 42, 122, .25); transition: transform .12s ease, box-shadow .12s ease;
}}
[data-testid="stBaseButton-primary"]:hover {{ transform: translateY(-1px); box-shadow: 0 6px 18px rgba(255, 126, 182, .45); }}
[data-testid="stBaseButton-secondary"] {{
  background: {MARSHMALLOW} !important; color: {VIOLET} !important;
  border: 1.5px solid {VIOLET} !important; font-weight: 500 !important;
  transition: background .12s ease, color .12s ease, border-color .12s ease;
}}
[data-testid="stBaseButton-secondary"]:hover {{ background: {PINK_LIGHT} !important; border-color: {PINK} !important; }}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
  background: transparent !important; color: {MARSHMALLOW} !important; border-color: {PINK} !important;
}}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{ background: {PINK} !important; color: {VIOLET} !important; }}

/* ---------- Bildirim kutuları ---------- */
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentSuccess"],
[data-testid="stAlertContentWarning"], [data-testid="stAlertContentError"] {{ color: {VIOLET_DARK} !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
  background: {LILAC} !important; border-left: 5px solid {VIOLET_2}; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
  background: #FFEAF3 !important; border-left: 5px solid {PINK}; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
  background: #FFF4E0 !important; border-left: 5px solid {VERDICT["SINIRDA"]}; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
  background: #FDE3EA !important; border-left: 5px solid {VERDICT["OFSAYT"]}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: #E4D6F0 !important; }}

/* ---------- Yan menü markası ---------- */
.ot-brand {{ display: flex; align-items: center; gap: .75rem; margin: .25rem 0 1rem; }}
.ot-brand svg {{ width: 48px; height: 48px; flex: none; filter: drop-shadow(0 4px 10px rgba(0,0,0,.25)); }}
.ot-brand-title {{ color: {MARSHMALLOW}; font-weight: 700; font-size: 1.35rem; line-height: 1.1; }}
.ot-brand-sub {{ color: {PINK_LIGHT}; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; }}

/* ---------- Adım göstergesi ---------- */
.ot-steps {{ display: flex; align-items: flex-start; margin: .2rem 0 1.2rem; }}
.ot-step {{ display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; width: 92px; text-align: center; }}
.ot-dot {{ width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
          font-weight: 700; font-size: .95rem; border: 2px solid #D9C8EA; color: #A893C2; background: {MARSHMALLOW}; }}
.ot-step.done .ot-dot {{ background: {PINK}; border-color: {PINK}; color: {MARSHMALLOW}; }}
.ot-step.current .ot-dot {{ background: {VIOLET}; border-color: {VIOLET}; color: {MARSHMALLOW};
          box-shadow: 0 0 0 5px rgba(255, 126, 182, .35); }}
.ot-label {{ margin-top: .35rem; font-size: .78rem; color: #8C78A8; line-height: 1.15; }}
.ot-step.current .ot-label, .ot-step.done .ot-label {{ color: {VIOLET}; font-weight: 600; }}
.ot-bar {{ flex: 1 1 auto; height: 4px; border-radius: 4px; background: #E4D6F0; margin-top: 16px; min-width: 16px; }}
.ot-bar.done {{ background: linear-gradient(90deg, {PINK}, {VIOLET_2}); }}

/* ---------- Karşılama kartı ---------- */
.ot-hero {{ border-radius: 20px; padding: 1.6rem 1.8rem; color: {MARSHMALLOW}; margin: .5rem 0 1rem;
           background: radial-gradient(circle at 85% 20%, rgba(255,126,182,.55), transparent 45%),
                       linear-gradient(135deg, {VIOLET_DARK} 0%, {VIOLET} 55%, {VIOLET_2} 100%);
           box-shadow: 0 10px 30px rgba(74, 42, 122, .25); }}
.ot-hero h3 {{ color: {MARSHMALLOW} !important; margin: 0 0 .3rem; font-size: 1.5rem; border: none; }}
.ot-hero p {{ color: {PINK_LIGHT}; margin: 0 0 1.1rem; }}
.ot-hero-head {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin-bottom: .6rem; }}
@media (max-width: 640px) {{ .ot-hero-head .ot-ball3d {{ --s: 64px !important; }} }}
.ot-hero-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .7rem; }}
.ot-hero-card {{ background: rgba(253, 248, 243, .1); border: 1px solid rgba(253, 248, 243, .18);
                border-radius: 14px; padding: .8rem .9rem; }}
.ot-hero-card b {{ display: block; color: {MARSHMALLOW}; margin: .25rem 0 .15rem; }}
.ot-hero-card span {{ color: #E6DAF5; font-size: .85rem; line-height: 1.3; }}
.ot-hero-ico {{ font-size: 1.5rem; }}

/* ---------- Karar kartı ---------- */
.ot-verdict {{ border-radius: 20px; padding: 1.3rem 1.6rem 1.1rem; color: {MARSHMALLOW}; margin: .4rem 0 1rem;
              background: linear-gradient(135deg, {VIOLET_DARK} 0%, {VIOLET} 60%, {VIOLET_2} 100%);
              box-shadow: 0 10px 30px rgba(46, 26, 79, .3); position: relative; overflow: hidden; }}
.ot-verdict-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; flex-wrap: wrap; }}
.ot-verdict-tag {{ font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; color: {PINK_LIGHT}; }}
.ot-verdict-word {{ font-size: 2.6rem; font-weight: 800; line-height: 1.05; margin-top: .15rem; }}
.ot-verdict-diff {{ text-align: right; }}
.ot-verdict-diff b {{ font-size: 2.4rem; font-weight: 800; line-height: 1; }}
.ot-verdict-diff span {{ display: block; color: {PINK_LIGHT}; font-size: .8rem; }}
.ot-verdict-reason {{ color: #E6DAF5; margin: .55rem 0 .9rem; font-size: .95rem; }}
.ot-gauge {{ position: relative; height: 14px; border-radius: 8px; overflow: hidden; display: flex; }}
.ot-gauge div {{ height: 100%; }}
.ot-marker {{ position: absolute; top: -6px; width: 4px; height: 26px; border-radius: 3px; background: {MARSHMALLOW};
             box-shadow: 0 0 0 3px rgba(46,26,79,.55); transform: translateX(-2px); }}
.ot-gauge-wrap {{ position: relative; padding-top: 6px; }}
.ot-gauge-labels {{ display: flex; justify-content: space-between; font-size: .75rem; color: #D9C8EA; margin-top: .35rem; }}

/* ---------- 3B top (saf CSS): desen döner, ışık/gölge sabit kalır ---------- */
.ot-ball3d {{ --s: 110px; width: var(--s); height: var(--s); border-radius: 50%; position: relative;
             overflow: hidden; flex: none; isolation: isolate; background: {MARSHMALLOW}; }}
.ot-ball3d i {{ position: absolute; inset: -25%; transform: rotate(-22deg);
               background: url("{_ball_texture()}") repeat-x 0 50% / calc(var(--s) * 3) calc(var(--s) * 1.5);
               animation: ot-ball-spin 4s linear infinite; }}
.ot-ball3d::after {{ content: ""; position: absolute; inset: 0; border-radius: 50%; z-index: 1;
  background: radial-gradient(circle at 32% 26%, rgba(255,255,255,.95) 0 7%, rgba(255,255,255,.35) 18%, transparent 36%),
              radial-gradient(circle at 42% 38%, transparent 42%, rgba(46,26,79,.28) 70%, rgba(46,26,79,.75) 100%); }}
@keyframes ot-ball-spin {{ to {{ background-position-x: calc(var(--s) * -3); }} }}
.ot-ball3d.fast i {{ animation-duration: .9s; }}
.ot-float {{ display: flex; flex-direction: column; align-items: center; flex: none; }}
.ot-float .ot-ball3d {{ animation: ot-bob 3.2s ease-in-out infinite;
                       box-shadow: 0 0 0 6px rgba(255,126,182,.18), 0 0 40px rgba(255,126,182,.45); }}
.ot-float-shadow {{ width: 70%; height: 12px; border-radius: 50%; margin-top: 14px;
                   background: radial-gradient(rgba(20,8,40,.55), transparent 70%); animation: ot-bob-shadow 3.2s ease-in-out infinite; }}
@keyframes ot-bob {{ 50% {{ transform: translateY(-12px); }} }}
@keyframes ot-bob-shadow {{ 50% {{ transform: scaleX(.75); opacity: .55; }} }}

/* ---------- 3B kart eğimi ---------- */
.ot-hero, .ot-verdict {{ transform-style: preserve-3d; transition: transform .5s cubic-bezier(.2,.8,.2,1), box-shadow .5s ease; }}
.ot-hero:hover {{ transform: perspective(1400px) rotateX(2deg) rotateY(-2deg); box-shadow: 0 22px 44px rgba(74,42,122,.35); }}
.ot-hero-grid {{ perspective: 800px; }}
.ot-hero-card {{ position: relative; transform-style: preserve-3d; overflow: hidden;
                transition: transform .45s cubic-bezier(.2,.8,.2,1), background .45s ease, box-shadow .45s ease; }}
.ot-hero-card .ot-hero-ico {{ display: inline-block; transition: transform .45s cubic-bezier(.2,.8,.2,1); }}
.ot-hero-card::after {{ content: ""; position: absolute; inset: 0; pointer-events: none; opacity: 0;
  background: linear-gradient(115deg, transparent 30%, rgba(255,255,255,.28) 48%, transparent 62%);
  transform: translateX(-60%); transition: transform .7s ease, opacity .3s ease; }}
.ot-hero-card:hover {{ transform: rotateX(10deg) rotateY(-12deg) translateZ(18px);
  background: rgba(253,248,243,.17); box-shadow: 12px 16px 26px rgba(20,8,40,.35); }}
.ot-hero-card:nth-child(even):hover {{ transform: rotateX(10deg) rotateY(12deg) translateZ(18px);
  box-shadow: -12px 16px 26px rgba(20,8,40,.35); }}
.ot-hero-card:hover .ot-hero-ico {{ transform: translateZ(34px) scale(1.25); }}
.ot-hero-card:hover::after {{ opacity: 1; transform: translateX(60%); }}
.ot-verdict {{ animation: ot-flip-in .8s cubic-bezier(.2,.9,.25,1.15) backwards; transform-origin: 50% 0; }}
.ot-verdict:hover {{ transform: perspective(1200px) rotateX(3deg); box-shadow: 0 24px 46px rgba(46,26,79,.4); }}
@keyframes ot-flip-in {{ from {{ transform: perspective(1200px) rotateX(-55deg); opacity: 0; }}
                        to {{ transform: perspective(1200px) rotateX(0); opacity: 1; }} }}

/* ---------- 3B logo (kalınlığı olan, ara sıra dönen kare) ---------- */
.ot-logo3d {{ width: 48px; height: 48px; flex: none; position: relative; transform-style: preserve-3d;
             animation: ot-logo-spin 9s cubic-bezier(.6,0,.2,1) infinite; }}
.ot-logo3d svg {{ position: absolute; inset: 0; transform: translateZ(4px); }}
.ot-logo3d::before, .ot-logo3d::after {{ content: ""; position: absolute; inset: 1px; border-radius: 12px; background: {VIOLET_DARK}; }}
.ot-logo3d::before {{ transform: translateZ(0); }}
.ot-logo3d::after {{ transform: translateZ(-4px); background: {PINK_DARK}; }}
.ot-brand {{ perspective: 400px; }}
.ot-brand:hover .ot-logo3d {{ animation-duration: 1.2s; }}
@keyframes ot-logo-spin {{ 0%, 70% {{ transform: rotateX(12deg) rotateY(-22deg); }}
                          85% {{ transform: rotateX(12deg) rotateY(338deg) translateZ(6px); }}
                          100% {{ transform: rotateX(12deg) rotateY(338deg); }} }}
@media (prefers-reduced-motion: reduce) {{
  .ot-ball3d i, .ot-float .ot-ball3d, .ot-float-shadow, .ot-logo3d, .ot-verdict {{ animation: none !important; }}
}}

/* ---------- Yükleme animasyonu ---------- */
.ot-busy {{ display: flex; align-items: center; gap: .8rem; color: {VIOLET}; font-weight: 600; margin: .3rem 0; }}
.ot-ball {{ width: 28px; height: 28px; border-radius: 50%; position: relative;
           background: radial-gradient(circle at 35% 35%, #fff 0 35%, {MARSHMALLOW} 36% 100%);
           border: 2px solid {VIOLET}; animation: ot-roll 0.9s linear infinite, ot-bounce 0.9s ease-in-out infinite; }}
.ot-ball::after {{ content: ""; position: absolute; inset: 8px; background: {VIOLET}; border-radius: 3px;
                  clip-path: polygon(50% 0, 100% 38%, 82% 100%, 18% 100%, 0 38%); }}
.ot-busy-ball .ot-ball3d {{ animation: ot-bounce 0.9s ease-in-out infinite; }}
.ot-shadow {{ width: 22px; height: 5px; border-radius: 50%; background: rgba(255,126,182,.55);
             margin: -2px 0 0 3px; animation: ot-shadow 0.9s ease-in-out infinite; }}
@keyframes ot-roll {{ to {{ transform: rotate(360deg); }} }}
@keyframes ot-bounce {{ 50% {{ translate: 0 -7px; }} }}
@keyframes ot-shadow {{ 50% {{ transform: scaleX(.6); opacity: .5; }} }}
</style>
"""


# --------------------------------------------------------------------------------------
# Bileşenler (HTML)
# --------------------------------------------------------------------------------------
def stepper(labels, current, done):
    """labels: adım adları, current: aktif adım (0..), done: tamamlanan adımlar (bool listesi)."""
    parts = []
    for i, lab in enumerate(labels):
        cls = "current" if i == current else ("done" if done[i] else "")
        mark = "✓" if done[i] and i != current else str(i + 1)
        parts.append(f'<div class="ot-step {cls}"><div class="ot-dot">{mark}</div>'
                     f'<div class="ot-label">{html.escape(lab)}</div></div>')
        if i < len(labels) - 1:
            parts.append(f'<div class="ot-bar {"done" if done[i] else ""}"></div>')
    return '<div class="ot-steps">' + "".join(parts) + "</div>"


def welcome():
    cards = [("🎬", "1 · Kare", "Video yükleyin; pas anı bulunur, video o anda durur."),
             ("⚪", "2 · Top", "Topun zemine değdiği noktayı işaretleyin."),
             ("🧍", "3 · Oyuncular", "YOLOv8 oyuncuları bulur, forma rengine göre ayırır."),
             ("📐", "4 · Kalibrasyon", "Sahada bilinen 4+ noktaya tıklayın: piksel → metre."),
             ("🚩", "5 · Karar", "Perspektife uygun ofsayt çizgisi ve metre cinsinden fark.")]
    grid = "".join(f'<div class="ot-hero-card"><div class="ot-hero-ico">{i}</div><b>{t}</b><span>{d}</span></div>'
                   for i, t, d in cards)
    return (f'<div class="ot-hero"><div class="ot-hero-head"><div><h3>Ofsayt pozisyonunu beş adımda inceleyin</h3>'
            f'<p>Başlamak için yukarıdan 10–15 saniyelik bir maç videosu ya da tek bir kare yükleyin.</p></div>'
            f'<div class="ot-float">{ball3d(104)}<div class="ot-float-shadow"></div></div></div>'
            f'<div class="ot-hero-grid">{grid}</div></div>')


def verdict_card(decision, diff_m, band, reason, line_source=""):
    """VAR tarzı karar kartı: büyük karar, metre farkı ve fark ölçeği (bant gösterimi)."""
    color = VERDICT.get(decision, PINK)
    span = max(2.0, 2.5 * band, abs(diff_m) * 1.15)          # ölçeğin yarı genişliği (m)
    def pos(x):
        return float(np.clip((x + span) / (2 * span) * 100, 1.5, 98.5))
    left, right = pos(-band), pos(band)
    gauge = (f'<div style="width:{left:.1f}%;background:{VERDICT["OFSAYT DEĞİL"]}"></div>'
             f'<div style="width:{right - left:.1f}%;background:{VERDICT["SINIRDA"]}"></div>'
             f'<div style="flex:1;background:{VERDICT["OFSAYT"]}"></div>')
    src = f" · çizgi: {html.escape(line_source)}" if line_source else ""
    return f"""<div class="ot-verdict" style="border-top: 6px solid {color}">
  <div class="ot-verdict-top">
    <div><div class="ot-verdict-tag">VAR kontrolü · karar{src}</div>
         <div class="ot-verdict-word" style="color:{color}">{html.escape(decision)}</div></div>
    <div class="ot-verdict-diff"><b>{diff_m:+.2f} m</b><span>hücumcu − ofsayt çizgisi</span></div>
  </div>
  <div class="ot-verdict-reason">{html.escape(reason)}</div>
  <div class="ot-gauge-wrap">
    <div class="ot-gauge">{gauge}</div>
    <div class="ot-marker" style="left:{pos(diff_m):.1f}%"></div>
  </div>
  <div class="ot-gauge-labels"><span>−{span:.1f} m · geride</span><span>sınırda ±{band:.2f} m</span>
       <span>+{span:.1f} m · önde</span></div>
</div>"""


def busy(text):
    return (f'<div class="ot-busy"><div class="ot-busy-ball">{ball3d(30, "fast")}<div class="ot-shadow"></div></div>'
            f'<span>{html.escape(text)}</span></div>')


# --------------------------------------------------------------------------------------
# Adımlar arası geçiş ekranı
# --------------------------------------------------------------------------------------
STEP_INTRO = [("Kare seçimi", "Pas anını bulun"),
              ("Top konumu", "Topun zemine değdiği noktayı işaretleyin"),
              ("Oyuncu tespiti", "YOLOv8 oyuncuları bulur, takımlar forma rengiyle ayrılır"),
              ("Kalibrasyon", "Sahada bilinen noktalara tıklayın: piksel → metre"),
              ("Karar", "Ofsayt çizgisi hesaplanıyor")]

TRANSITION_CSS = f"""
<style>
.ot-trans {{ position: fixed; inset: 0; z-index: 999999; pointer-events: none;
            display: flex; align-items: center; justify-content: center; flex-direction: column; gap: .6rem;
            background: radial-gradient(circle at 80% 20%, rgba(255,126,182,.55), transparent 45%),
                        linear-gradient(135deg, {VIOLET_DARK} 0%, {VIOLET} 55%, {VIOLET_2} 100%);
            animation: ot-trans-out 1.35s ease forwards; }}
.ot-trans-num {{ width: 84px; height: 84px; border-radius: 50%; background: {PINK}; color: {MARSHMALLOW};
                display: flex; align-items: center; justify-content: center; font-size: 2.4rem; font-weight: 800;
                box-shadow: 0 0 0 10px rgba(255,126,182,.25); animation: ot-pop .45s cubic-bezier(.2,1.6,.4,1) both; }}
.ot-trans-title {{ color: {MARSHMALLOW}; font-size: 2rem; font-weight: 800; animation: ot-rise .45s .08s ease both; }}
.ot-trans-sub {{ color: {PINK_LIGHT}; font-size: 1.05rem; animation: ot-rise .45s .16s ease both; }}
.ot-trans-track {{ position: relative; width: 260px; height: 4px; border-radius: 4px; margin-top: 1rem;
                  background: rgba(253,248,243,.2); overflow: visible; }}
.ot-trans-track::before {{ content: ""; position: absolute; left: 0; top: 0; height: 100%; border-radius: 4px;
                          background: {PINK}; animation: ot-fill .9s ease forwards; }}
.ot-trans-ball {{ position: absolute; top: -13px; width: 30px; height: 30px; border-radius: 50%;
                 box-shadow: 0 8px 14px rgba(20,8,40,.45); animation: ot-kick .9s ease forwards; }}
.ot-trans-ball .ot-ball3d i {{ animation-duration: .45s; }}
.ot-trans-num {{ transform-style: preserve-3d; }}
@keyframes ot-trans-out {{ 0%, 70% {{ opacity: 1; }} 100% {{ opacity: 0; visibility: hidden; }} }}
@keyframes ot-pop {{ from {{ transform: perspective(500px) rotateY(-200deg) scale(.4); opacity: 0; }} to {{ transform: perspective(500px) rotateY(0) scale(1); opacity: 1; }} }}
@keyframes ot-rise {{ from {{ transform: perspective(600px) rotateX(70deg) translateY(14px); opacity: 0; }} to {{ transform: none; opacity: 1; }} }}
@keyframes ot-fill {{ from {{ width: 0; }} to {{ width: 100%; }} }}
@keyframes ot-kick {{ 0% {{ left: -15px; }} 45% {{ top: -30px; }} 100% {{ left: 245px; top: -13px; }} }}
</style>
"""


def transition(step_index):
    """Adım değişince ~1,3 sn görünüp kendiliğinden kaybolan tam ekran geçiş. Tıklamaları engellemez."""
    title, sub = STEP_INTRO[step_index]
    return (TRANSITION_CSS + f'<div class="ot-trans"><div class="ot-trans-num">{step_index + 1}</div>'
            f'<div class="ot-trans-title">{html.escape(title)}</div><div class="ot-trans-sub">{html.escape(sub)}</div>'
            f'<div class="ot-trans-track"><div class="ot-trans-ball">{ball3d(30, "fast")}</div></div></div>')
