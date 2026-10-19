"""
Tıklanabilir görsel bileşeni (streamlit-image-coordinates'ın düzeltilmiş, sadeleştirilmiş hali).

Neden kendi bileşenimiz? Orijinal bileşen bazı tarayıcılarda (Brave, Windows ekran ölçekleme)
görseli çerçeveden büyük çiziyor ve sağ/alt kenarlar kesiliyordu. Ayrıntı: frontend/main.js.
"""
import base64
from io import BytesIO
from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component("click_image", path=str(Path(__file__).parent / "frontend"))


def click_image(pil_image, key, width=None, jpeg_quality=92):
    """Görseli gösterir. Son tıklamayı {x, y, width, height, unix_time} olarak döndürür.

    x, y ve width ekranda ÇİZİLEN görsele göredir; orijinal piksel = x * (orijinal_genişlik / width).
    width: en fazla bu kadar CSS pikseli (çerçeve daha darsa çerçeveye sığdırılır).
    """
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return _component(src=src, width=width, key=key, default=None)
