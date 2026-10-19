// Tıklanabilir görsel bileşeni (streamlit-image-coordinates'ın düzeltilmiş hali).
//
// Orijinal bileşendeki sorunlar:
//   1) img.onload, img.src atandıktan SONRA bağlanıyordu; görsel önbellekten hızlı gelince
//      boyutlandırma hiç çalışmıyor, görsel çerçeveden taşıp kırpılıyordu.
//   2) Görselin genişliği piksel olarak sabitleniyordu; ekran ölçekleme / tarayıcı
//      yakınlaştırmasında çerçeveye sığmıyordu.
// Burada: görsel CSS ile çerçeveye sığdırılır (max-width: 100%) ve tıklama koordinatı ile
// birlikte GERÇEKTE ÇİZİLEN boyut gönderilir. Python tarafı orijinal piksele bunu kullanarak çevirir.

const img = document.getElementById("image");

function updateHeight() {
  const h = img.getBoundingClientRect().height;
  if (h > 0) Streamlit.setFrameHeight(Math.ceil(h));
}

img.addEventListener("load", updateHeight);
window.addEventListener("resize", updateHeight);

img.addEventListener("click", (event) => {
  const r = img.getBoundingClientRect();
  Streamlit.setComponentValue({
    x: event.clientX - r.left,
    y: event.clientY - r.top,
    width: r.width,          // ekranda çizilen genişlik (ölçekleme için)
    height: r.height,
    unix_time: Date.now(),
  });
});

function onRender(event) {
  const { src, width } = event.detail.args;
  img.style.width = width ? width + "px" : "";
  if (img.getAttribute("src") !== src) {
    img.setAttribute("src", src);
  }
  // Önbellekten anında yüklenmişse "load" olayı kaçmış olabilir
  if (img.complete && img.naturalWidth > 0) updateHeight();
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
