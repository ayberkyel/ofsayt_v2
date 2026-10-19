"""
Saha kalibrasyonu (homography) — rapor Bölüm 8.

Homography H, görüntüdeki zemin düzlemini (piksel) saha düzlemine (metre) taşıyan 3x3 matristir:

        [X]       [u]
    s * [Y] = H * [v]          (u, v) piksel  ->  (X, Y) metre
        [1]       [1]

En az 4 nokta eşleşmesiyle hesaplanır. 4'ten fazla nokta verilirse RANSAC ile hatalı
tıklamalar ayıklanır ve en küçük kareler çözümü kullanılır.
"""
import cv2
import numpy as np

from .pitch import template_lines


def compute_homography(image_pts, pitch_pts, ransac_thresh_m=2.0, ransac_min_pts=8):
    """image_pts: [(u,v),...] piksel, pitch_pts: [(x,y),...] metre. Dönüş: (H, inlier_mask).

    8'den az noktada TÜM noktalarla en küçük kareler çözümü kullanılır. RANSAC az noktada
    tehlikelidir: yayın kameralarının lens bozulması yüzünden birkaç piksellik doğal sapmayı
    "hata" sanıp en bilgilendirici noktayı (ör. kameraya yakın ceza sahası köşesi) atabiliyor;
    örnek görselde bu, o noktada 76 px hataya yol açtı. Gerçekten hatalı tıklamayı arayüz,
    nokta bazındaki yeniden izdüşüm hatasıyla gösterir; kullanıcı siler.
    """
    src = np.asarray(image_pts, dtype=np.float64)
    dst = np.asarray(pitch_pts, dtype=np.float64)
    if len(src) < 4:
        raise ValueError("Homography için en az 4 nokta gerekir.")
    if len(src) < ransac_min_pts:
        H, mask = cv2.findHomography(src, dst, 0)
    else:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh_m)
    if H is None:
        raise ValueError("Homography hesaplanamadı (noktalar doğrusal veya çakışık olabilir).")
    # H ölçek (işaret dahil) belirsizdir: H ve -H aynı dönüşümdür. pitch_to_image'daki
    # "kameranın arkası" kontrolü (w > 0) için işareti sabitliyoruz: görünen noktalarda w > 0.
    w = np.hstack([src, np.ones((len(src), 1))]) @ H[2]
    if np.median(w) < 0:
        H = -H
    return H, mask.ravel().astype(bool)


def _apply(H, pts):
    """Homojen dönüşüm; (N,2) nokta ve w (ölçek) değerini döndürür."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = hom[:, 2:3]
    return hom[:, :2] / w, w.ravel()


def image_to_pitch(pts, H):
    return _apply(H, pts)[0]


def pitch_to_image(pts, H):
    """Saha (metre) -> görüntü (piksel). Kameranın arkasına düşen noktalar (w<=0) NaN olur."""
    out, w = _apply(np.linalg.inv(H), pts)
    out[w <= 0] = np.nan
    return out


def reprojection_error(image_pts, pitch_pts, H):
    """Saha noktalarını görüntüye geri izdüşürüp tıklanan piksellerle farkını ölçer (piksel).

    Not: Tam 4 noktada H bu noktalardan tam geçtiği için hata ~0 çıkar; anlamlı bir
    ölçüm için 5-6 nokta kullanın.
    """
    proj = pitch_to_image(pitch_pts, H)
    err = np.linalg.norm(proj - np.asarray(image_pts, dtype=np.float64), axis=1)
    return float(np.nanmean(err)), err


def geometry_warnings(image_pts, img_w, img_h):
    """Kalibrasyon noktalarının dağılımı için uyarılar (rapor Bölüm 8.3)."""
    warns = []
    pts = np.asarray(image_pts, dtype=np.float32)
    if len(pts) < 4:
        return warns
    hull_area = cv2.contourArea(cv2.convexHull(pts))
    ratio = hull_area / float(img_w * img_h)
    if ratio < 0.05:
        warns.append(f"Noktalar görüntünün yalnızca %{ratio*100:.1f}'ini kaplıyor; "
                     "daha uzak noktalar seçin, aksi halde matris kararsız olur.")
    # Doğrusallık kontrolü: herhangi 3 nokta neredeyse aynı doğru üzerindeyse uyar
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = pts[i], pts[j], pts[k]
                area = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) / 2
                if n == 4 and area < 0.002 * img_w * img_h:
                    warns.append("4 noktanın 3'ü neredeyse aynı doğru üzerinde; matris güvenilmez olabilir.")
                    return warns
    return warns


def has_general_position(pitch_pts, tol_m2=1.0):
    """Homography için en az 4 noktanın 'genel konumda' olması gerekir: hiçbir 3'ü aynı doğru
    üzerinde olmayan 4 nokta bulunmalı. Ör. ceza sahasının iki ön köşesi + ceza yayının iki
    kesişimi (hepsi x = 16.5 m doğrusu üzerinde) matrisi belirlemez; sonuç rastgele bir şekil olur.
    tol_m2: üçgen alanı bu değerin (m²) altındaysa 3 nokta doğrusal sayılır.
    """
    from itertools import combinations
    P = np.asarray(pitch_pts, dtype=float)

    def area(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2

    for quad in combinations(range(len(P)), 4):
        if all(area(*P[list(t)]) > tol_m2 for t in combinations(quad, 3)):
            return True
    return False


def position_uncertainty(image_pts, pitch_pts, probe_img_pts, click_noise_px=2.0, trials=200, seed=0):
    """Kalibrasyonun, verilen görüntü noktalarında (ör. oyuncu ayakları) yarattığı x belirsizliği (m).

    Her tıklamaya ±click_noise_px gürültü eklenip homography yeniden hesaplanır; probe noktalarının
    saha x'indeki sapmanın %95'lik değeri döndürülür. Sınırda bandı en az bu kadar olmalıdır.
    """
    rng = np.random.default_rng(seed)
    img = np.asarray(image_pts, dtype=float)
    H0, _ = compute_homography(img, pitch_pts)
    x0 = image_to_pitch(probe_img_pts, H0)[:, 0]
    errs = []
    for _ in range(trials):
        try:
            H, _ = compute_homography(img + rng.uniform(-click_noise_px, click_noise_px, img.shape), pitch_pts)
        except ValueError:
            continue
        errs.append(np.abs(image_to_pitch(probe_img_pts, H)[:, 0] - x0))
    return float(np.percentile(np.concatenate(errs), 95)) if errs else float("nan")


def outside_pitch_ratio(H, image_pts, tol=5.0):
    """Görüntü noktalarının (ör. oyuncu ayakları) kaçta kaçı saha dışına düşüyor?

    Doğru bir kalibrasyonda oyuncuların neredeyse hepsi 105x68 m'lik sahanın içindedir.
    Oran yüksekse nokta isimleri ile tıklanan yerler uyuşmuyordur (ör. ceza sahası köşeleri
    "köşe bayrağı" olarak kaydedilmiş). Dönüş: (oran, saha koordinatları)
    """
    if len(image_pts) == 0:
        return 0.0, np.zeros((0, 2))
    p = image_to_pitch(image_pts, H)
    from .pitch import PITCH_LENGTH, PITCH_WIDTH
    out = (p[:, 0] < -tol) | (p[:, 0] > PITCH_LENGTH + tol) | (p[:, 1] < -tol) | (p[:, 1] > PITCH_WIDTH + tol)
    return float(out.mean()), p


def player_width_m(H, boxes, q=75):
    """Oyuncu kutularının alt kenarının sahadaki uzunluğu (metre), q. yüzdelik.

    İsimlerden bağımsız bir ölçek kontrolü: bir oyuncu sahada ~0.5-1.3 m yer kaplar. Nokta
    isimleri yanlışsa (ör. ceza sahası 'bütün saha' olarak girilmiş) ölçek şişer ve değer
    2-5 m'ye çıkar. Örnek görselde: doğru kalibrasyon ~1.1 m, yanlış isimler ~2.4 m.
    """
    if not boxes:
        return None
    w = []
    for x1, _, x2, y2 in boxes:
        a, b = image_to_pitch([(x1, y2), (x2, y2)], H)
        w.append(np.linalg.norm(a - b))
    return float(np.percentile(w, q))


def draw_template(img, H, color=(0, 255, 255), thickness=2):
    """Saha şablonunu görüntüye çizer. Çizgiler gerçek saha çizgileriyle örtüşüyorsa kalibrasyon doğrudur."""
    out = img.copy()
    for line in template_lines():
        p = pitch_to_image(line, H)
        # NaN'ları ayırarak parça parça çiz
        seg = []
        for q in p:
            if np.any(np.isnan(q)) or np.any(np.abs(q) > 1e5):
                if len(seg) > 1:
                    cv2.polylines(out, [np.int32(seg)], False, color, thickness, cv2.LINE_AA)
                seg = []
            else:
                seg.append(q)
        if len(seg) > 1:
            cv2.polylines(out, [np.int32(seg)], False, color, thickness, cv2.LINE_AA)
    return out
