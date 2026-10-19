# Ofsayt Tespit Sistemi (VAR Benzeri) — Adım Adım Kılavuz

Bu klasör, bitirme projesi raporundaki sistemin sıfırdan, temiz bir uygulamasıdır.
Her dosyanın başında ne yaptığı ve **neden** öyle yapıldığı Türkçe açıklanmıştır.
Bu belge ise bütün resmi ve sırayla yapılacakları anlatır.

---

## 0. Büyük resim: Sistem ne yapıyor?

```
 Video ──► [1] Pas anı karesi ──► [2] Top tıklama
                    │
                    ▼
          [3] YOLOv8: oyuncu / kaleci / hakem kutuları
              + KMeans: forma rengiyle Takım A / Takım B
                    │
                    ▼
          [4] Kalibrasyon: 4+ bilinen saha noktası ──► Homography H (piksel → metre)
                    │
                    ▼
          [5] Ofsayt motoru (saha koordinatında, metre)
              çizgi = max(sondan 2. savunmacı, top, orta çizgi)
              fark  = hücumcu − çizgi  ──►  OFSAYT / OFSAYT DEĞİL / SINIRDA
                    │
                    ▼
              Çizgi H⁻¹ ile görüntüye geri çizilir (perspektife uygun, eğik)
```

**Neden saha koordinatı?** TV görüntüsünde perspektif vardır: kameradan uzaklaştıkça
1 metre daha az piksel kaplar ve kale çizgisine paralel çizgiler ekranda eğik görünür.
Ekranda düz bir çizgi çekip "hangisi daha sağda?" diye bakmak yanlıştır. Doğrusu, her
oyuncunun ayak noktasını **gerçek saha düzlemine** (105 × 68 m) taşıyıp orada karşılaştırmaktır.
Bu taşımayı yapan 3×3 matrise **homography** denir.

---

## 1. Klasör yapısı

```
ofsayt_v2/
├── app.py                     Streamlit arayüzü (5 adım)
├── src/
│   ├── pitch.py               Saha ölçüleri, kalibrasyon noktaları, saha çizgi şablonu
│   ├── calibration.py         Homography, yeniden izdüşüm hatası, şablon çizimi
│   ├── detector.py            YOLOv8 tespiti + KMeans takım ayrımı
│   ├── offside.py             Ofsayt karar motoru (kural + sınırda bandı)
│   ├── pipeline.py            "Sahne" → karar (arayüz + analizler aynı kodu kullanır)
│   ├── drawing.py             Kutular, çizgiler, kuşbakışı mini saha
│   └── video.py               Video bilgisi ve kare okuma
├── scripts/
│   ├── prepare_dataset.py     ADIM 1: CVAT zip → YOLO veri seti (sızıntısız bölme)
│   ├── relabel_roles.py       ADIM 1b: kaybolan kaleci/hakem sınıflarını yarı otomatik geri getirme
│   ├── train.py               ADIM 2: YOLOv8 eğitimi (GPU)
│   ├── evaluate_model.py      ADIM 3: test setinde P / R / mAP / karışıklık matrisi
│   ├── evaluate_decisions.py  ADIM 6: karar doğruluğu, bant taraması
│   └── sensitivity.py         ADIM 7: tıklama hatası duyarlılık analizi
├── tests/
│   ├── test_core.py           Geometri + kural testleri (sentetik, bilinen cevaplı)
│   └── test_app.py            Arayüzün tüm adımlarının hatasız çalıştığı testi
├── data/dataset/              (prepare_dataset.py üretir)
├── models/best.pt             (train.py kopyalar)
├── runs/                      (eğitim çıktıları, grafikler)
├── scenes/                    (arayüzden kaydedilen test pozisyonları)
└── results/                   (rapora girecek tablolar ve grafikler)
```

---

## 2. Kurulum (bir kere)

Bilgisayarında `ofsayt` adlı bir conda ortamı zaten var ve kurulumu yapıldı. Ortamda
PyTorch **CPU** sürümü kuruluydu; **CUDA 12.8** sürümüyle değiştirildi, artık RTX 3050 Ti kullanılıyor.

Her yeni terminalde:

```bash
conda activate ofsayt
cd "C:\Users\ayber\Desktop\OFSAYT TESPİT SİSTEMİ\ofsayt_v2"
```

GPU'nun görüldüğünü kontrol et (`True` yazmalı):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Başka bir bilgisayarda sıfırdan kurulum:

```bash
conda create -n ofsayt python=3.10 -y
conda activate ofsayt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## 3. ADIM 1 — Veri setini CVAT'tan indir ve hazırla

### 3.1 Veri
Veri `Desktop\OFSAYT GORSEL-ETİKETLER` klasöründe: 6 maç, her maç için `gorsel_macN.zip`
(görseller) + `etiket_macN.zip` (CVAT YOLO 1.1 etiketleri). Toplam **674 görsel, 674 etiket
dosyası** (4'ü boş, yani oyuncusuz kare; bunlar arka plan örneği olarak kalır).

### 3.2 Dönüştür
```bash
python scripts/prepare_dataset.py --dir "C:\Users\ayber\Desktop\OFSAYT GORSEL-ETİKETLER"
```
Betik klasördeki bütün zip'leri açar ve görselleri etiketlerle **dosya adına göre** eşleştirir.
Tek tek vermek için `--zip a.zip --zip b.zip` de kullanılabilir.

**Burada ne oluyor? (Rapor 7.2 — veri sızıntısı)**
Aynı videonun art arda kareleri neredeyse aynıdır. Rastgele bölersek, modelin eğitimde
gördüğü karenin bir sonraki karesi teste düşer ve test sonucu yapay olarak yüksek çıkar.
Betik bunu iki şekilde önler:
- Dosya adlarında **3 veya daha fazla farklı maç** varsa (`mac1_…`, `mac2_…`, `mac3_…`), bölme
  **maç bazında** yapılır: bir maçın bütün kareleri aynı bölüme gider.
- Daha az maç varsa her video **zamana göre ardışık bloklara** bölünür (%70 / %15 / %15) ve
  bloklar arasında birkaç kare atlanır.

6 maç olduğu için **maç bazında bölme** yapıldı (seed = 42):

| bölüm | maçlar | görsel |
|---|---|---|
| train | mac2, mac3, mac4, mac6 | 440 |
| val | mac1 | 109 |
| test | mac5 | 125 |

Yani test sonucu, modelin **hiç görmediği bir maçtaki** başarısını ölçer. Bu, raporda güçlü bir nokta.

### 3.3 ADIM 1b — Kaleci / hakem sınıflarını geri getir (yarı otomatik etiketleme)

**Sorun:** CVAT'tan alınan son etiketlerde bütün kutular `player` (sınıf 0) olarak kaydedilmiş;
kaleci ve hakem sınıfları kaybolmuş. (Eski mac1 export'unda 74 kaleci ve 86 hakem kutusu vardı,
yenisinde aynı kutular `player`.) Kutuların **konumları** doğru, yalnızca **sınıfları** eksik.

```bash
python scripts/relabel_roles.py
```

**Yöntem:** Daha önce kaleci ve hakemi öğrenmiş eski model (`../models/best.pt`) her görselde
çalışır. Her etiket kutusu modelin tahminleriyle **IoU ≥ 0.5** ile eşleştirilir. Tahmin
kaleci/hakemse ve güveni **≥ 0.4** ise yalnızca sınıf değiştirilir; kutunun konumu insan
etiketinden gelir ve değişmez. Bir karede en fazla 2 kaleci ve 4 hakem kabul edilir.

**Doğrulama:**
1. mac1'de elle yapılmış eski etiketlerle karşılaştırma:

   | | precision | recall |
   |---|---|---|
   | goalkeeper | 0.967 | 0.952 |
   | referee | 0.987 | 0.893 |

   (Eski model mac1 ile eğitildiği için bu değerler iyimserdir. Raporda bunu belirt.)
2. Değişen 881 kutunun hepsi `results/relabel_review/` altındaki görsellerde gözle kontrol edildi.
   Bulunan **2 hata** `data/manual_fixes.csv` ile düzeltildi: bir saha oyuncusu "kaleci", bir
   oyuncu "hakem" sanılmıştı.

Son sınıf dağılımı:

| bölüm | player | goalkeeper | referee |
|---|---|---|---|
| train | 6413 | 259 | 341 |
| val | 1716 | 62 | 78 |
| test | 1915 | 60 | 79 |

> Kaleci ve hakem örnekleri oyuncuya göre ~25 kat azdır (sınıf dengesizliği). Raporda yorumla.
> Raporda bu adımı **"yarı otomatik etiketleme (model destekli etiket düzeltme + insan kontrolü)"**
> olarak anlat; jüri için artı puan.

---

## 4. ADIM 2 — Modeli eğit

```bash
python scripts/train.py
```

| Ayar | Değer | Neden |
|---|---|---|
| model | `yolov8s.pt` | 4 GB VRAM'e sığar. COCO'da "person" öğrenmiş ağırlıklardan başlar (transfer learning); 600 görsel için şart |
| imgsz | 960 | Geniş açıda oyuncular ~20×60 px. 640'ta kayboluyorlar |
| batch | 4 | Test edildi: 960 px'te ~2.1 GB VRAM. `--batch 6` de denenebilir |
| epochs / patience | 100 / 25 | 25 epoch iyileşme olmazsa erken durur |
| fliplr | 0.5 | Saha yatayda simetrik, aynalama gerçekçi |

Süre: 600 görselde epoch başına yaklaşık 30-40 saniye, toplamda **~1 saat**.

Bitince:
- `runs/ofsayt_yolov8s/` → `results.png` (kayıp ve mAP eğrileri), `confusion_matrix.png`, `PR_curve.png`
- En iyi model otomatik olarak `models/best.pt`'ye kopyalanır.

**`results.png` nasıl okunur?** `train/box_loss` ve `val/box_loss` birlikte düşmeli. Eğitim
kaybı düşerken doğrulama kaybı yükseliyorsa aşırı öğrenme (overfitting) vardır.

---

## 5. ADIM 3 — Modeli test setinde değerlendir

```bash
python scripts/evaluate_model.py
```
`results/model_test/metrics.csv` ve karışıklık matrisini üretir. **Rapora bu sayılar girer**,
eğitimdeki val sayıları değil (val, en iyi epoch'u seçmek için kullanıldığından iyimserdir).

### 5.1 Elde edilen sonuçlar (yolov8s, 960 px)

Eğitim 52. epoch'ta erken durdu; en iyi model **27. epoch** (val mAP50-95 = 0.760).
Test seti = **mac5**, yani modelin hiç görmediği bir maç (125 görsel, 2054 kutu):

| sınıf | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| player | 0.966 | 0.893 | 0.966 | 0.803 |
| goalkeeper | 0.915 | 0.899 | 0.917 | 0.777 |
| referee | 0.756 | 0.904 | 0.838 | 0.744 |
| **tümü** | **0.879** | **0.899** | **0.907** | **0.775** |

Hız: kare başına ~10 ms (RTX 3050 Ti).

### 5.2 Önemli bulgu: sınıftan bağımsız NMS (agnostic NMS)
İlk karışıklık matrisinde kalecilerin %32'si ve hakemlerin %46'sı "player" görünüyordu.
Sebep: YOLO, NMS'yi (çakışan kutuları eleme) **her sınıf için ayrı** yapar. Bu yüzden aynı kişiye
hem "player" hem "referee" kutusu verebiliyordu (test setinde **43 kez**). Ofsayt için bu tehlikelidir:
hakem fazladan bir "savunmacı" olarak sayılabilir.

Çözüm: `agnostic_nms=True` (her kişiye tek kutu). Çift kutu 43'ten **0'a** indi, karışıklık
matrisinde kaleci doğruluğu %55'ten **%93'e**, hakem doğruluğu %54'ten **%90'a** çıktı. Uygulama
(`src/detector.py`) ve değerlendirme betiği bu ayarı kullanır. Sınıf bazlı NMS sonuçlarını görmek
için `--class-nms` kullanılabilir. **Bu, raporda anlatılabilecek güzel bir "sorun → analiz → çözüm" örneğidir.**

Hakem precision'ının düşük olmasının (0.756) olası sebebi: etiketlenmemiş kişiler (4. hakem,
top toplayıcılar, yedekler) modelce "hakem" olarak bulunuyor ve yanlış pozitif sayılıyor.

| Metrik | Anlamı |
|---|---|
| Precision | Modelin "oyuncu" dediklerinin yüzde kaçı gerçekten oyuncu |
| Recall | Gerçek oyuncuların yüzde kaçını buldu (ofsayt için kritik: kaçan savunmacı = yanlış çizgi) |
| mAP@0.5 | Kutu %50 örtüşmeyle doğru sayıldığında ortalama başarı |
| mAP@0.5:0.95 | Daha sıkı ölçüt: kutunun ne kadar hassas oturduğu (ayak noktası için önemli) |

---

## 6. ADIM 4 — Uygulamayı çalıştır

```bash
streamlit run app.py
```
Tarayıcıda `http://localhost:8501` açılır.

| Adım | Yapılacak | Arka planda |
|---|---|---|
| 1 Kare | Video yükle (veya dosya yolunu yaz), slider ve ±1 / ±10 butonlarıyla pas anını bul | `cv2.CAP_PROP_POS_FRAMES` ile kare okunur |
| 2 Top | Topun yere değdiği noktaya tıkla | Ekran koordinatı orijinal çözünürlüğe ölçeklenir |
| 3 Tespit | "Tespit et" → tabloyu kontrol et, yanlış takımı düzelt, yanlış tespiti "Yok say" yap, eksik oyuncuyu ekleme moduyla ekle | YOLOv8 + KMeans |
| 4 Kalibrasyon | Önce kalenin görüntüde hangi tarafta olduğunu seç. Sağdaki şemada kırmızı halkayla gösterilen noktayı görüntüde bul ve tıkla; liste kendiliğinden sonraki noktaya geçer. 5-6 nokta. Sarı şablon gerçek çizgilerle örtüşmeli | `cv2.findHomography` (8'den az noktada tüm noktalarla en küçük kareler) + tutarlılık kontrolleri |
| 5 Sonuç | Hücum yönü, hücum eden takım, hücumcu ID → karar | `src/offside.py` |

**Kalibrasyonda isimlendirme:** SOL = ekranda soldaki kale (x = 0), SAĞ = sağdaki (x = 105),
**yakın** = kameraya yakın, yani ekranda alttaki yan çizgi (y = 0), **uzak** = karşı yan çizgi (y = 68).

**İyi kalibrasyon için (rapor 8.3):**
- Noktalar birbirinden uzak olsun ve oyuncuların olduğu bölgeyi çevrelesin.
- 4 değil 5-6 nokta kullan. 4 noktada "yeniden izdüşüm hatası" hep 0 çıkar ve bir şey ölçmez;
  5. noktadan sonra gerçek hata görünür.
- Hata 5 pikselden büyükse en hatalı noktayı silip yeniden tıkla.
- Kuşbakışı mini sahaya bak: oyuncular mantıklı yerlerde mi?
- **Köşe bayrağı ve yan çizgi noktalarından kaçın.** IFAB saha genişliğinin 64-75 m arasında
  olmasına izin verir; şablon 68 m varsayar. Ceza sahası, altıpas, penaltı noktası ve ceza yayı
  ise **her sahada aynı ölçüdedir**. Örnek görselde köşe bayrağı kullanıldığında ceza yayı 20-30 px
  kaydı; yerine yayın ceza sahası çizgisiyle kesiştiği noktalar kullanılınca tam oturdu.
- **Hiçbir 3 nokta sahada aynı doğru üzerinde olmamalı** (en az 4 nokta için). Ceza sahasının iki ön
  köşesi ve ceza yayı kesişimleri hep aynı doğrudadır (x = 16.5 m); bunlarla matris belirlenemez.
  Uygulama bunu yakalar ve hesap yapmaz.
- **Sadece altıpas yetmez.** Ölçüm (testgörsel2, ±2 px tıklama hatası, oyuncu bölgesi):

  | Noktalar | Ortalama x hatası | %95 |
  |---|---|---|
  | Sadece altıpas (4 köşe) | 0.76 m | 2.63 m |
  | Altıpas + penaltı noktası | 0.58 m | 1.90 m |
  | Ceza sahası 3 köşe + altıpas 2 | **0.11 m** | **0.32 m** |

  Görünmeyen nokta için **"⏭ Görünmüyor, atla"** butonu var. Uygulama ayrıca kaleye en yakın
  oyuncular için **konum belirsizliğini** (±m) gösterir; sınırda bandı en az bu kadar olmalı.
- 5+ noktada kalan ~3-6 px hata normaldir: yayın kameralarının hafif **lens bozulması** vardır
  ve homography bunu modelleyemez (düz çizgiyi düz kabul eder). Raporda sınırlılık olarak yaz.

### 6.1 Örnek deneme: `testgörsel2.jpg` (Nottingham Forest - Arsenal)

| | |
|---|---|
| Tespit | 16 kişi; KMeans takımları doğru ayırdı (Forest kırmızı / Arsenal krem) |
| Kalibrasyon | 6 nokta (ceza sahası 3 köşe, altıpas kale çizgisi köşesi, ceza yayı 2 kesişim), hata 5.5 px |
| Hücum | Arsenal, sol kaleye; top karede net görünmediği için top kuralı kullanılmadı |
| Sondan 2. savunmacı | Forest #14, x = 13.9 m (kaleci x = 3.2 m) |
| Hücumcu 12 | x = 13.9 m → fark **−0.05 m** → **SINIRDA** |

5 cm'lik fark, kalibrasyon belirsizliğinin (±0.5 m bant) çok altında. Sistem burada kesin karar
vermeyi reddeder; sınırda bandının neden gerekli olduğunu gösteren iyi bir örnek. Sonuç görselleri:
`results/demo_testgorsel2_sonuc.png`, `results/demo_testgorsel2_kusbakisi.png`.

---

## 7. Teori: Kodun kalbi

### 7.1 Homography (`src/calibration.py`)
Görüntüdeki bir zemin noktası (u, v) ile sahadaki karşılığı (X, Y) şu bağıntıyla ilişkilidir:

```
s·[X, Y, 1]ᵀ = H · [u, v, 1]ᵀ        H: 3×3, 8 serbestlik derecesi
```
Her nokta eşleşmesi 2 denklem verir, bu yüzden en az 4 nokta gerekir. OpenCV bunu DLT
(Direct Linear Transform) ile çözer. 4'ten fazla noktada **RANSAC** hatalı tıklamaları ayıklar.

**Önemli kısıt:** H yalnızca **zemin düzlemi** için geçerlidir. Bu yüzden oyuncunun ayağı
(kutunun alt orta noktası) kullanılır, kafası değil. Havadaki top yanlış konumlanır.

### 7.2 Ofsayt kuralı (`src/offside.py`)
Hücum yönünden bağımsız çalışmak için her x değeri "ilerilik" değerine çevrilir:
`u = x` (sağa hücum) veya `u = 105 − x` (sola hücum). Sonra:

```
savunmacılar (kaleci dahil) u'ya göre büyükten küçüğe sıralanır
sondan_ikinci = sıralı[1]
çizgi = max(sondan_ikinci, top, 52.5)      ← orta çizgi: kendi yarı sahasında ofsayt olmaz
fark  = hücumcu − çizgi
```

**Saha dışındaki kişiler otomatik olarak hesap dışı bırakılır.** Model sahadaki herkesi tespit eder:
top toplayıcılar, yedekler, fotoğrafçılar, yan hakemler. Formaları savunan takıma benziyorsa
"savunmacı" sayılırlar. `testgörseli.jpg`'de kale arkasındaki bir görevli (x = 109 m) savunmacı
sayıldı, ofsayt çizgisi kalecinin hizasına (99 m) kaydı ve 1.6 m önde olan #16 "OFSAYT DEĞİL" çıktı.
Artık kalibrasyondan sonra ayağı saha dışına (±1 m pay) düşen herkes çıkarılıyor; arayüz bunları
listeler ve istenirse geri eklenebilir. Sonuç ekranında ayrıca **tüm hücumcuların karar tablosu**
var ve varsayılan hücumcu **en ileride olan** oyuncudur.

### 7.3 Sınırda bandı (rapor 9.2)
| fark | karar |
|---|---|
| > +bant | OFSAYT |
| < −bant | OFSAYT DEĞİL |
| arada | SINIRDA |

Bant genişliğini rastgele seçme: **duyarlılık analizindeki σ değerinden** belirle
(örneğin bant ≈ 2σ). Bu, raporun bilimsel değerini artırır.

### 7.4 Takım ayrımı (`src/detector.py`)
Model yalnızca `player / goalkeeper / referee` bilir. Takımlar **forma rengiyle** ayrılır:
1. Kutunun gövde bölgesi kırpılır (baş, şort ve bacaklar hariç).
2. Çim pikselleri (HSV'de yeşil) atılır.
3. Lab renk uzayında yalnızca **a, b** (renk tonu) kanallarının medyanı alınır. **L (parlaklık)
   bilerek atılır.** Testte L kullanıldığında KMeans oyuncuları formaya göre değil,
   **güneş/gölgeye göre** ayırdı. Bunu raporda "karşılaşılan sorun ve çözümü" olarak anlatabilirsin.
4. KMeans (k = 2) ile iki küme oluşturulur.

Kaleciler kümelenmez (formaları farklıdır); hangi yarı sahada olduklarına göre savunan
takıma atanırlar. Hakemler hesaba katılmaz.

---

## 8. ADIM 5-7 — Değerlendirme (rapor Bölüm 11)

### 8.1 Test pozisyonlarını topla
10-15 farklı ofsayt pozisyonu klibi bul (yarısı ofsayt, yarısı değil). Her biri için
arayüzde 5 adımı tamamla, sonra **"💾 Sahneyi kaydet"** bölümünde maçtaki **resmi kararı**
seçip kaydet. `scenes/` klasörüne JSON + JPG olarak kaydedilir.

### 8.2 Karar doğruluğu
```bash
python scripts/evaluate_decisions.py
```
Karışıklık matrisi, doğruluk ve **bant taraması** üretir: bant 0 / 0.25 / 0.5 / 0.75 / 1.0 m
iken kaç kararın kesin verildiği ve bunların ne kadar doğru olduğu. Bant büyüdükçe kesin
kararların doğruluğu artar ama "SINIRDA" oranı da artar. Bu ödünleşimi grafikle göster.

### 8.3 Duyarlılık analizi
```bash
python scripts/sensitivity.py --trials 500
python scripts/sensitivity.py --foot-noise 3     # ayak noktası belirsizliği de eklenir
```
Her sahnede kalibrasyon tıklamalarına ±3, ±5, ±10 px rastgele gürültü ekler, 500 kez
yeniden hesaplar. Rapora gidecek cümle şu şekilde olur:
*"±5 px tıklama hatasında ofsayt farkının standart sapması 0.05 m'dir ve kararlar %100 aynı
kalmaktadır; ±10 px'te σ = 0.15 m'ye çıkar ve sınırdaki pozisyonların %17'sinde karar değişir."*
(Bu sayılar sentetik test sahnesinden alınmıştır; gerçek sahnelerle yeniden üret.)

---

## 9. Testler

```bash
python tests/test_core.py     # geometri ve kurallar (bilinen cevaplı sentetik sahne)
python tests/test_app.py      # arayüzün 5 adımı hatasız çalışıyor mu
```
`test_core.py` sahte bir kamera kurar, oyuncuları sahada **bilinen** metre konumlarına koyar
ve sistemin bu konumları doğru geri hesaplayıp doğru kararı verdiğini doğrular. Jüri
"Geometrinin doğru olduğunu nasıl biliyorsun?" diye sorarsa cevabın bu testtir.

---

## 10. Sınırlılıklar (raporda dürüstçe yaz)

1. **Ayak noktası yaklaşımı:** Kutunun alt ortası her zaman kaleye en yakın vücut bölümü değildir.
   Arayüzdeki "alt kenarın kaleye en yakın ucu" seçeneğiyle iki yöntemin farkını göster.
2. **Zemin düzlemi varsayımı:** Havadaki top veya yerden kalkmış ayak hatalı konumlanır.
3. **Pas anı:** 25 FPS'de bir kare 40 ms'dir; oyuncular bu sürede 20-30 cm yer değiştirir.
4. **Takım karışması:** Gölge, benzer forma renkleri. Arayüzde elle düzeltilebilir.
5. **Kadraj:** Kaleci görünmüyorsa sondan ikinci savunmacı yanlış belirlenir (arayüz uyarır).
6. **Tek kamera, akademik prototip:** Gerçek VAR'ın çok kameralı 3B altyapısıyla karşılaştırılamaz.

---

## 11. Sık karşılaşılan sorunlar

| Sorun | Çözüm |
|---|---|
| `torch.cuda.is_available()` False | `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| Eğitimde `CUDA out of memory` | `python scripts/train.py --batch 2` |
| `Zip içinde görsel bulunamadı` | CVAT export'ta "Save images" işaretli değil |
| Kalibrasyon şablonu kaymış | Nokta ismi ile tıklanan yer uyuşmuyor; "yakın/uzak" ve "SOL/SAĞ" tanımını kontrol et |
| Takımlar yanlış ayrılmış | Adım 3 tablosunda "Takım" sütunundan düzelt |
| Windows'ta yol hatası | Dosya yollarını tırnak içinde yaz (klasör adında boşluk ve Türkçe karakter var) |
