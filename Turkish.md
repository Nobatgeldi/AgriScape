Sentinel-2 verilerini kullanarak $100\text{ km} \times 100\text{ km}$ boyutunda bir alanda ürün sınıflandırması yapmak ve ardından bunları Unreal Engine weightmap’lerine dönüştürmek harika bir mühendislik projesidir.

Bu süreci kendi bilgisayarınızda en kararlı şekilde yürütmek için **QGIS (ücretsiz coğrafi bilgi sistemi yazılımı)** ve onun **SCP (Semi-Automatic Classification)** eklentisini kullanabilirsiniz.

Aşağıda, sıfırdan başlayarak Unreal Engine aşamasına kadar izlemeniz gereken veri akış şeması ve adım adım yol haritası özetlenmiştir.

---

### Veri Akış Şeması (Pipeline)

Projenizin genel mantığı şu veri akışı üzerinden ilerleyecektir:

```text
[CDSE Browser] ──> Sentinel-2 Verilerini İndir (3 Farklı Tarih)
       │
[QGIS] └──> 1. Adım: Her Tarih İçin NDVI Hesapla (B8 ve B4 Bantları)
       └──> 2. Adım: NDVI Katmanlarını Birleştir (Layer Stack - Çok Zamanlı Raster)
       └──> 3. Adım: Örnek Bölgeleri Çiz (Buğday, Arpa, Mısır, Üzüm Poligonları)
       └──> 4. Adım: Random Forest Algoritmasını Çalıştır (Sınıflandırılmış Harita)
       └──> 5. Adım: Ürün Katmanlarını Siyah-Beyaz Maskelere (0-1) Ayır
       │
[Unreal Engine] ──> PNG Maskelerini Landscape Modunda Katmanlara Aktar (Import)

```

---

### Nereden Başlamalısınız? (Ön Hazırlık)

1. **Gerekli Yazılım:** Bilgisayarınıza güncel bir **QGIS** sürümünü indirin ve kurun.
2. **QGIS Eklentisi:** QGIS içinden `Eklentiler > Eklentileri Yönet ve Kur` menüsüne gidin, arama kısmına **SCP** yazarak **Semi-Automatic Classification Plugin** eklentisini kurun.
3. **Veri Portalı Hesabı:** Avrupa Uzay Ajansı'nın resmi veri portalı olan **Copernicus Data Space Ecosystem** (browser.dataspace.copernicus.eu) adresine giderek ücretsiz bir üyelik oluşturun.

---

### Yapacağınız Adımların Özeti

#### Adım 1: Doğru Tarihlerde Uydu Görüntüsü İndirmek

* Portal üzerinde $100\times100\text{ km}$'lik alanınızı (AOI - Area of Interest) çizin.
* Veri tipi olarak **Sentinel-2 L2A** seçin (L2A, atmosferik düzeltmesi yapılmış, analize hazır veridir).
* Bitkilerin gelişim dönemlerini yakalamak için **aynı yıla ait** şu 3 dönemi bulutsuz olarak indirin:
1. **Nisan sonu / Mayıs başı:** Kışlık tahılların (buğday/arpa) en yeşil olduğu, mısırın henüz ekilmediği dönem.
2. **Temmuz ortası:** Buğday ve arpanın hasat edildiği (sarı/kahverengi), mısır ve üzümün en yeşil olduğu dönem.
3. **Eylül başı:** Mısırın hasat edildiği/kuruduğu, üzüm bağlarının hala yeşil kalabildiği dönem.



#### Adım 2: Çok Zamanlı NDVI Kümesi (Layer Stack) Oluşturmak

* İndirdiğiniz her görüntü klasörünün içinden **B04 (Kırmızı)** ve **B08 (Yakın Kızılötesi)** bantlarını QGIS'e aktarın (Bu bantlar 10 metre çözünürlüktedir).
* `Raster > Raster Hesaplayıcı (Raster Calculator)` aracını kullanarak her tarih için ayrı ayrı şu formülü uygulayın:

$$NDVI = \frac{B08 - B04}{B08 + B04}$$


* Elinizde 3 adet NDVI haritası (Örn: Mayıs_NDVI, Temmuz_NDVI, Eylul_NDVI) olacak.
* `Raster > Araçlar > Birleştir (Merge)` menüsünü kullanın, **"Giriş katmanlarını ayrı bantlara yerleştir"** seçeneğini işaretleyerek bu 3 NDVI'ı tek bir raster dosyasında birleştirin.

#### Adım 3: Yapay Zekayı Eğitmek (Training Data)

* SCP eklentisini açın ve yeni bir "Eğitim Girişi" (Training Input) dosyası oluşturun.
* Harita üzerinde Google Earth yardımıyla veya yerel bilginizle emin olduğunuz tarlaları bulun.
* Çokgen çizim aracıyla bu tarlaların sınırlarını çizin ve kimliklendirin:
* *Poligon 1-15 arası:* Makro Sınıf: 1 (Tarım), Sınıf: 1 (Buğday)
* *Poligon 16-30 arası:* Makro Sınıf: 1 (Tarım), Sınıf: 2 (Arpa)
* *Poligon 31-45 arası:* Makro Sınıf: 1 (Tarım), Sınıf: 3 (Mısır)
* *Poligon 46-60 arası:* Makro Sınıf: 1 (Tarım), Sınıf: 4 (Üzüm)
* *Poligon 61+: * Makro Sınıf: 2 (Tarım Dışı), Sınıf: 5 (Yollar, Yerleşim, Su)



#### Adım 4: Sınıflandırma ve Doğrulama

* SCP eklentisinin "Classification" sekmesinde algoritma olarak **Random Forest (Rastgele Orman)** seçin.
* Algoritmayı çalıştırın. Bilgisayarınızın gücüne ve alanın büyüklüğüne bağlı olarak işlem birkaç dakika veya yarım saat sürebilir.
* Çıktı olarak elinizde her pikselin 1, 2, 3, 4 veya 5 değerini aldığı, rengarenk bir tematik harita olacaktır.

#### Adım 5: Unreal Engine İçin Weightmap Çıktısı Almak

* Unreal Engine her katman için ayrı bir siyah-beyaz maske ister. Tekrar `Raster Hesaplayıcı`yı açın:
* Mısır maskesi için formül: `Sınıflandırılmış_Harita@1 = 3` (Bu işlem mısır olan pikselleri 1, diğerlerini 0 yapar).
* Üzüm maskesi için formül: `Sınıflandırılmış_Harita@1 = 4`


* Oluşturduğunuz bu siyah-beyaz raster katmanlarına sağ tıklayıp `Dışa Aktar > Farklı Kaydet` deyin. Formatı **PNG** yapın ve renk modunu **Grayscale (Gri Tonlamalı)** olarak ayarlayın.

Artık elinizde Unreal Engine'deki Landscape sistemine doğrudan besleyebileceğiniz, pikselleri birebir eşleşen 10 metre çözünürlüklü ve hatasız tarım katmanı maskeleriniz hazır olacaktır.