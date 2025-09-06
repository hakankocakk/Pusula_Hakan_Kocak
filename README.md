
## Hakan Koçak - khakan924@gmail.com
## Proje Açıklaması

Bu repo, Talent Academy Data Science Intern Case Study kapsamında verilen sağlık verilerinin işlenmesi, analiz edilmesi ve makine öğrenmesi modellerine uygun hale getirilmesi için geliştirilmiş bir veri işleme ve analiz pipeline'ı içerir. Proje, veri toplama, ön işleme ve modellemeye hazır hale getirme adımlarını kapsar.

## Yapılan İşlemler (Pipeline Özeti)  

1. **Veri Toplama ve Bölme**  
	- Ham veri dosyası (`Talent_Academy_Case_DT_2025.xlsx`) okunur.  
	- Hastaların benzersiz numaralarına göre eğitim ve test setlerine ayrılır.  
	- Sonuçlar `datas/interim/train.csv` ve `datas/interim/test.csv` olarak kaydedilir.  

2. **Veri Ön İşleme**  
	 - Eksik değerler doldurulur.  
		 - Referans veri ile doldurma (ör. cinsiyet, kan grubu, alerji, kronik hastalık gibi temel bilgiler).  
		 - Mod (en sık görülen değer) ile doldurma.  
		 - "Yok", "Diğer", "Eksik" gibi özel anahtar kelimelerle doldurma.  
	 - Metin normalizasyonu:  
		 - Unicode karakterlerin düzeltilmesi ve küçük harfe çevirme.  
		 - Fuzzy matching ile benzer kelimelerin eşleştirilmesi (ör. alerji isimleri).  
	 - Kategorik değişkenlerin dönüştürülmesi:  
		 - OneHotEncoder ile tekil kategorik sütunların binary vektörlere çevrilmesi.  
		 - MultiLabelBinarizer ile birden fazla kategori içeren sütunların çoklu vektörlere çevrilmesi.  
		 - LabelEncoder (pipeline'da map olarak) ile sıralı kategorik değişkenlerin sayısal olarak kodlanması.  
	 - Sayısal değişkenlerin işlenmesi:  
		 - StandardScaler ile tüm sayısal sütunların standartlaştırılması (ortalama=0, std=1).  
		 - Aykırı değerlerin tespiti ve gerekirse işlenmesi.  
	 - Metin sütunlarının embedding işlemleri:  
		 - Bert ile metinlerin vektörleştirilmesi.  
		 - PCA ile boyut indirgeme uygulanarak daha kompakt temsiller elde edilir.  
	 - Sütun birleştirme ve yeni özellikler oluşturma:  
		 - Benzer veya tamamlayıcı sütunlar birleştirilir.  
		 - Tedavi süresi, seans sayısı gibi yeni kategorik veya sayısal özellikler eklenir.


4. **Veri Analizi ve Görselleştirme**  
	 - Jupyter notebook dosyalarında veri analizi ve görselleştirme işlemleri yapılır.  
	 - Görselleştirme örnekleri:  
		 - **Eksik Veri Analizi:** Eksik değerlerin oranı ve dağılımı için barplot, matris ve heatmap (missingno, seaborn).  
		 - **Dağılım Grafikleri:** Yaş, tedavi süresi gibi sayısal değişkenler için histogram, boxplot, violin plot.  
		 - **Kategorik Dağılımlar:** Cinsiyet, kan grubu, uyruk, bölüm gibi kategorik değişkenler için countplot ve pasta grafikler.  
		 - **Çapraz Dağılımlar:** Cinsiyet ve kan grubu, uyruk ve cinsiyet gibi iki kategorik değişkenin birlikte dağılımı için gruplu barplotlar.  
		 - **Alerji ve Kronik Hastalık Analizi:** Alerji ve kronik hastalıkların hasta sayısına göre yatay barplotları, cinsiyete göre dağılımı.  
		 - **Tanı Analizi:** En sık görülen tanıların dağılımı için yatay barplot.  
		 - **Tedavi Süresi ve Uygulama Yeri Analizi:** Tedavi süresi ve uygulama yerlerinin dağılımı için barplotlar.  
		 - **Outlier Analizi:** Sayısal değişkenlerde aykırı değerlerin tespiti için boxplotlar.  
	 - Tüm görsellerde renk paletleri ve etiketler ile anlaşılır sunum amaçlanmıştır.  

5. **Model ve Encoder Kaydı**  
	- Kullanılan encoder, scaler ve PCA modelleri `models/` klasöründe saklanır.  

6. **DVC ile Pipeline Yönetimi**  
	- Tüm adımlar DVC ile takip edilir ve yeniden üretilebilirlik sağlanır.  

## Klasör Yapısı
  
```
Pusula_Hakan_Kocak/
├── datas/
│   ├── raw/          # Ham veri dosyaları (ör: .xlsx)
│   ├── interim/      # Ara işlenmiş veriler (ör: train.csv, test.csv)
│   └── processed/    # Nihai işlenmiş veriler
├── models/           # Kaydedilen modeller ve encoderlar (.pkl)
├── notebooks/
│   ├── data-pre-processing.ipynb
│   └── exploratory-data-analysis-eda.ipynb
├── src/
│   ├── data/
│   │   └── data_collection.py
│   └── features/
│       └── data_prepocess.py
├── dvc.yaml
├── dvc.lock
├── README.md
└── docs/
```

## Kullanım

1. **Veri Toplama ve Bölme**
	- `src/data/data_collection.py` dosyası, ham veriyi okur ve eğitim/test olarak böler.
	- Çıktılar `datas/interim/` klasörüne kaydedilir.

2. **Veri Ön İşleme ve Özellik Mühendisliği**
	- `src/features/data_prepocess.py` dosyası, verileri temizler, eksik değerleri doldurur, encoding ve embedding işlemlerini uygular.
	- Sonuçlar `datas/processed/` klasörüne kaydedilir.

3. **Notebooks**
	- Veri analizi ve görselleştirme için Jupyter notebook dosyalarını kullanabilirsiniz.

4. **Modeller**
	- Eğitimde kullanılan encoder ve scaler modelleri `models/` klasöründe saklanır.

## Gereksinimler

- Python 3.8+
- Pandas, scikit-learn, joblib, sentence-transformers, rapidfuzz, seaborn, matplotlib, dvc, streamlit, unidecode

Kurulum için:
```bash
pip install -r requirements.txt
```

## Çalıştırma

1. Ham veriyi bölmek için:
	```bash
	python src/data/data_collection.py
	```
2. Veri ön işleme için:
	```bash
	python src/features/data_prepocess.py
	```
3. DVC ile pipeline'ı çalıştırmak için:
	```bash
	dvc repro
	```
