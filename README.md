Ağ Yönlendirme Optimizasyonu – GA ve Q-Learning

Bu proje, ağ yönlendirme problemini çözmek amacıyla Genetik Algoritma (GA) ve Q-Learning algoritmalarının karşılaştırmalı olarak incelenmesini amaçlamaktadır. Kullanıcıdan alınan kaynak (S) ve hedef (D) düğümleri, talep miktarı ve algoritma ağırlıklarına göre en uygun yol hesaplanmaktadır.

Proje, Tkinter tabanlı grafiksel kullanıcı arayüzü ile desteklenmiş olup dinamik görselleştirme ve asenkron algoritma çalıştırma özellikleri sunmaktadır.

-----------------------------------------------------------------------------------------------------

Kullanılan Algoritmalar

Genetik Algoritma (GA)

Q-Learning
-----------------------------------------------------------------------------------------------------

Gereksinimler

Gerekli Python kütüphaneleri requirements.txt dosyasında listelenmiştir.

Kurulum için:

pip install -r requirements.txt
-----------------------------------------------------------------------------------------------------

Seed Bilgisi

Deneylerin tekrarlanabilirliğini sağlamak amacıyla rastgelelik içeren işlemlerde sabit seed değeri kullanılmıştır.

random.seed(42)
numpy.random.seed(42)
-----------------------------------------------------------------------------------------------------

Çalıştırma Adımları
python main.py

-----------------------------------------------------------------------------------------------------

Kullanım

Kaynak (S) ve hedef (D) düğümlerini seçiniz

Talep miktarını giriniz

Algoritma ağırlıklarını belirleyiniz

Algoritmayı çalıştırınız

Sonuçları arayüz üzerinden inceleyiniz