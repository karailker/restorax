# RestoraX — Geliştirme Planı ve Yol Haritası (2026 Q4 / 2027)

Bu geliştirme planı, projenin mevcut durumu, açık kalan boşluklar (gap'ler), mimari iyileştirmeler ve kullanıcı talepleri göz önüne alınarak hazırlanmıştır.

---

## 1. Faz 1: Altyapı ve Veritabanı Hazırlığı (Production Readiness)
**Hedef:** SQLite tabanlı yerel geliştirme ortamından PostgreSQL üretim ortamına sorunsuz geçişi sağlamak.

- **Alembic Migration'larının Oluşturulması:**
  - `alembic revision --autogenerate -m "initial_schema"` komutu ile mevcut `Base.metadata` şemasına uygun ilk migration revizyonunun üretilmesi.
  - Test veritabanı üzerinde migration up/down testlerinin koşturulması.
- **Test Süiti İzolasyonu:**
  - `conftest.py` içinde SQLAlchemy `AsyncAdaptedQueuePool` bağlantı sızıntılarının ve event-loop çakışmalarının giderilmesi.
  - `restorax` conda ortamının otomatize edilmesi ve CI/CD pipeline içinde tam test coverage raporlaması.

---

## 2. Faz 2: Model Entegrasyonu ve Doğrulama (Model Verification)
**Hedef:** Henüz uçtan uca (e2e) çalışması doğrulanmamış veya ağırlıkları eksik olan modellerin tamamlanması.

- **Kalan Modellerin Doğrulanması:**
  - `codeformer_pp` (RestoreFormer) ve `dicface` modellerinin gerçek ağırlıklarla GPU üzerinde uçtan uca çıkarım (inference) testlerinin yapılması.
  - `upscale_a_video` ve `hdrtvdm` modellerinin mimari adaptasyonlarının ve ağırlık kaynaklarının netleştirilmesi.
- **Audio-DAG Entegrasyonu:**
  - Audio restorörlerin (Demucs, VoiceFixer, RNNoise) `AudioRestorerParams` yapısından DAG `restore` düğüm mimarisine uyarlanması için bir adaptör katmanı tasarlanması (böylece video ile ses pipeline'larının tek bir DAG üzerinde birleştirilebilmesi).

---

## 3. Faz 3: Frontend ve Kullanıcı Deneyimi Geliştirmeleri
**Hedef:** UI/UX bileşenlerinin sağlamlaştırılması ve kod kalitesinin artırılması.

- **Frontend Lint & Test Entegrasyonu:**
  - ESLint (flat config) kurallarına uygunluk denetimi ve CI/CD adımlarına `npm run lint` eklenmesi.
  - Vitest ile temel bileşen ve hook testlerinin (`useJobProgress`, pipeline builder etkileşimleri) yazılması.
- **Gerçek Zamanlı İlerleme ve Hata Yönetimi:**
  - WebSocket bağlantılarının proxy arkasındaki kararlılığının artırılması.
  - Canvas üzerinden başlatılan DAG çalıştırma (`M4` akışı) sürecinin canlı Redis + Celery kuyruğu ile uçtan uca doğrulanması.

---

## 4. Faz 4: Performans ve Güvenlik Sıkılaştırma
**Hedef:** Kurumsal dağıtımlar için güvenlik ve gözlemlenebilirlik katmanının güçlendirilmesi.

- **Kimlik Doğrulama Katmanı (Auth Layer):**
  - İsteğe bağlı / yapılandırılabilir API Key veya JWT tabanlı kimlik doğrulama mekanizmasının tasarlanması (açık kaynak / lokal kullanım varsayılan olarak serbest kalacak şekilde).
- **Benchmark ve Kıyaslama Raporları:**
  - Xiph ve Netflix test klipleriyle PSNR, SSIM, LPIPS, VMAF metriklerinin otomatik raporlanması için CLI araçlarının geliştirilmesi.
