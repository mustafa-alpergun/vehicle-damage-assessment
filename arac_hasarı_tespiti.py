import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 1. AYARLAR VE DOSYA YOLLARI
TRAIN_PATH = r'C:\Users\muham\Downloads\archive (8)\data3a\training'
VAL_PATH = r'C:\Users\muham\Downloads\archive (8)\data3a\validation'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. VERİ YÜKLEME VE ÖN İŞLEME
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

print("Eğitim verileri yükleniyor...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Doğrulama verileri yükleniyor...")
val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Sınıfları kontrol et: {'01-minor': 0, '02-moderate': 1, '03-severe': 2}
print(f"Sınıf İndeksleri: {train_generator.class_indices}")

# 3. MODEL MİMARİSİ (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax') # 3 Sınıf için çıkış
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. EĞİTİM (TRAINING)
print("Model eğitimi başlıyor...")
history = model.fit(train_generator, validation_data=val_generator, epochs=5)

# 5. HASAR VE KULLANILABİLİRLİK ANALİZ FONKSİYONU
def hasar_analizi(resim_yolu):
    if not os.path.exists(resim_yolu):
        print("HATA: Belirtilen resim dosyası bulunamadı.")
        return

    # Resmi modele hazırla
    img = image.load_img(resim_yolu, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin yap
    tahmin = model.predict(img_array)
    sinif_indeksi = np.argmax(tahmin)
    
    print(f"\n--- ANALİZ SONUCU ({resim_yolu}) ---")
    
    if sinif_indeksi == 0: # 01-minor
        print("Hasar Seviyesi: HAFİF (Minor)")
        print("Tahmini Hasar Oranı: %0 - %30")
        print("DURUM: ARAÇ KULLANILABİLİR.")
        
    elif sinif_indeksi == 1: # 02-moderate
        print("Hasar Seviyesi: ORTA (Moderate)")
        print("Tahmini Hasar Oranı: %30 - %60")
        print("DURUM: TAMİR GEREKLİ (Trafiğe Çıkamaz).")
        
    else: # 03-severe (2. indeks)
        print("Hasar Seviyesi: AĞIR (Severe)")
        print("Tahmini Hasar Oranı: %60 - %100")
        print("DURUM: PERT (KULLANILAMAZ).")

hasar_analizi(r'C:\Users\muham\Downloads\ChatGPT Image 15 Şub 2026 16_07_19.png')