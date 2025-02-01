import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Veriyi yükle ve ön işleme
vs = pd.read_csv(r"C:\Users\esine\OneDrive\Desktop\data.csv").copy()
vs = vs.dropna()

# Gereksiz sütunları çıkar
columns_to_drop = ['CustomerID', 'Gender', 'Age', 'Contract Length', 'Subscription Type']
df = vs.drop(columns=columns_to_drop, errors='ignore')

# Hedef değişkeni etiketleme
if 'Churn' in df.columns:
    label_encoder = LabelEncoder()
    df['Churn'] = label_encoder.fit_transform(df['Churn'])

# Özellik ve hedef değişkenlerini ayır
y = df["Churn"]
X = df.drop(["Churn"], axis=1)

# Kategorik değişkenleri sayısal hale getir
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.layers import BatchNormalization
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Modeli derle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Erken durdurma
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğit
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stopping]
)

# Tahmin yap ve performansı değerlendirme
y_pred = (model.predict(X_test) >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("Karışıklık Matrisi:\n", cm)
print(classification_report(y_test, y_pred))


# Classification Report ile Precision, Recall ve F1-Score
report = classification_report(y_test, y_pred)
print("Sınıflandırma Raporu:\n", report)

# Eğitim doğruluklarını al
train_accuracies = history.history['accuracy']

# Genel eğitim doğruluğunu hesapla
general_train_accuracy = np.mean(train_accuracies)
print(f"Genel Eğitim Doğruluğu (Ortalama): {general_train_accuracy:.4f}")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: {test_accuracy}")

# Eğitim ve doğrulama doğruluklarını görselleştirme
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.title('Eğitim ve Doğrulama Doğrulukları')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kayıp')
plt.plot(history.history['val_loss'], label='Doğrulama Kayıp')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.title('Eğitim ve Doğrulama Kayıpları')

plt.tight_layout()
plt.show()


