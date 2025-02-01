import numpy as np
import pandas as pd 
import statsmodels.api as sm # type: ignore
import statsmodels.formula.api as smf # type: ignore
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf # type: ignore
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier # type: ignore
from lightgbm import LGBMClassifier # type: ignore
from catboost import CatBoostClassifier # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn import preprocessing



#lojistik regresyon(doğrusal öğrenir doğrusal olmayan bir sonuç verir, sigmoit fonks(olasılık değeri verir))
vs=pd.read_csv(r"C:\Users\esine\OneDrive\Desktop\data.csv").copy()
vs=vs.dropna()
print(vs.head())

print(vs.info())
print(vs.value_counts("Churn"))

print(vs.describe())

columns_to_drop = ['CustomerID', 'Gender', 'Age', 'Contract Length', 'Subscription Type']#Gereksiz Sütunları Çıkarma
df = vs.drop(columns=columns_to_drop, errors='ignore')


if 'Churn' in df.columns:# Kategorik Veriyi Sayısal Hale Getirdim churn sütunu varsa Label Encoding
    label_encoder = LabelEncoder()
    df['Churn'] = label_encoder.fit_transform(df['Churn'])


y = vs["Churn"]
X = vs.drop(["Churn"], axis=1)

for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

columns_to_drop = ['CustomerID', 'Gender', 'Age', 'Contract Length', 'Subscription Type']#Gereksiz Sütunları Çıkarma
df = vs.drop(columns=columns_to_drop, errors='ignore')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

scaler = preprocessing.MinMaxScaler((-1,1))
scaler.fit(X)
XX_train = scaler.transform(X_train.values)
XX_test  = scaler.transform(X_test.values)
YY_train = y_train.values 
YY_test  = y_test.values

loj= sm.Logit(y_train,X_train)#statsmodels kütüphanesi kullanılarak lojistik regresyon modelini kurdum. Y bağımlı X bağımsız değişken olduğu için önce y prmetresi
loj_model = LogisticRegression(max_iter=1000, warm_start=True, solver='lbfgs', random_state=42)
loj_model= loj.fit()#Modeli eğittim
print(loj_model.summary())

y_pred= (loj_model.predict(X_test) >= 0.5).astype(int)#0.5 eşik değer BURAYA BAK
cm = confusion_matrix(y_test, y_pred)
print("Karışıklık Matrisi:\n", cm)#sınıflandırma problemindeki tahmin sonuçlarının özetini sunan tablomuz


#matris görsel
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("conf matrix")
plt.ylabel("T Label")
plt.xlabel("Predict Label")
plt.show()


print(classification_report(y_test, y_pred))#sonuç


print("test doğruluk oranı:")
print(accuracy_score(y_test, y_pred))#doğruluk oranı yazdırma
print("bitti")

train_accuracy = accuracy_score(y_train, (loj_model.predict(X_train) >= 0.5).astype(int))
print(f"Eğitim Doğruluğu: {train_accuracy:.4f}")


probabilities = loj_model.predict(X_test)[:5]
predictions = (probabilities >= 0.5).astype(int)  #sınıf tahmini

print("Olasılıklar:\n", probabilities)
print("Tahminler:\n", predictions)

#Eğitim setindeki churn oranları
train_cr_oran = pd.Series(y_train).value_counts(normalize=True)
train_cr_sonuc = train_cr_oran * 100

#Test setindeki churn oranları
test_cr_oran = pd.Series(y_test).value_counts(normalize=True)
test_cr_sonuc = test_cr_oran * 100

print("Eğitim Setindeki Churn Oranları:\n", train_cr_sonuc)
print("\nTest Setindeki Churn Oranları:\n", test_cr_sonuc)


# Farklı eşik değerleri için bir liste
thresholds = np.linspace(0, 1, 100)

# Boş bir liste oluşturarak accuracy değerlerini sakla
accuracies = []

# Her bir eşik değeri için accuracy
for threshold in thresholds:
    y_pred = (loj_model.predict(X_train) >= threshold).astype(int)  # statsmodels'ta predict olasılık döndürür
    accuracy = accuracy_score(y_train, y_pred)
    accuracies.append(accuracy)

# Grafik oluşturma
plt.plot(thresholds, accuracies)
plt.xlabel("Eşik Değeri")
plt.ylabel("Accuracy")
plt.title("Lojistik Regresyon Modelinin Accuracy Grafiği")
plt.grid(True)
plt.show()

# Lojistik regresyon modelini tanımla
loj_model = LogisticRegression(max_iter=1, warm_start=True, solver='lbfgs', random_state=42)

# Epoch sayısı
epochs = 10
losses = []

# Epoch bazlı model eğitimi
for epoch in range(epochs):
    loj_model.fit(X_train, y_train)  # Modeli eğit
    y_pred_proba = loj_model.predict_proba(X_train)[:, 1]  # Olasılık tahminlerini al
    loss = log_loss(y_train, y_pred_proba)  # Log loss'u hesapla
    losses.append(loss)  # Loss değerini listeye ekle
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Log loss değerlerini görselleştirme
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Eğitim Sırasındaki Log Loss Değeri")
plt.grid(True)
plt.show()

