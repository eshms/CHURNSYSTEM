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
from sklearn.preprocessing import StandardScaler  
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve

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

# Veriyi eğitim, doğrulama ve test setlerine ayırma
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("GÖZLEMMMMMMM:")
print(X_train_scaled[0:5])

#gözetimli öğrenme
#MLPClassifier(Multi-Layer Perceptron Classifier) modeli oluşturdum
mlp = MLPClassifier(
    hidden_layer_sizes=(400, 100,100),  # İki gizli katman: 200 nöron ve 100 nöron
    max_iter=100000,
    alpha=1e-7,
    learning_rate_init=0.0001,
    early_stopping=False,
    random_state=42
)

#train verisi eğitimde kullandım
mlp.fit(X_train_scaled, y_train)

# Tahmin yap ve performansı değerlendirme
y_pred = (mlp.predict(X_test) >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("Karışıklık Matrisi:\n", cm)
print(classification_report(y_test, y_pred))

# Classification Report ile Precision, Recall ve F1-Score
report = classification_report(y_test, y_pred)
print("Sınıflandırma Raporu:\n", report)

train_accuracy = accuracy_score(y_train, (mlp.predict(X_train) >= 0.5).astype(int))
print(f"Eğitim Doğruluğu: {train_accuracy:.4f}")

#Test verisi tahmin ve sonuç için kullandıım
y_pred = mlp.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

#y_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]#ROC(Receiver Operating Characteristic) İLE GRAFİK ÇİZDİRDİM
#fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
#plt.plot(fpr, tpr, label='MLP (AUC = %0.2f)' % roc_auc_score(y_test, y_pred_proba))
#plt.plot([0, 1], [0, 1], 'r--')
#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.title("ROC Curve")
#plt.legend()
#plt.show()

#derin öğrenme modelleri

#y_train,y_test churn yapan yapmayan oranlarını incele BUNA BAKTIM(126-131. SATIRLAR)
#model eğitildekten sonra x_test vererek elde edilen sonucu y_test ile kıyasla(satır 93-94)

#Eğitim setindeki churn oranları
train_cr_oran = pd.Series(y_train).value_counts(normalize=True)
train_cr_sonuc = train_cr_oran * 100


#Test setindeki churn oranları
test_cr_oran = pd.Series(y_test).value_counts(normalize=True)
test_cr_sonuc = test_cr_oran * 100

print("Eğitim Setindeki Churn Oranları:\n", train_cr_sonuc)
print("\nTest Setindeki Churn Oranları:\n", test_cr_sonuc)


# Eğitim sürecinde accuracy değerlerini kaydetmek için bir liste oluşturun
train_acc_history = []

# Eğitim döngüsü içerisinde her epoch sonunda accuracy değerini kaydedin
for epoch in range(500):  # Örnek olarak 100 epoch
    mlp.partial_fit(X_train_scaled, y_train)
    y_pred_train = mlp.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_pred_train)
    train_acc_history.append(train_acc)

# Accuracy eğrisini çizdirin
plt.plot(range(1, 501), train_acc_history, label='Eğitim Verisi')
plt.title('Accuracy Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
