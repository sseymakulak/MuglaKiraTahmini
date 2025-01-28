#VERİLERİN SINIFLANDIRMA İLE TAHMİN EDİLMNESİ
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv('data_temizlenmis.csv')
df.info()


df['şehir'] = df['şehir'].astype('category')
df['ilçe'] = df['ilçe'].astype('category')
df['mahalle'] = df['mahalle'].astype('category')
df['oda sayısı'] = df['oda sayısı'].astype('int64')
df['salon sayısı'] = df['salon sayısı'].astype('int64')
df['alan'] = df['alan'].astype('int64')
df['yaş'] = df['yaş'].astype('int64')
df['kat'] = df['kat'].astype('int64')
df['fiyat'] = df['fiyat'].astype('int64')


print(df.info())


kategorik_ozellik = ['şehir', 'ilçe', 'mahalle']
numerik_ozellik = ['oda sayısı', 'salon sayısı', 'alan', 'yaş', 'kat']


full_pipeline = ColumnTransformer([
    ('numerik', StandardScaler(), numerik_ozellik),
    ('kategorik', OneHotEncoder(handle_unknown='ignore'), kategorik_ozellik)
])


X = df.drop('fiyat', axis=1)
y = df['fiyat']
"""
bins = [x for x in range(0, 70000, 10000)]
labels = [x for x in range(1, 7)]
print(bins)
print(labels)
"""
# = pd.cut(y, bins=bins, labels=labels)
#y = pd.cut(y, bins=bins, labels=labels).astype(int)  # Kategorik değerleri tamsayıya dönüştür
# Hedef değişkeni sınıflandırma
bins = [x for x in range(0, 100001, 10000)]  # Üst aralığı artırdık
labels = [x for x in range(1, len(bins))]
y = pd.cut(y, bins=bins, labels=labels, right=False)

"""
# NaN değerleri temizleme
print(f"NaN sayısı: {y.isna().sum()}")
df = df.loc[y.notna()]  # NaN değerleri olan satırları kaldır
X = df.drop('fiyat', axis=1)
y = y.dropna().astype(int)  # NaN olmayanları tamsayıya çevir
"""
print(y.unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Pipeline([
    ('hazırlama', full_pipeline),
    ('model', RandomForestClassifier(n_estimators=100))
])


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))






