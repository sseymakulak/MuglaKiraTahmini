import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df = pd.read_csv('data_temizlenmis.csv')
print(df.info())

df['şehir'] = df['şehir'].astype('category')
df['ilçe'] = df['ilçe'].astype('category')
df['mahalle'] = df['mahalle'].astype('category')
df['oda sayısı'] = df['oda sayısı'].astype('int64')
df['salon sayısı'] = df['salon sayısı'].astype('int64')
df['alan'] = df['alan'].astype('int64')
df['yaş'] = df['yaş'].astype('int64')
df['kat'] = df['kat'].astype('int64')
df['fiyat'] = df['fiyat'].astype('int64')

df.info()

kategorik_ozellik = ['şehir', 'ilçe', 'mahalle']
numerik_ozellik = ['oda sayısı', 'salon sayısı', 'alan', 'yaş', 'kat']


full_pipeline = ColumnTransformer([
    ('numerik', StandardScaler(), numerik_ozellik),
    ('kategorik', OneHotEncoder(handle_unknown='ignore'), kategorik_ozellik)
])

X = df.drop('fiyat', axis=1)
y = df['fiyat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('hazırlama', full_pipeline),
    ('model', LinearRegression())
])

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

"""
0.5 in altına inerse senin modelin o kadar başarısız ki eğer ki 
bu kira değerlerinin senin modelinden bir tahmini değil de 
ortalmalarını kullansaydın daha başarılı olurdu demek  
"""

feature_importances = model.named_steps['model'].coef_
print(len(feature_importances))
print(feature_importances)

print("Numerik Özellik")
for i in range(len(numerik_ozellik)):
    print(numerik_ozellik[i], feature_importances[i])


print("Kategorik Özellik")
for i in range(len(kategorik_ozellik)):
    for j in range(len(model.named_steps['hazırlama'].transformers_[1][1].categories_[i])):
        print(model.named_steps['hazırlama'].transformers_[1][1].categories_[i][j], feature_importances[len(numerik_ozellik) + j])


new_data = pd.DataFrame({
    'şehir': ['muğla'],
    'ilçe': ['fethiye'],
    'mahalle': ['oludeniz'],
    'oda sayısı': [4],
    'salon sayısı': [1],
    'alan': [100],
    'yaş': [4],
    'kat': [3]
})

print(model.predict(new_data))


print(df[(df['şehir'] == 'muğla') & (df['ilçe'] == 'fethiye') & (df['mahalle'] == 'patlangıc')])


def tolerance_r2(y_true, y_pred, tolerance):
    residuals = y_pred - y_true
    residuals[np.abs(residuals) <= tolerance] = 0
    ssr = np.sum(residuals**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ssr / sst)

def tolerance_percentage_r2(y_true, y_pred, tolerance):
    residuals = y_pred - y_true
    residuals[(np.abs(residuals) / y_true) <= tolerance] = 0
    ssr = np.sum(residuals**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ssr / sst)

print("r2 score:")
print(r2_score(y_test, y_pred))
print("tolerance_r2:")
print(tolerance_r2(y_test, y_pred, 10000))
print("tolerance_percentage_r2")
print(tolerance_percentage_r2(y_test, y_pred, 0.50))
