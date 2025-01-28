import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

print(df.info())

#veri tipi düzenleme
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


# Nümerik değişkenlerin minimum, maximum ve çeyreklik değerlerinin bulunması

columns = df.select_dtypes(include=[np.number]).columns
min_values = []
max_values = []
for column in columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    min_deger = Q1 - 1.5 * IQR
    max_deger = Q3 + 1.5 * IQR
    min_values.append(min_deger)
    max_values.append(max_deger)
    print(f"Column: {column}, min: {min_deger}, max: { max_deger}")
    
    
#Aykırı değerlerin temizlenmesi
for i, column in enumerate(columns):
    df = df[(df[column] >= min_values[i]) & (df[column] <= max_values[i])]    
    
print(df.info())    
df.describe()

df.to_csv('data_temizlenmis.csv', index=False)