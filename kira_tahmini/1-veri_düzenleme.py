import numpy as np
import pandas as pd

#dosyayı içeri aktarma
df=pd.read_csv("hepsiemlak_mugla.csv")
df

#ilk 5 satır
h=df.head()

#son 5 satır
t=df.tail()

#veri bilgileri
df.info()

df.columns
"""
'img-link href':"img bağlantısı"
'photo-count': "fotoğraf sayısı"
'list-view-price':"fiyat"
list-view-date :"görüntüleme tarihi"
'left':"kiralık konut şekli"
'celly':"oda sayısı"
'celly 2': "brüt m2"
'celly 3': "bina yaşı"
'celly 4': "kat sayısı"
'eids-badge__description': "Elektronik İlan Doğrulama Sistemi"  
'he-lazy-image src: sil
'eids-badge__label': EİDS
'list-view-header: ilan başlığı
'img-wrp href': gereksiz
'he-lazy-image src 2': sil
'list-view-location':"konum"
'listing-card--owner-info__firm-name': firma adı
##'he-lazy-image src 3': sil
"""
"""
Kalacaklar:
'img-link href':"img bağlantısı"
'list-view-price':"fiyat"
'celly':"oda sayısı"
'celly 2': "brüt m2"
'celly 3': "bina yaşı"
'celly 4': "kat bilgisi"

"""

#istenmeyen verileri silme
df.drop(['photo-count','list-view-date','left','eids-badge__description', 'he-lazy-image src', 'eids-badge__label','list-view-header', 'list-view-location', 'img-wrp href','he-lazy-image src 2', 'listing-card--owner-info__firm-name','he-lazy-image src 3'],axis=1, inplace=True)
df.info()

df['img-link href'].unique()

#muğla-ula-gokova sehir-ilçe-mahalle olarak ayırma
s="https://www.hepsiemlak.com/mugla-ula-gokova-kiralik/daire/94718-2625"
s.split('/')
x=s.split('/')[3]
x.split('-')

df['konum']= df['img-link href'].apply(lambda x:x.split("/")[3])
df['şehir']= df['konum'].str.split("-").str[0]
df['ilçe']= df['konum'].str.split("-").str[1]
df['mahalle']= df['konum'].str.split("-").str[2]


df['ilçe'].unique()
df['mahalle'].unique()


# sütun isimlerinin değiştirilmesi
df.rename(columns={'img-link href'    : 'img bağlantısı',
                   'list-view-price'  : 'fiyat bilgi',
                     'celly'   : 'oda bilgisi',
                     'celly 2'   : 'brüt m2',
                     'celly 3': 'bina yaşı',
                     'celly 4': 'kat no'
                     }, inplace=True) # inplace = True dediğimiz zaman ismi değiştirilen veri otomatik olarak veri variable'a kaydedilir
df.head(2)


#oda bilgisi kolonunu düzenleme
df['oda bilgisi']=df['oda bilgisi'].apply(lambda x:x.replace('Stüdyo','1 + 0'))
df['oda bilgisi']=df['oda bilgisi'].apply(lambda x:x.replace('\n',' '))
print(df['oda bilgisi'])
df['oda sayısı']=df['oda bilgisi'].apply(lambda x:x.split('+')[0]).astype(int)
df['salon sayısı']=df['oda bilgisi'].apply(lambda x:x.split('+')[1]).astype(int)

#oda bilgisi verileri düzenlendi. Kolonu sil
df.drop(['oda bilgisi'] ,axis=1,inplace=True)
df.info()


#brüt m2 sütunun düzenlenmesi
df['brüt m2']=df['brüt m2'].apply(lambda x: x.replace('.',''))
df['alan'] = df['brüt m2'].apply(lambda x: x.split(' ')[0]).astype(int)


#brüt m2 kolonu düzenlendi sil
df.drop(['brüt m2'],axis=1, inplace=True)
df.info()


print(df['bina yaşı'].unique())
print(df['bina yaşı'].dtype)

#bina yaşı sütunun düzenlenmesi

# NaN değerinin bulunduğu satırı görüntüleyin
print(df[df['bina yaşı'].isna()])   #1 tane değer çıktı ve yaşı 0 

# veriyi string'e çevir ve eksik değerleri '0 Yaşında' ile doldur
df['bina yaşı'] = df['bina yaşı'].fillna('0 Yaşında')

#eksik verilerin giderilmesinden sonra bina yaşı sütunun düzenlenmesi
df['bina yaşı'] = df['bina yaşı'].astype(str).apply(lambda x: x.replace('Sıfır Bina', '0 Yaşında'))
df['bina yaşı'] = df['bina yaşı'].apply(lambda x: x.replace('\n', ' '))
df['yaş'] = df['bina yaşı'].apply(lambda x: int(x.split(' ')[0]) if x.lower() != 'nan' else 0)


print(df['yaş'].unique())

#bina yaşı kolonu düzenlendi sil
df.drop(['bina yaşı'],axis=1, inplace=True)

df.info() 


#kat no düzenlenmesi
print(df['kat no'].unique())


replace_dict = {
    'Kot 2': '-2. Kat',
    'Kot 1': '-1. Kat',
    'Yüksek Giriş': '1. Kat',
    'Ara Kat': '3. Kat',
    'En Üst Kat': '5. Kat',
    'Bahçe Katı': '0. Kat',
    'Yarı Bodrum': '0. Kat',
    'Bodrum': '0. Kat',
    'Kot 3': '-3. Kat',
    'Çatı Katı': '5. Kat',
    'Zemin': '0. Kat',
    'Giriş Katı': '0. Kat',
    'Villa Katı': '0. Kat',
    '21 ve üzeri': '21. Kat',
    'Bodrum ve Zemin': '0. Kat',
    'Asma Kat': '1. Kat',
    'Tripleks': '0. Kat',
    'Teras Katı': '5. Kat',
    'nan': '2. Kat',
}

df['kat no'] = df['kat no'].replace(replace_dict.keys(), replace_dict.values()).astype(str)
df['kat no'].dropna(inplace=True)
#df['kat'] = df['kat no'].apply(lambda x: x.split('.')[0]).astype(int)
df['kat'] = df['kat no'].apply(lambda x: str(x).split('.')[0] if pd.notna(x) else '0').astype(int)

print(df['kat'].unique())


#kat no sil
df.drop(['kat no'],axis=1, inplace=True)

df.info() #düzenlenmiş veri bilgisi


#fiyat sütunun düzenlenmesi
df['fiyat bilgi'].unique()

df['fiyat bilgi']= df['fiyat bilgi'].astype(str).apply(lambda x: x.replace('.', ''))
df['fiyat'] = df['fiyat bilgi'].astype(int)
print(df['fiyat'].unique())


#fiyat bilgi sil
df.drop(['fiyat bilgi'],axis=1, inplace=True)
print(df.info())

df.drop(['img bağlantısı'] ,axis=1,inplace=True)
df.drop(['konum'] ,axis=1,inplace=True)

df.to_csv("data.csv", index=False)