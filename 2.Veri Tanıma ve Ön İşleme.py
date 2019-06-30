#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
 

sozluk = {'İsim':pd.Series(['Ada','Cem','Sibel','Ahmet','Mehmet','Ali','Veli',
          'Ayşe','Hüseyin','Necmi','Nalan','Namık']),
          'Meslek':pd.Series(['işçi','işçi','memur','serbest','serbest',None,None,
          'sigortacı','işsiz',None,None,'memur']),
          'Tarih':pd.Series(['11.11.2010','11.11.2010','11.11.2010','18.11.2011','18.11.2011',None,None,
          None,'11.11.2010',None,'18.11.2011','18.11.2011']),          
          'Yaş':pd.Series([21, 24, 25, 44, 31, 27, 35, 33, 42, 29, 41, 43]),
          'ÇocukSayısı':pd.Series([None, None, None, None, None, 1, 2, 0, None, None, None, None]),
          'Puan':pd.Series([89, 87, 77, 55, 70, 79, 73, 79, 54, 92, 61, 69])}
 
df = pd.DataFrame(sozluk)
df


# In[ ]:


#BÖLÜM 1: PANDAS DATAFRAME 


# In[ ]:


df.head() #ilk 5 satır


# In[ ]:


df.tail() #son 5 satır       


# In[ ]:


df.sample(5) #rassal 5 satır


# In[ ]:


df.shape #satır ve sütun sayısı


# In[ ]:


df.info() #bellek kullanımı ve veri türleri


# In[ ]:


df.describe() #basit istatistikler


# In[ ]:


df['Yaş'] #kolon seç


# In[ ]:


df['İsim'][:10] #İsim kolonundaki ilk 10 satırı seç


# In[ ]:


df[['Yaş', 'İsim']]  #Birden fazla kolon seç


# In[ ]:


df[(df['Yaş']>30) & (df['Puan']>50)]  #30 yaşından büyük olup 50 puandan yüksek alanları seç


# In[ ]:


df.sort_values('Puan', axis = 0, ascending = False) # Sıralama


# In[ ]:


df[(df['Yaş']>30) & (df['Puan']>50)].sort_values('Puan', axis=0, ascending=False)  #Filtreleme ve Sıralama birlikte


# In[ ]:


#Apply fonksiyonu kullanarak sınavdan geçip geçmediğini yeni kolon olarak ekle
def basari_durumu(puan):
    return (puan >= 70)

df['Geçti'] = df['Puan'].apply(basari_durumu)
df


# In[ ]:


#Tarih alanındaki yıl bilgisini kullanarak 'Yıl' isimli yeni bir öznitelik oluşturuyoruz
tarih = pd.to_datetime(df['Tarih'])
df['Yıl'] = tarih.dt.year
df


# In[ ]:


#Groupby kullanımı, meslek bazında puan sayıları
#df.groupby('Meslek')['Puan'].apply(lambda x: x.count())
df.groupby('Meslek').size()


# In[ ]:


#Groupby kullanımı, meslek bazında ortalamaları
df.groupby('Meslek')['Puan'].apply(lambda x: np.mean(x))


# In[ ]:


#BÖLÜM 2: BASİT İSTATİSTİK

#Puan özniteliğinin ortalaması
df['Puan'].mean()


# In[ ]:


#Sayısal tüm özniteliklerin ortalaması
df.mean(axis=0,skipna=True)


# In[ ]:


#Puan özniteliğinin medyanı
df['Puan'].median()


# In[ ]:


#Puan özniteliğinin modu
df['Puan'].mode()

#Yaş özniteliğinin modu
#df['Yaş'].mode()


# In[ ]:


#Puan özniteliğinin standart sapması
df['Puan'].std()


# In[ ]:


#Kovaryans matrisi hesapla
df.cov()


# In[ ]:


#Kovaryans matrisi hesapla
df.corr()
df.plot(x='Puan', y='Yaş', style='o')


# In[ ]:


#Korelasyon Gösterim
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


#BÖLÜM 3: ÖN İŞLEME

#1.Eksik Değer Doldurma
#Null olan öznitelikleri buluyoruz
df.isnull().sum()


# In[ ]:


#Null olan özniteliklere sahip, toplam kayıt sayısını buluyoruz
df.isnull().sum().sum()


# In[ ]:


#Eksik değer tablosu
def eksik_deger_tablosu(df): 
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns


# In[ ]:


eksik_deger_tablosu(df)


# In[ ]:


df


# In[ ]:


#%70 üzerinde null değer içeren kolonları sil
tr = len(df) * .3
df.dropna(thresh = tr, axis = 1, inplace = True)


# In[ ]:


df


# In[ ]:


#Meslek kolonundaki Null değerleri 'boş' değeri ile doldur
df['Meslek'] = df['Meslek'].fillna('boş')


# In[ ]:


df


# In[ ]:


#Tarih kolonundaki Null değerleri Tarih kolonundaki benzersiz değerlerden ilki ile doldur
print(df['Tarih'].unique()[0])
df['Tarih'] = df['Tarih'].fillna(df['Tarih'].unique()[0])


# In[ ]:


df


# In[ ]:


#2.Aykırı Değer Tespiti
import seaborn as sns
sns.boxplot(x=df['Puan'])


# In[ ]:


P = np.percentile(df.Puan, [10, 100])
P


# In[ ]:


new_df = df[(df.Puan > P[0]) & (df.Puan < P[1])]


# In[ ]:


new_df


# In[ ]:


df


# In[ ]:


#3.Veri Normalleştirme
from sklearn import preprocessing

#Puan özniteliğini normalleştirmek istiyoruz
x = df[['Puan']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['Puan2'] = pd.DataFrame(x_scaled)

df


# In[ ]:




