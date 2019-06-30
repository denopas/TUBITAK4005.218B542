#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Değişkenleri tanımla
mesaj = "Merhaba Dünya"
plaka = 35
boy = 1.85
veriBiliminiSeviyorum = (1 == True)


# In[ ]:


#mesaj değişken değerini yazdır
print("Mesaj: ", mesaj)
#Çıktı: Mesaj:  Merhaba Dünya


# In[ ]:


#veriBiliminiSeviyorum değişkenini yazdır
print(veriBiliminiSeviyorum)
#Çıktı: True


# In[ ]:


#Tek satırda birden fazla değişkene değer atama
plaka, boy  = 35, 1.85


# In[ ]:


print(type(mesaj), type(plaka), type(boy), type(veriBiliminiSeviyorum))
#Çıktı: <class 'str'> <class 'int'> <class 'float'> <class 'bool'>


# In[ ]:


renkListesi = ["sarı","kırmızı", "siyah", "beyaz", "bordo", "mavi"]


# In[ ]:


renkListesi[0] #sarı


# In[ ]:


renkListesi[5] #mavi


# In[ ]:


renkListesi[-1] #mavi


# In[ ]:


renkListesi[-3] #beyaz


# In[ ]:


renkListesi[::] #['sarı', 'kırmızı', 'siyah', 'beyaz', 'bordo', 'mavi']


# In[ ]:


renkListesi[1::1] #['kırmızı', 'siyah', 'beyaz', 'bordo', 'mavi']


# In[ ]:


renkListesi[1::2] #['kırmızı', 'beyaz', 'mavi']


# In[ ]:


renkListesi[:4] #['sarı', 'kırmızı', 'siyah', 'beyaz']


# In[ ]:


renkListesi[::-1] #['mavi', 'bordo', 'beyaz', 'siyah', 'kırmızı', 'sarı']


# In[ ]:


renkListesi[:2:-1] #['mavi', 'bordo', 'beyaz']


# In[ ]:


renkListesi[0:2] = ['sarı', 'lacivert'] 


# In[ ]:


renkListesi #['sarı', 'lacivert', 'siyah', 'beyaz', 'bordo', 'mavi']


# In[ ]:


renkListesi.append('yeşil')
renkListesi #['sarı', 'lacivert', 'siyah', 'beyaz', 'bordo', 'mavi', 'yeşil']


# In[ ]:


renkListesi.remove('siyah')
renkListesi #['sarı', 'lacivert', 'beyaz', 'bordo', 'mavi', 'yeşil']


# In[ ]:


renkListesi = renkListesi + ['turuncu']
renkListesi #['sarı', 'lacivert', 'beyaz', 'bordo', 'mavi', 'yeşil', 'turuncu']


# In[ ]:


#liste metotları


# In[ ]:


renkListesi.reverse()
renkListesi #['turuncu', 'yeşil', 'mavi', 'bordo', 'beyaz', 'lacivert', 'sarı']


# In[ ]:


renkListesi.sort()
renkListesi #['beyaz', 'bordo', 'lacivert', 'mavi', 'sarı', 'turuncu', 'turuncu', 'yeşil']


# In[ ]:


#sözlük kullanımı
notlar = {
    "0001-Ada": 83,
    "0002-Cem": 79,
    "0003-Sibel" : 82 
}

#sözlük elemanına erişim
notlar["0001-Ada"] #83


# In[ ]:


#sözlüğe yeni eleman ekle
notlar["0004-Nil"] = 99
notlar #{'0001-Ada': 83, '0002-Cem': 79, '0003-Sibel': 82, '0004-Nil': 99}


# In[ ]:


#sözlükteki elemanı sil
del(notlar["0001-Ada"])
notlar #{'0002-Cem': 79, '0003-Sibel': 82, '0004-Nil': 99}


# In[ ]:


("0004-Nil" in notlar) #True


# In[ ]:


#sözlük metotları


# In[ ]:


notlar.keys() #dict_keys(['0002-Cem', '0003-Sibel', '0004-Nil'])


# In[ ]:


notlar.items() #dict_items([('0002-Cem', 79), ('0003-Sibel', 82), ('0004-Nil', 99)])


# In[ ]:


notlar.values() #dict_values([79, 82, 99])


# In[ ]:


#Referans tür örneği (yöntem 1)
sayilar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# In[ ]:


#yanlış bir liste kopyalama yöntemi
kopya_sayilar = sayilar


# In[ ]:


#aşağıdaki değişiklik sadece kopya_sayilar nesnesinde mi yapılıyor?
kopya_sayilar[0] = 99


# In[ ]:


#her iki nesnenin de 0 indisli elemanı değişti
print("sayilar:", sayilar) #sayilar: [99, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("kopya_sayilar:",kopya_sayilar) #kopya_sayilar: [99, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# In[ ]:


#liste kopyalama için diğer yöntem (yöntem 2)
sayilar = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#legal kopyalama yöntemi
kopya_sayilar = sayilar[::] #kopya_sayilar = list(sayilar)


# In[ ]:


#her iki nesnenin de 0 indisli elemanı değişiyor mu?
kopya_sayilar[0] = 99
print("sayilar:", sayilar) #sayilar: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("kopya_sayilar:",kopya_sayilar) #kopya_sayilar: [99, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# In[ ]:


#fonksiyon tanimi
def kabarcikSiralama(dizi):
    eleman_sayisi = len(dizi)
    #Tüm elemanları dön
    for i in range(eleman_sayisi):
        for j in range(0, eleman_sayisi - i - 1):
            #Yer değiştirme
            if dizi[j] > dizi[j+1] :
                dizi[j], dizi[j+1] = dizi[j+1], dizi[j]


# In[ ]:


#fonksiyon kullanimi
sayilar = [3, 12, 9, 4, 5, 8, 11, 14, 2, 1]


# In[ ]:


kabarcikSiralama(sayilar)
for i in range(len(sayilar)):
    print ("%d" %sayilar[i]), 


# In[ ]:


import numpy as np

#Dizi oluşturma
d1 = np.array([5.5,9,10])
d2 = np.array([(3.5,8,11), (4,7,9), (2,2,1.1)], dtype=float)

#Fark alma 1. Yöntem
d3 = d2 - d1
print("Fark 1 / d3 --> ", d3)

#Fark alma 2. Yöntem
d3 = np.subtract(d1, d2) 
print("Fark 2 / d3 --> ", d3)

#d1 ve d2'yi toplayıp d1 üzerine yazma
d1 = d1 + d2
print("Toplam d1 --> ", d1)
d1


# In[ ]:


#Değeri 9'dan büyük elemanların indislerini bul
sonuc = d1 > 9
print(sonuc)


# In[ ]:



#Bulunan indisleri kullanarak, elemanları ekranra yazdır
print("9'dan büyük elemanlar -->", d1[sonuc])


# In[ ]:


#İki matrisin çarpımı
d4 = np.dot(d1, d2)
print("Çarpım d4:", d4)

#Matristen 1.sütunu çıkartma
d4 = np.delete(d4,0,1)
print("Çıkartma d4:", d4)


# In[ ]:


#2x5’lik sıfır matrisi yaratma
SifirMatrisi = np.zeros([2,5])


# In[ ]:


#Dizideki en küçük eleman bulma
print("d4 min:", np.min(d4))

#Dizideki en büyük eleman bulma
print("d4 max:", np.max(d4))

#Dizinin ortalamasını alma
print("d4 ortalama:", d4.mean())

#Dizinin toplamını bulma
print("d4 toplam:", d4.sum())

#Karekök alma
print("d4 karekök-->", np.sqrt(d4))

#Dizinin logaritmasını hesaplama
print("d4 logaritma-->", np.log(d4))

#Tranpoz alma
print("d4 transpoz:", np.transpose(d4))


# In[ ]:


list1 = [1, 2, 3, 4, 5, 6]
list2 = [i/2 for i in list1]
print(list2) #[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# In[ ]:


x, y = 5, 5
if (x > y):
    print("x > y")
elif (y > x):
    print("y > x")
else:
    print("y = x")


# In[ ]:


#while döngüsü
kosul, j = True, 0
while (kosul):
    print(j)
    j += 1
    kosul = (j != 5)
#çıktı: 0, 1, 2, 3, 4

#for döngüsü
for i in range(0, 5):
    print(i)
#çıktı: 0, 1, 2, 3, 4


# In[ ]:


#lambda fonksiyon 1
fnc = lambda x : x + 1
print(fnc(1)) #Çıktı: 2
print(fnc(fnc(1))) #Çıktı: 3

#lambda fonksiyon 2
fnc2 = lambda x, y : x + y
print(fnc2(4,7)) #Çıktı: 11
print(fnc2(4,fnc(1))) #Çıktı: 6


# In[ ]:


#lamba fonksiyon içeren fonksiyon tanımları
def fnc3(n):
  return lambda x : x ** n

fnc_kare_al = fnc3(2) #Dinamik kare alma fonksiyonu oluşturuluyor
fnc_kup_al = fnc3(3) #Dinamik küp alma fonksiyonu oluşturuluyor

print(fnc_kare_al(3))
print(fnc_kup_al(3))


# In[ ]:


import pandas as pd
data = [
        ['D1', 'Sunny','Hot', 'High', 'Weak', 'No'],
        ['D2', 'Sunny','Hot', 'High', 'Strong', 'No'],
        ['D3', 'Overcast','Hot', 'High', 'Weak', 'Yes'],
        ['D4', 'Rain','Mild', 'High', 'Weak', 'Yes'],
        ['D5', 'Rain','Cool', 'Normal', 'Weak', 'Yes'],
        ['D6', 'Rain','Cool', 'Normal', 'Strong', 'No'],
        ['D7', 'Overcast','Cool', 'Normal', 'Strong', 'Yes'],
        ['D8', 'Sunny','Mild', 'High', 'Weak', 'Yes'],
        ['D9', 'Sunny','Cool', 'Normal', 'Weak', 'No'],
        ['D10', 'Rain','Mild', 'Normal', 'Weak', 'Yes'],
        ['D11', 'Sunny','Mild', 'Normal', 'Strong', 'Yes'],
        ['D12', 'Overcast','Mild', 'High', 'Strong', 'No'],
        ['D13', 'Overcast','Hot', 'Normal', 'Weak', 'Yes'],
        ['D14', 'Rain','Mild', 'High', 'Strong', 'No'],
       ]
df = pd.DataFrame(data,columns=['day', 'outlook', 'temp', 'humidity', 'windy', 'play'])
df


# In[ ]:


df.info()


# In[ ]:


df.min()


# In[ ]:


df.max()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() 
df['outlook'] = lb.fit_transform(df['outlook']) 
df['temp'] = lb.fit_transform(df['temp'] ) 
df['humidity'] = lb.fit_transform(df['humidity'] ) 
df['windy'] = lb.fit_transform(df['windy'] )   
df['play'] = lb.fit_transform(df['play'] ) 


# In[ ]:


df


# In[ ]:


df.describe()


# In[ ]:


X = df.iloc[:,1:5] 
Y = df.iloc[:,5]


# In[ ]:


X


# In[ ]:


Y


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 100)

model = model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
    
#Accuracy değeri 
print(" ACC: %%%.2f" % (metrics.accuracy_score(Y_test, Y_pred)*100))


# In[ ]:





# In[ ]:




