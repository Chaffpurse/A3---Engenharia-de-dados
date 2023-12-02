# A3---Engenharia-de-dados
Códigos do projeto A3, Anhembi Morumbi


#3.0 - Desenvolvimento 
3.1 - Importando as bibliotecas

import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import io
from sklearn.metrics import r2_score
import numpy as np
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from google.colab import drive
drive.mount('/content/drive')


3.2 - Importando Dataset

df = pd.read_csv('/content/drive/MyDrive/autoscout24-germany-dataset.csv')


3.3 - Limpando as colunas vazias. Primeiro uma contagem de quantos valores vazio temos no dataset

df1 = df
df1.isnull().sum()

mileage        0
make           0
model        143
fuel           0
gear         182
offerType      0
price          0
hp            29
year           0
dtype: int64



3.4 - Limpando os valores vazios do dataset

print("Numero de linhas antes:", df1.shape)
df1.dropna(subset=['model','gear','hp'], inplace=True)
print("Numero de linhas depois:", df1.shape)
print("\n\n\n", df1)

Numero de linhas antes: (46405, 9)
Numero de linhas depois: (46071, 9)


        mileage        make   model               fuel    gear       offerType  \
0       235000         BMW     316             Diesel  Manual            Used   
1        92800  Volkswagen    Golf           Gasoline  Manual            Used   
2       149300        SEAT    Exeo           Gasoline  Manual            Used   
3        96200     Renault  Megane           Gasoline  Manual            Used   
4       156000     Peugeot     308           Gasoline  Manual            Used   
...        ...         ...     ...                ...     ...             ...   
46400       99        Fiat     500  Electric/Gasoline  Manual  Pre-registered   
46401       99        Fiat     500  Electric/Gasoline  Manual  Pre-registered   
46402       99        Fiat     500  Electric/Gasoline  Manual  Pre-registered   
46403       99        Fiat     500  Electric/Gasoline  Manual  Pre-registered   
46404       99        Fiat     500  Electric/Gasoline  Manual  Pre-registered   

       price     hp  year  
0       6800  116.0  2011  
1       6877  122.0  2011  
2       6900  160.0  2011  
3       6950  110.0  2011  
4       6950  156.0  2011  
...      ...    ...   ...  
46400  12990   71.0  2021  
46401  12990   71.0  2021  
46402  12990   71.0  2021  
46403  12990   71.0  2021  
46404  12990   71.0  2021  

[46071 rows x 9 columns]

3.5 - Marcas mais vendidas de carros

df1['make'].value_counts()

Volkswagen    6907
Opel          4789
Ford          4410
Skoda         2874
Renault       2792
              ... 
Isuzu            1
Others           1
Zhidou           1
Brilliance       1
Alpine           1
Name: make, Length: 71, dtype: int64




3.6 - Gráficos de dispersão por ano x preço com divisão entre tipos de combustível

fig = sns.scatterplot(data=df, x='price', y='year', hue='fuel')
plt.show(fig)
























3.7 - Gráficos de progressão de ano por valor

fig = sns.scatterplot(data=df1, x='year', y='price')
plt.show(fig)



















3.8 - Plotando gráfico para entender a evolução de preços por ano

sns.lineplot(x="year", y="price",data=df1)


























3.9 - Modelos mais vendidos de carros

df1['model'].value_counts()

Golf                         1489
Corsa                        1485
Fiesta                       1273
Astra                        1190
Focus                         985
                             ... 
John Cooper Works Clubman       1
323                             1
Rodius                          1
Journey                         1
NV250                           1
Name: model, Length: 835, dtype: int64








3.10 - Plotando gráfico de distribuição para entender qual é a melhor forma de padronização dos dados 

ax = sns.distplot(df.mileage, color='g')


















3.11 - Analisando os nossos dados de PREÇO em um gráfico de dispersão

ax = sns.distplot(df.price, color='y')






















3.12 - Analisando os nossos dados de POTÊNCIA EM CAVALOS em um gráfico de dispersão

ax = sns.distplot(df.hp, color='r')





















3.13 - Analisando os nossos dados de anos em um gráfico de dispersão

ax = sns.distplot(df.year, color='g')

















3.14 - Carros vendidos por menos de 5 mil

df1[(df1['price'] < 5000)]


mileage
make
model
fuel
gear
offerType
price
hp
year


364
62302
Dacia
Sandero
Gasoline
Manual
Used
4700
73.0
2018
406
135000
smart
forTwo
Diesel
Automatic
Used
3940
54.0
2011
407
234730
Opel
Astra
Diesel
Automatic
Used
3949
160.0
2011
408
42571
Hyundai
i10
Gasoline
Manual
Used
3950
69.0
2011
409
368777
Volkswagen
Caddy
CNG
Manual
Used
3950
109.0
2011
...
...
...
...
...
...
...
...
...
...
40405
113121
Fiat
Punto
Gasoline
Manual
Used
4500
69.0
2016
40494
47000
Renault
Twizy
Electric
Automatic
Used
4700
11.0
2016
40855
116720
Mitsubishi
Space Star
Gasoline
Manual
Used
4500
71.0
2017
41301
9304
Renault
Twizy
Electric
Automatic
Used
4990
11.0
2017
41756
123893
Renault
Twingo
Gasoline
Manual
Used
2300
69.0
2017

3500 rows × 9 columns


3.15 - Colocando as principais marcas do preço em um histograma

g1=df1[df1['make'].isin(['Volkswagen','Opel', 'Ford','Skoda','Renault'])]
fig = sns.displot(data=g1, x='make',y='price')
























3.16 - Convertendo colunas do tipo “object” para “category”

df['gear'] = df['gear'].astype('category')
df['make'] = df['make'].astype('category')
df['model'] = df['model'].astype('category')
df['fuel'] = df['fuel'].astype('category')
df['offerType'] = df['offerType'].astype('category')
df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 46071 entries, 0 to 46404
Data columns (total 9 columns):
 #   Column     Non-Null Count  Dtype   
---  ------     --------------  -----   
 0   mileage    46071 non-null  int64   
 1   make       46071 non-null  category
 2   model      46071 non-null  category
 3   fuel       46071 non-null  category
 4   gear       46071 non-null  category
 5   offerType  46071 non-null  category
 6   price      46071 non-null  int64   
 7   hp         46071 non-null  float64 
 8   year       46071 non-null  int64   
dtypes: category(5), float64(1), int64(3)
memory usage: 3.1 MB


3.17 - Transformando nossos dados categóricos em numéricos

dfdum = pd.get_dummies(df)
dfdum = dfdum.drop('price', axis=1, inplace=False)


3.18 - Criando X e Y para nosso modelo
Separando o preço do Data Frame para a variável dependente (y)

y = df['price']
X = dfdum


3.19 - Separando bases de treino e teste

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.30, random_state=25)


3.20 - Treinando nosso modelo

regressor = lm.LinearRegression()
regressor.fit(X_train, y_train)


3.21 - Testando nosso modelo

y_pred = regressor.predict(X_test)


result = pd.DataFrame(y_pred)
result



0
5949.390498
1
12406.709816
2
42207.577998
3
7165.018262
4
18355.404976
...
...
13817
3960.017651
13818
4083.605862
13819
3138.136363
13820
18165.490564
13821
11435.445975

13822 rows × 1 columns


3.22 - Verificando o score quadrático do nosso modelo

r2_score(y_test, y_pred)

0.7494508707160606








3.23 - Importando SQL e conectando ao banco

import sqlite3


con = sqlite3.connect('/content/drive/MyDrive/autoscout24-germany-dataset.csv')



3.24 - Criando o cursor

cur = con.cursor()



3.25 - Criando uma table

c = con.cursor()
c.execute('''
           CREATE TABLE carros
          (mileage, make, model, fuel, gear, offerType, price, hp, year)
          ''')


#Salvando alterações


con.commit()



3.26 - Buscando registros na tabela

cur.execute('SELECT * FROM carros')




3.27 - Código para visualizar os resultados na forma de Data Frame
import pandas as pd


resultados = cur.fetchall()
resultados = pd.DataFrame(resultados)
display(resultados)
