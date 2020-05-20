import os
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

GISEMENT_PATH = "C:/Users/khaled/IA/projet_ia_khaled/resources"
GISEMENT_URL = GISEMENT_PATH + "/gisementLearn.csv"

#chargement des données a partir de Pandas

def load_gisement_data(gisement_path=GISEMENT_PATH):
    csv_path = os.path.join(gisement_path, "gisementLearn.csv")
    return pd.read_csv(csv_path)
def load_gisement_data2(gisement_path=GISEMENT_PATH):
    csv_path = os.path.join(gisement_path, "gisementTestNolabel.csv")
    return pd.read_csv(csv_path)

#affichage des données
    

gisement = load_gisement_data()
print(gisement.head())
print(gisement["AGE"].value_counts())
corr_matrix = gisement.corr()
corr_matrix["NBR_SEISM"].sort_values(ascending=False)

#fonction permettant de changer les strings en entiers

def encode_target(df, target_column,name):

    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[name] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


gisement21, targets = encode_target(gisement, "ROCK","ROCK2")
gisement22, targets = encode_target(gisement21, "AGE","AGE2")
gisement2, targets = encode_target(gisement22, "OR","Target")

del gisement2["AGE"]
del gisement2["ROCK"]
del gisement2["OR"]
for k in range(2,26):
    A=[ 0.8181818181818182,0.8409090909090909,0.8333333333333334,0.8409090909090909,0.8181818181818182,0.8181818181818182,0.8181818181818182,0.8409090909090909,0.8333333333333334,0.8181818181818182,0.8257575757575758,0.8257575757575758,0.8560606060606061,0.8333333333333334,0.8787878787878788,0.8560606060606061,0.8333333333333334,0.8560606060606061,0.8409090909090909,0.8636363636363636,0.8636363636363636,0.8636363636363636,0.8636363636363636,0.8636363636363636]
    B=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
  
    gisement_split = gisement2[:1500]

    gisement_split_2 = gisement2[1501:]

    y_test=gisement_split_2["Target"]

    y_train=gisement_split["Target"]

    features_test = list(gisement_split_2.columns[:22])

    features_train = list(gisement_split.columns[:22])

    X_train = gisement_split[features_train]

    X_test = gisement_split_2[features_test]

    dt_train = DecisionTreeClassifier(min_samples_split= k)

    dt_train.fit(X_train, y_train)

    predict_test=dt_train.predict(X_test)

    q= accuracy_score(y_test,predict_test, normalize=True)
    
   
plt.figure(figsize=(50,20))
plt.plot(B,A)
plt.ylabel('accuracy',fontsize=30)   
plt.xlabel('min_sample_split',fontsize=30)   
plt.show 
    
features = list(gisement2.columns[:22])

print(gisement2.head())
#print(gisement2.tail())

y = gisement2["Target"]
X = gisement2[features]
dt = DecisionTreeClassifier(min_samples_split=16)
dt.fit(X, y)
plt.figure(figsize=(50,20))
tree.plot_tree(dt)
plt.show

test = load_gisement_data2()
gisement21, targets = encode_target(test, "ROCK","ROCK2")
test2, targets = encode_target(gisement21, "AGE","AGE2")
del test2["AGE"]
del test2["ROCK"]

predict = dt.predict(test2)

test2['OR']=predict
print(test2.head())

map_dict = {0: "STERILE", 1: "GISEMENT"}
test2.OR = test2.OR.map(map_dict)


print(test2.head())

result = load_gisement_data2()
result['OR']=test2['OR']

result.to_csv('C:/Users/khaled/IA/projet_ia_khaled/resources/gisementTest.txt')

print(result.head())

