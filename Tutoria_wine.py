import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


#Ver propiedades del dataset

#opcion1
dataset = load_wine()
print(dataset.DESCR)

#opcion2 dataframe
#Conver to pandas dataframe
df = pd.DataFrame(data=np.c_[dataset['data'],dataset['target']],columns=dataset['feature_names']+['target'])

#primero vamos a entender los datos antes de evaluarlos. Estructura del dataset,
#propiedades de las variables, como están distribuidos los datos, etc.

#Check data with info function
print("Propiedades del dataset - Opción 2:")
print(df.info())

# Search for missing, NA and null values)
print((df.isnull() | df.empty | df.isna()).sum())

#Frequency.
freq=df['target'].value_counts()
print("\nFrecuencia:\n" + str(freq))
#Let's check graphically.
freq.plot(kind='bar')
plt.show()

#Let's show the histograms of the variables alcohol, magnesium y color_intensity.
#Histogramas
df[['alcohol','magnesium','color_intensity']].hist()
plt.show()

#Posibilidad de hacer histogramas de variables separando los datos por clases:
fig, (ax, bx, cx) = plt.subplots(1, 3, sharey=True)
# Alcohol variable histograms.
x1 = df.loc[df.target==0, 'alcohol']
x2 = df.loc[df.target==1, 'alcohol']
x3 = df.loc[df.target==2, 'alcohol']

kwargs = dict(alpha=0.3,bins=25)

ax.hist(x1, **kwargs, color='g', label='Tipo 0')
ax.hist(x2, **kwargs, color='b', label='Tipo 1')
ax.hist(x3, **kwargs, color='r', label='Tipo 2')
ax.set(title='Frecuencia de alcohol por tipo de vino', ylabel='Frecuencia')

ax.legend();

#Magnesium histograms

x1 = df.loc[df.target==0, 'color_intensity']
x2 = df.loc[df.target==1, 'color_intensity']
x3 = df.loc[df.target==2, 'color_intensity']

bx.hist(x1, **kwargs, color='g', label='Tipo 0')
bx.hist(x2, **kwargs, color='b', label='Tipo 1')
bx.hist(x3, **kwargs, color='r', label='Tipo 2')
bx.set(title='Intensidad del color por tipo de vino', ylabel='Frecuencia')

bx.legend();
#Magnesium histograms

x1 = df.loc[df.target==0, 'magnesium']
x2 = df.loc[df.target==1, 'magnesium']
x3 = df.loc[df.target==2, 'magnesium']

cx.hist(x1, **kwargs, color='g', label='Tipo 0')
cx.hist(x2, **kwargs, color='b', label='Tipo 1')
cx.hist(x3, **kwargs, color='r', label='Tipo 2')
cx.set(title='Frecuencia de magnesio por tipo de vino', ylabel='Frecuencia')
cx.legend()
plt.show()

"""
¿Qué variable distingue mejor entre clases de vino?. El alcohol -> menos solape (mirar gráfica)
"""

#Mismas gráficas con valor medio y cálculo de deviación estándar
fig, (ax, bx, cx) = plt.subplots(1, 3, sharey=True)
# Alcohol variable histograms.
x1 = df.loc[df.target==0, 'alcohol']
x2 = df.loc[df.target==1, 'alcohol']
x3 = df.loc[df.target==2, 'alcohol']

kwargs = dict(alpha=0.3,bins=25)

ax.hist(x1, **kwargs, color='g', label='Tipo 0'+  str("{:6.2f}".format(x1.std())))
ax.hist(x2, **kwargs, color='b', label='Tipo 1'+  str("{:6.2f}".format(x2.std())))
ax.hist(x3, **kwargs, color='r', label='Tipo 2'+  str("{:6.2f}".format(x3.std())))
ax.set(title='Frecuencia de alcohol por tipo de vino', ylabel='Frecuencia')
ax.axvline(x1.mean(), color='g', linestyle='dashed', linewidth=1)
ax.axvline(x2.mean(), color='b', linestyle='dashed', linewidth=1)
ax.axvline(x3.mean(), color='r', linestyle='dashed', linewidth=1)
ax.legend();

#Magnesium histograms

x1 = df.loc[df.target==0, 'color_intensity']
x2 = df.loc[df.target==1, 'color_intensity']
x3 = df.loc[df.target==2, 'color_intensity']

bx.hist(x1, **kwargs, color='g', label='Tipo 0'+  str("{:6.2f}".format(x1.std())))
bx.hist(x2, **kwargs, color='b', label='Tipo 1'+  str("{:6.2f}".format(x2.std())))
bx.hist(x3, **kwargs, color='r', label='Tipo 2'+  str("{:6.2f}".format(x3.std())))
bx.axvline(x1.mean(), color='g', linestyle='dashed', linewidth=1)
bx.axvline(x2.mean(), color='b', linestyle='dashed', linewidth=1)
bx.axvline(x3.mean(), color='r', linestyle='dashed', linewidth=1)

bx.set(title='Intensidad del color por tipo de vino', ylabel='Frecuencia')
bx.legend();
#Magnesium histograms

x1 = df.loc[df.target==0, 'magnesium']
x2 = df.loc[df.target==1, 'magnesium']
x3 = df.loc[df.target==2, 'magnesium']

cx.hist(x1, **kwargs, color='g', label='Tipo 0'+  str("{:6.2f}".format(x1.std())))
cx.hist(x2, **kwargs, color='b', label='Tipo 1'+  str("{:6.2f}".format(x2.std())))
cx.hist(x3, **kwargs, color='r', label='Tipo 2'+  str("{:6.2f}".format(x3.std())))
cx.axvline(x1.mean(), color='g', linestyle='dashed', linewidth=1)
cx.axvline(x2.mean(), color='b', linestyle='dashed', linewidth=1)
cx.axvline(x3.mean(), color='r', linestyle='dashed', linewidth=1)
cx.set(title='Frecuencia de magnesio por tipo de vino', ylabel='Frecuencia')
cx.legend()
plt.show()

#tabla de correlación con las tres variables anteriores:
df_reducido =df[['alcohol','magnesium','color_intensity']]
print(df_reducido.corr())

#scatter plots
df_reducido=df[['alcohol','magnesium','color_intensity','target']]
sns.pairplot(df_reducido,hue='target')
plt.show()

#Carga en variables de entrada y salida por separado
X, y = load_wine(return_X_y=True, as_frame=True)

#Vamos a dividir el dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
#mostramos la cantidad de ejemplos para aprender y probar
print("Conjunto de entrenamiento (nº muestras - dimensionalidad): " + str(X_train.shape))
print("Conjunto de test (nº muestras - dimensionalidad): " + str(X_test.shape))

#Create the classifier

clf=RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train,y_train.values.ravel())
scores=cross_val_score(clf, X, y.values.ravel(),cv=5)
print("Scores: " + str(scores))

#Cálculo de métricas RandomForest

y_pred = clf.predict(X_test)

print("Cálculo de métricas para clasificador RandomForest")
print("Exactitud: " + str(accuracy_score(y_test, y_pred)))
print("Precision: " + str(precision_score(y_test, y_pred, average="micro")))
print("Recall: " + str(recall_score(y_test, y_pred, average='micro')))
print("Valor-F: " + str(f1_score(y_test, y_pred, average='micro')))

#¿cómo hacer diferentes evaluaciones con diferentes particiones del conjunto de entrada?
repeticiones = 3

for i in range (1, repeticiones+1):
    print("Iteracion: " + str(i))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
    y_pred = clf.predict(X_test)
    print("Exactitud: " + str(accuracy_score(y_test, y_pred)))
    print("Precision: " + str(precision_score(y_test, y_pred, average="micro")))
    print("Recall: " + str(recall_score(y_test, y_pred, average='micro')))
    print("Valor-F: " + str(f1_score(y_test, y_pred, average='micro')))
    print("----------------------------------------------------")

#Hacemos uso de otro clasificador con el que podamos comparar

clf= svm.SVC(kernel="linear", C=0.01).fit(X_train, y_train)
clf.fit(X_train,y_train.values.ravel())
scores=cross_val_score(clf, X, y.values.ravel(),cv=5)
print("Scores SVC: " + str(scores))

y_pred = clf.predict(X_test)

print("Cálculo de métricas para clasificador SVC")
print("Exactitud: " + str(accuracy_score(y_test, y_pred)))
print("Precision: " + str(precision_score(y_test, y_pred, average="micro")))
print("Recall: " + str(recall_score(y_test, y_pred, average='micro')))
print("Valor-F: " + str(f1_score(y_test, y_pred, average='micro')))

#Comparación gráfica y rápida entre varios clasificadores

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('SupportVectorClassifier', svm.SVC(kernel="linear", C=0.01)))
models.append(('RandomForestClassifier', RandomForestClassifier(n_estimators=10, random_state=42)))
models.append(('StochasticGradientDecentC', SGDClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'


for name, model in models:
    cv_results = cross_val_score(model, X, y.values.ravel(), cv=5)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Comparación de Algoritmos')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
