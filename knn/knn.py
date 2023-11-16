from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import (confusion_matrix,
precision_score,recall_score, f1_score)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from const import dir,colMax,colMin,ppDecicionTree


#dir = 'C:/Angel/Programacion/project_Python/AA_Python/diabetes+130-us+hospitals+for+years+1999-2008/'
df = pd.read_csv(dir + 'diabetic_dataPP.csv')


x = ['Health_index','severity_disease']

scaler = MinMaxScaler()  #MinMaxScaler  StandardScaler

df[colMin] = scaler.fit_transform(df[colMin])


X = df[colMin]
y = df['readmitted']

#Equilibrar datos SMOTE
'''
smote = SMOTE(sampling_strategy='auto',random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

over_sampler = RandomOverSampler(sampling_strategy=0.5)
X_resampled, y_resampled = over_sampler.fit_resample(X, y)

adasyn = ADASYN(sampling_strategy=0.5)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X_train, y_train)
'''
smote_enn = SMOTEENN(sampling_strategy=0.4)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)




# Crear un clasificador k-NN con k= raiz cuadrada de cant de tuplas
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el clasificador
knn.fit(X_resampled, y_resampled)

# Realizar predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)


print(f'Precisión Knn: {accuracy}')


matriz_confusion = confusion_matrix(y_test, y_pred)

print("Matriz de Confusión:")
print(matriz_confusion)


#y = df.iloc[:, -1]

print(f'Precision Score: {precision_score(y_test, y_pred)}')
print(f'Recall Score: {recall_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred)}')


from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt 

probabilidades_positivas = knn.predict_proba(X_test)[:, 1]
# Supongamos que 'y_test' son las etiquetas reales y 'probabilidades_positivas' son las probabilidades predichas
fpr, tpr, umbrales = roc_curve(y_test, probabilidades_positivas)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea Base')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR) o Sensibilidad')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

### Graficar Knn
#x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
#y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

# Graficar resultado X_resampled, y_resampled
X = X_resampled
y= y_resampled
dftest = X.iloc[5]
X = X.drop([5])
y = y.drop([5])
plt.figure(figsize=(8, 6))
##plt.scatter(X[:, 0], X[:, 1], 
plt.scatter(X['Health_index'].values,X['severity_disease'].values,
c=y,
edgecolors='k',
cmap=plt.cm.Paired)
plt.scatter(dftest['Health_index'], dftest['severity_disease'], label='Test Sample', c='k',
marker='D')
plt.title('KNN Classifier Decision')
plt.xlabel('Health_index')
plt.ylabel('severity_disease')
plt.show()




correlation_matrix = df.corr()
class_correlation = correlation_matrix['readmitted'].abs().sort_values(ascending=False)
print(class_correlation)

# guardar knn
model_pkl_file = "Knn_classifier_model.pkl"  

joblib.dump(knn, model_pkl_file)

