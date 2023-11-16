from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (confusion_matrix,
precision_score,recall_score, f1_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN
import numpy as np
import pandas as pd
from math import sqrt
import joblib 
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from const import dir,colMax,colMin,ppDecicionTree,colAttr
df = pd.read_csv(dir + 'diabetic_dataPP.csv')


# Trabajar con los datos doumies
df = pd.get_dummies(df, columns=colAttr )

colNum = ['readmitted', 'Health_index', 'severity_disease']

ppDecicionTree(df,df.drop(colNum,axis=1))

df = df.sort_index(axis=1)

df.to_csv(dir +  'diabetic_dataPP_Perceptron.csv',index=False)

X = df.drop(columns=['readmitted'])
y = df['readmitted']

'''
smote = SMOTE(sampling_strategy='auto',random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

over_sampler = RandomOverSampler(sampling_strategy=0.5)
X_resampled, y_resampled = over_sampler.fit_resample(X, y)

adasyn = ADASYN(sampling_strategy=1)
X_resampled, y_resampled = adasyn.fit_resample(X, y)



tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X_train, y_train)


smote_enn = SMOTEENN(sampling_strategy=0.4)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)


smote = SMOTE(sampling_strategy=0.3)
X_resampled, y_resampled = smote.fit_resample(X, y)
'''


#Equilibrar datos SMOTE
adasyn = ADASYN(sampling_strategy=1)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,random_state=42)

#Crear un clasificador de perceptrón
perceptron = Perceptron(alpha=0.0001,max_iter=1000)

#Entrenar el perceptrón
perceptron.fit(X_train, y_train)

coeficientes = perceptron.coef_
sesgo = perceptron.intercept_

# Puedes ver los coeficientes asociados a cada característica

df_coeficientes = pd.DataFrame({'Caracteristica': X.columns, 'Coeficiente': coeficientes[0]})
df_coeficientes.to_csv(dir +  'caracteristicas.csv',index=False)  


# Predecir etiquetas para el conjunto de prueba
y_pred = perceptron.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)



print(f'Precisión del modelo Perceptron: {accuracy}')

matriz_confusion = confusion_matrix(y_test, y_pred)

print("Matriz de Confusión:")
print(matriz_confusion)


#y = df.iloc[:, -1]

print(f'Precision Score: {precision_score(y_test, y_pred)}')
print(f'Recall Score: {recall_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred)}')


from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt 

decision_function_scores = perceptron.decision_function(X_test)
# Supongamos que 'y_test' son las etiquetas reales y 'probabilidades_positivas' son las probabilidades predichas
fpr, tpr, umbrales = roc_curve(y_test, decision_function_scores)
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






model_pkl_file = "Perceptron_classifier_model.pkl"  




joblib.dump(perceptron, model_pkl_file)