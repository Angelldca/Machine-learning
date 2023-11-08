from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import joblib 
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from const import dir,colMax,colMin
df = pd.read_csv(dir + 'diabetic_dataPP.csv')


X = df[colMin]
y = df['readmitted']

#Equilibrar datos SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=67)

#Crear un clasificador de perceptrón
perceptron = Perceptron(alpha=0.00001,max_iter=1000)

#Entrenar el perceptrón
perceptron.fit(X_train, y_train)

# Predecir etiquetas para el conjunto de prueba
y_pred = perceptron.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

print(f'Precisión del modelo: {accuracy}')

#print(classification_report(y_test, y_pred))
# Guardar el modelo entrenado

model_pkl_file = "Perceptron_classifier_model.pkl"  




joblib.dump(perceptron, model_pkl_file)