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

from const import dir,colMax,colMin,ppDecicionTree
df = pd.read_csv(dir + 'diabetic_dataPP.csv')



# Trabajar con los datos doumies
df = pd.get_dummies(df, columns=df.drop(columns=['readmitted']).columns )

ppDecicionTree(df,df.drop(columns=['readmitted']).columns)

df = df.sort_index(axis=1)

df.to_csv(dir +  'diabetic_dataPP_Perceptron.csv',index=False)

X = df.drop(columns=['readmitted'])
y = df['readmitted']

#Equilibrar datos SMOTE
smote = SMOTE(sampling_strategy='auto',random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#Crear un clasificador de perceptrón
perceptron = Perceptron(alpha=0.0001,max_iter=1000)

#Entrenar el perceptrón
perceptron.fit(X_train, y_train)

coeficientes = perceptron.coef_
sesgo = perceptron.intercept_

# Puedes ver los coeficientes asociados a cada característica
print("Coeficientes:", coeficientes)
print("Sesgo (Intercept):", sesgo)

df_coeficientes = pd.DataFrame({'Caracteristica': X.columns, 'Coeficiente': coeficientes[0]})
df_coeficientes.to_csv(dir +  'caracteristicas.csv',index=False)  
# Predecir etiquetas para el conjunto de prueba
y_pred = perceptron.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

print(f'Precisión del modelo Perceptron: {accuracy}')

#print(classification_report(y_test, y_pred))
# Guardar el modelo entrenado

model_pkl_file = "Perceptron_classifier_model.pkl"  




joblib.dump(perceptron, model_pkl_file)