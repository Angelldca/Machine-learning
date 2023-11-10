from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from const import dir,colMax,colMin,ppDecicionTree


#dir = 'C:/Angel/Programacion/project_Python/AA_Python/diabetes+130-us+hospitals+for+years+1999-2008/'
df = pd.read_csv(dir + 'diabetic_dataPP.csv')


x = ['service_utilization','time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_diagnoses']

scaler = MinMaxScaler()

# Aplicar la transformación de normalización a los datos
df[colMin] = scaler.fit_transform(df[colMin])


X = df[colMin]
y = df['readmitted']

#Equilibrar datos SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)

# Crear un clasificador k-NN con k= raiz cuadrada de cant de tuplas
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el clasificador
knn.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión Knn: {accuracy}')

# guardar knn

model_pkl_file = "Knn_classifier_model.pkl"  

joblib.dump(knn, model_pkl_file)

