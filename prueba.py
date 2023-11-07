import pickle
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib


dir = 'C:/Angel/Programacion/project_Python/diabetes+130-us+hospitals+for+years+1999-2008/'
df = pd.read_csv(dir + 'diabetic_dataPP_PSimple.csv')

perceptron = Perceptron

X = df[['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
'time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient',
'number_emergency','number_inpatient','diag_1','diag_2','diag_3','number_diagnoses','max_glu_serum',
'A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone',
'tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin',
'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','change', 'diabetesMed']]
y = df['readmitted']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# cargar el modelo con joblib
model_pkl_file = "Perceptron_classifier_model.pkl" 

perceptron = joblib.load(model_pkl_file)

# evaluate model 
y_predict = perceptron.predict(X_test)

# check results
print(classification_report(y_test, y_predict)) 