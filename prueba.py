import pickle
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib
import seaborn as sns

dir = 'C:/Angel/Programacion/project_Python/AA_Python/diabetes+130-us+hospitals+for+years+1999-2008/'
df = pd.read_csv(dir + 'diabetic_dataPP.csv')

perceptron = Perceptron

'''
X = df['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
'time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient',
'number_emergency','number_inpatient','diag_1','diag_2','diag_3','number_diagnoses','max_glu_serum',
'A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone',
'tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin',
'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','change', 'diabetesMed']
y = df['readmitted']
'''

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df['diag_1'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Número de Casos')
plt.ylabel('Frecuencia')
plt.title('Histograma de Número de Casos por Diagnóstico')
plt.show()


varianzas =[]
columnasMed = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone',
'tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin',
'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

columnasValidas =['metformin','glipizide','glyburide','pioglitazone','rosiglitazone','insulin']


for e in columnasValidas:
    varianzas.append(df[e].var())


# Graficar las varianzas
plt.xticks(rotation=45)
plt.bar(columnasValidas, varianzas)
plt.xlabel('Columnas')
plt.ylabel('Varianza')
plt.title('Varianza de las Columnas')
plt.show()


#### relacion entre el tiempo en el hospital con otros atributos
x = ['service_utilization','time_in_hospital',
'num_lab_procedures','num_procedures','num_medications','number_diagnoses']

#plt.hist(df['time_in_hospital'], bins=10)
plt.plot(df['time_in_hospital'].value_counts().sort_index(), marker='o')
plt.xlabel('Tiempo en time_in_hospital')
plt.ylabel('Frecuencia')
plt.show()

sns.countplot(x='diag_1', hue='readmitted', data=df)
plt.show()
sns.countplot(x='diag_2', hue='readmitted', data=df)
plt.show()
sns.countplot(x='diag_3', hue='readmitted', data=df)
plt.show()
# import libraries needed
import IPython
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# load original data into dataframe and check shape
df_ori = pd.read_csv(dir + "diabetic_data.csv")
print(df_ori.shape)
# examine the data types and descriptive stats
print(df_ori.info())
print(df_ori.describe())
# make a copy of the dataframe for preprocessing
df = df_ori.copy(deep=True)


sns.countplot(x='discharge_disposition_id', hue='readmitted', data=df)
plt.show()


sns.countplot(x='admission_source_id', hue='readmitted', data=df)
plt.show()


sns.countplot(x='admission_type_id', hue='readmitted', data=df)
plt.show()

sns.countplot(x='age', hue='readmitted', data=df)
plt.show()

sns.countplot(x='medical_specialty', hue='readmitted', data=df)
plt.xticks(rotation=45, ha='right')
plt.show()

#'time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient',
#'number_emergency','number_inpatient' 'change', 'diabetesMed' 'insulin' medical_specialty



sns.scatterplot(x='time_in_hospital', y='number_emergency', hue='readmitted', data=df)
plt.xlabel('Columna 1')
plt.ylabel('Columna 2')
plt.title('Relación entre Columna 1 y Columna 2, coloreada por Clase')
plt.show()


