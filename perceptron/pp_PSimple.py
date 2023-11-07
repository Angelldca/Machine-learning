
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from const import dir,colMax,colMin
df = pd.read_csv(dir + 'diabetic_dataPP.csv')
   

   ## preprocesar los datos categoricos a numericos

## convertir el atributo race a numerico
def preprocesarRace(elemento):
    if elemento == "Caucasian":
        return 1
    elif elemento == "AfricanAmerican":
        return -1
    else :
        return 0

## preprocesar el atributo gender a numerico
def preprocesarGender(elemento):
    if elemento == "Male":
        return 1
    elif elemento == "Female":
        return -1
    else :
        return 0
## preprocesar el atributo age a numerico
def preprocesarAge(elemento):
    if elemento == "younger":
        return 1
    elif elemento == "older":
        return 3
    else :
        return 2

def ppMax_glu_serum(element):
    if element == 'none':
        return 0
    elif element == '>200':
        return 2
    elif element == '>300':
        return 3
    else :
        return 1


def ppA1Cresult(element):
    if element == 'none':
        return 0
    elif element == '>7':
        return 2
    elif element == '>8':
        return 3
    else: return 1




# Definir un mapeo personalizado entre categorías de los medicamentos y números
mapeo_personalizado = {'No': 0, 'Steady': 1,'Down':2, 'Up': 3}

# Aplicar el mapeo personalizado a las columnas de los medicamentos
df['metformin'] = df['metformin'].map(mapeo_personalizado)
df['repaglinide'] = df['repaglinide'].map(mapeo_personalizado)
df['nateglinide'] = df['nateglinide'].map(mapeo_personalizado)
df['chlorpropamide'] = df['chlorpropamide'].map(mapeo_personalizado)
df['glimepiride'] = df['glimepiride'].map(mapeo_personalizado)
df['acetohexamide'] = df['acetohexamide'].map(mapeo_personalizado)
df['glipizide'] = df['glipizide'].map(mapeo_personalizado)
df['glyburide'] = df['glyburide'].map(mapeo_personalizado)
df['tolbutamide'] = df['tolbutamide'].map(mapeo_personalizado)
df['pioglitazone'] = df['pioglitazone'].map(mapeo_personalizado)
df['rosiglitazone'] = df['rosiglitazone'].map(mapeo_personalizado)
df['acarbose'] = df['acarbose'].map(mapeo_personalizado)
df['miglitol'] = df['miglitol'].map(mapeo_personalizado)
df['troglitazone'] = df['troglitazone'].map(mapeo_personalizado)
df['tolazamide'] = df['tolazamide'].map(mapeo_personalizado)
df['examide'] = df['examide'].map(mapeo_personalizado)
df['citoglipton'] = df['citoglipton'].map(mapeo_personalizado)
df['insulin'] = df['insulin'].map(mapeo_personalizado)
df['glyburide-metformin'] = df['glyburide-metformin'].map(mapeo_personalizado)
df['glipizide-metformin'] = df['glipizide-metformin'].map(mapeo_personalizado)
df['glimepiride-pioglitazone'] = df['glimepiride-pioglitazone'].map(mapeo_personalizado)
df['metformin-rosiglitazone'] = df['metformin-rosiglitazone'].map(mapeo_personalizado)
df['metformin-pioglitazone'] = df['metformin-pioglitazone'].map(mapeo_personalizado)

encoder = LabelEncoder()

# Aplicar la transformación a la columna correspondoente de categoria a numero
df['race'] = encoder.fit_transform(df['race'])
df['change'] = encoder.fit_transform(df['change'])
df['diabetesMed'] = encoder.fit_transform(df['diabetesMed'])

df['gender'] = encoder.fit_transform(df['gender'])
df['age'] = encoder.fit_transform(df['age'])
df['max_glu_serum'] = encoder.fit_transform(df['max_glu_serum'])
df['A1Cresult'] = encoder.fit_transform(df['A1Cresult'])


# Aplicar la transformación a la columnas 'diag'
df['diag_1'] = encoder.fit_transform(df['diag_1'])
df['diag_2'] = encoder.fit_transform(df['diag_2'])
df['diag_3'] = encoder.fit_transform(df['diag_3'])

#diag = df['diag_1'] == 1
#filasCondicion = df[diag]

#print(filasCondicion)
'''
# Aplicar la función a la columna 'race'
df['race'] = df['race'].apply(preprocesarRace)

'''





df.to_csv(dir + 'diabetic_dataPP_PSimple.csv',index=False)