
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
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





encoder = LabelEncoder()

# Aplicar la transformación a la columna correspondoente de categoria a numero
df['race'] = encoder.fit_transform(df['race'])
mapeo_personalizado={'No': 0, 'Yes': 1,'Ch':1}
df['change'] = df['change'].map(mapeo_personalizado)
df['diabetesMed'] = df['diabetesMed'].map(mapeo_personalizado)

df['gender'] = encoder.fit_transform(df['gender'])
mapeo_personalizado={'younger': 1, 'middle': 1.5,'older':2}
df['age'] = df['age'].map(mapeo_personalizado)

mapeo_personalizado={'none': 0, 'Norm': 1,'>200':2, '>300':3}
df['max_glu_serum'] = df['max_glu_serum'].map(mapeo_personalizado)
mapeo_personalizado={'none': 0, 'Norm': 1,'>7':2, '>8':3}
df['A1Cresult'] = df['A1Cresult'].map(mapeo_personalizado)


# Aplicar la transformación a la columnas 'diag'
df['diag_1'] = encoder.fit_transform(df['diag_1'])
df['diag_2'] = encoder.fit_transform(df['diag_2'])
df['diag_3'] = encoder.fit_transform(df['diag_3'])

discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform',subsample=None)
df['diag_1'] = discretizer.fit_transform(df[['diag_1']])
df['diag_2'] = discretizer.fit_transform(df[['diag_2']])
df['diag_3'] = discretizer.fit_transform(df[['diag_3']])
df['time_in_hospital'] = discretizer.fit_transform(df[['time_in_hospital']])
df['discharge_disposition_id'] = discretizer.fit_transform(df[['discharge_disposition_id']])
df['admission_source_id'] = discretizer.fit_transform(df[['admission_source_id']])

#diag = df['diag_1'] == 1
#filasCondicion = df[diag]

#print(filasCondicion)
'''
# Aplicar la función a la columna 'race'
df['race'] = df['race'].apply(preprocesarRace)

'''





df.to_csv(dir + 'diabetic_dataPP_PSimple.csv',index=False)