
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from const import dir,colMax,colMin

df = pd.read_csv(dir + 'diabetic_dataPP.csv')


def preprocesarReadmitte(e):
    if e == 1:
        return 'Si'
    else :
        return 'No'

df['readmitted'] = df['readmitted'].apply(preprocesarReadmitte)

#df = df.sort_values(by='number_diagnoses')
#conteo_valores = df['number_diagnoses'].value_counts() #number_inpatient ,number_emergency number_outpatient 
#num_medications num_procedures num_lab_procedures time_in_hospital, number_diagnoses
#print(conteo_valores)

##Discretizar
cuantiles = [-1, 1, 5, float('inf')]

# Asigna las etiquetas
etiquetas = ['ninguno', 'poco', 'mucho']

# Discretiza la columna 'number_inpatient'
df['number_inpatient'] = pd.cut(df['number_inpatient'], bins=cuantiles, labels=etiquetas,right=False)
# Discretiza la columna 'number_emergency'
cuantiles = [-1, 1, 3, float('inf')]
df['number_emergency'] = pd.cut(df['number_emergency'], bins=cuantiles, labels=etiquetas,right=False)

# Discretiza la columna 'number_diagnoses'
cuantiles = [-1, 1, 5, float('inf')]
df['number_diagnoses'] = pd.cut(df['number_diagnoses'], bins=cuantiles, labels=etiquetas,right=False)



# Discretiza la columna 'number_outpatient'
cuantiles = [-1, 1, 16, float('inf')]
df['number_outpatient'] = pd.cut(df['number_outpatient'], bins=cuantiles, labels=etiquetas,right=False)

# Discretiza la columna 'number_outpatient'
cuantiles = [-1, 1, 8, float('inf')]
df['num_medications'] = pd.cut(df['num_medications'], bins=cuantiles, labels=etiquetas,right=False)
# Discretiza la columna 'num_procedures'
cuantiles = [-1, 1, 3, float('inf')]
df['num_procedures'] = pd.cut(df['num_procedures'], bins=cuantiles, labels=etiquetas,right=False)

# Discretiza la columna 'num_lab_procedures'
cuantiles = [-1, 1, 46, float('inf')]
df['num_lab_procedures'] = pd.cut(df['num_lab_procedures'], bins=cuantiles, labels=etiquetas,right=False)

# Discretiza la columna 'time_in_hospital'
cuantiles = [-1,1, 4, float('inf')]
df['time_in_hospital'] = pd.cut(df['time_in_hospital'], bins=cuantiles, labels=etiquetas,right=False)


####admission_source_id
def ppAdmission_source_id(e):
    if e in [4,5,6,10, 18,22,25]:
        return "Transfer"
    elif e in [1,2,3]:
        return "Referral"
    elif e in [23,24]:
        return "Born"
    elif e == 21:
        return "Unknown"
    else :
        return "Otherwise"

df['admission_source_id'] = df['admission_source_id'].apply(ppAdmission_source_id)
####discharge_disposition_id
def ppDischarge_disposition_id(e):
    if e == 1:
        return "Discharged to home"
    else :
        return "Otherwise"
df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(ppDischarge_disposition_id)

##admission_type_id
admission_type_id = {
    1: 'Emergency',
    2: 'Urgent',
    3: 'Elective',
    4: 'Newborn',
    5: 'Not Available',
    6: 'NULL',
    7: 'Trauma Center',
    8: 'Not Mapped'
}
df['admission_type_id'] = df['admission_type_id'].map(admission_type_id)
## Diagnosticos
mapeo_icd9_a_categorias = {
    '390-459': 'Circulatory',
    '460-519': 'Respiratory',
    '520-579': 'Digestive',
    '250-250.99': 'Diabetes',
    '800-999': 'Injury',
    '710-739': 'Musculoskeletal',
    '580-629': 'Genitourinary',
    '140-239': 'Neoplasms'
}

def mapear_icd9_a_categoria(codigo):
    for rango, categoria in mapeo_icd9_a_categorias.items():
        inicio, fin = rango.split('-')
        if 'V' in codigo or 'E' in codigo :
            return 'Other' 
        elif float(inicio) <= float(codigo) <= float(fin):
            return categoria
    return 'Other'
df['diag_1'] = df['diag_1'].apply(mapear_icd9_a_categoria)
df['diag_2'] = df['diag_2'].apply(mapear_icd9_a_categoria)
df['diag_3'] = df['diag_3'].apply(mapear_icd9_a_categoria)



columnas  = ['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
'time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient',
'number_emergency','number_inpatient','diag_1','diag_2','diag_3','number_diagnoses','max_glu_serum',
'A1Cresult','metformin','glipizide','glyburide','pioglitazone','rosiglitazone','insulin','change', 'diabetesMed']

encoder = LabelEncoder()

df.to_csv(dir + 'diabetic_dataPP_Categorias.csv',index=False)
# Aplicar la transformación a la columna correspondoente de categoria a numero
df['readmitted'] = encoder.fit_transform(df['readmitted'])

for element in columnas:
    df[element] = encoder.fit_transform(df[element])

df.to_csv(dir + 'diabetic_dataPP_DecisionTree.csv',index=False)