import pandas as pd
import numpy as np
import joblib 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
dir = 'C:/Angel/Programacion/project_Python/AA_Python/diabetes+130-us+hospitals+for+years+1999-2008/'
df = pd.read_csv(dir + 'diabetic_data.csv')

encoder= LabelEncoder()
filas = df.shape[0]+1
columnas = df.shape[1]

nbr = df["encounter_id"] == 149190

filas_cumplen_condicion = df[nbr]
#print(filas_cumplen_condicion)


# Etiquetar los valores faltantes como 'Other'
df['race'].fillna('Other', inplace=True)

#rellenar campos vacios con none
df['max_glu_serum'].fillna('none', inplace=True)
df['A1Cresult'].fillna('none', inplace=True)

# remplazar los valores ? 
df['race'] = df['race'].replace('?', 'Other')
df['medical_specialty'] = df['medical_specialty'].replace('?', 'Other')
df['diag_1'] = df['diag_1'].replace('?', 0)
df['diag_2'] = df['diag_2'].replace('?', 0)
df['diag_3'] = df['diag_3'].replace('?', 0)





#df['medical_specialty'] = df['medical_specialty'].replace('?','missing')


# Definir una función para convertir los rangos a valores numéricos(usando el valor medio)
def convertir_rango_a_valor(rango):
    inicio, fin = map(int, rango.strip('[]()').split('-'))
# return (fin + inicio) /2
    if fin <= 30 :
        return "younger"
    elif inicio > 30 and fin <= 60:
        return "middle"
    else :
        return "older"
def convertir_rango_a_media(rango):
    inicio, fin = map(int, rango.strip('[]()').split('-'))
    return (fin + inicio) /2
    
# Aplicar la función a la columna 'edad'
df['age'] = df['age'].apply(convertir_rango_a_valor)
mapeo_personalizado={'younger': 1, 'middle': 5,'older':10}
df['age'] = df['age'].map(mapeo_personalizado)



# eliminar los pacientes en hospicio o muertos (11,13,14,19,20,21)
discharge_id = [11,13,14,19,20,21]
df = df[~df['discharge_disposition_id'].isin(discharge_id)]

#Eliminar filas de gender
df = df[df['gender'] != 'Unknown/Invalid']

# Eliminar elementos repetidos en la columna 'patient_nbr'
df = df.drop_duplicates(subset=['patient_nbr'], keep = 'first')

#eliminar la columna weight, payer_code,medical_specialty ya que está (97%, 40%, 47%) incompleta
df = df.drop(['weight','payer_code','encounter_id','patient_nbr'], axis=1)

#df = df.loc[df['readmitted'] == "NO"]  #11315 52528 35503

#asignar clasificacion (si el pasiente reingresa dentro de 30 dias o no)
def ppReadmitted(element):
    if element == '<30':
        return 1
    else :
        return -1
# Aplicar la función a la columna 'readmitted'
df['readmitted'] = df['readmitted'].apply(ppReadmitted)
# Cuenta cuántas veces aparece el valor en la columna
conteo_valor = df['readmitted'].value_counts().get(-1, 0)

filas = df.shape[0]
# Calcula el porcentaje
total_filas = len(df)
porcentaje = (conteo_valor / total_filas) * 100
print(porcentaje)


# Definir un mapeo personalizado entre categorías de los medicamentos y números
mapeo_personalizado = {'No': 0, 'Steady': 1,'Down':-10, 'Up': 10}

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
#Eliminar columnas con varianza < 0.5


denominator = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

# Calcular el Health_index y redondear a 4 lugares decimales
df['Health_index'] = np.where(denominator != 0,np.round(1 / denominator, 4),0)

df['severity_disease'] = df['time_in_hospital']+df['num_procedures']+df['num_medications']+df['num_lab_procedures']+df['number_diagnoses']





df = df.drop(['number_outpatient','number_emergency','number_inpatient',
'time_in_hospital','num_procedures','num_medications','num_lab_procedures','number_diagnoses',
'repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
'tolbutamide','acarbose','miglitol','troglitazone',
'tolazamide','examide','citoglipton','glyburide-metformin','glipizide-metformin',
'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone'], axis=1)

#Normalizar los atributos enteros
##"time_in_hospital"   "num_lab_procedures" "num_procedures"    
## [4] "num_medications"    "number_outpatient"  "number_emergency"  
## [7] "number_inpatient"   "number_diagnoses"
#X = ['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient',
#'number_emergency','number_inpatient','number_diagnoses']

# Crear un objeto de escalador MinMax
scaler = MinMaxScaler()

# Aplicar la transformación de normalización a los datos
#df[X] = scaler.fit_transform(df[X])

## Diagnosticos a categorias
mapeo_icd9 = {
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
    for rango, categoria in mapeo_icd9.items():
        if 'V' in str(codigo) or 'E' in str(codigo) :
            return 'Born'
        else : 
            inicio, fin = rango.split('-')
            if float(inicio) <= float(codigo) <= float(fin):
                return categoria
    return 'Other'
df['diag_1'] = df['diag_1'].apply(mapear_icd9_a_categoria)
df['diag_2'] = df['diag_2'].apply(mapear_icd9_a_categoria)
df['diag_3'] = df['diag_3'].apply(mapear_icd9_a_categoria)



# Aplicar la transformación a la columnas 'diag'
df['diag_1'] = encoder.fit_transform(df['diag_1']) +1
df['diag_2'] = encoder.fit_transform(df['diag_2']) +1
df['diag_3'] = encoder.fit_transform(df['diag_3']) +1

#####
mapeo={}
diag = dict(zip(range(1, len(encoder.classes_)+1), encoder.classes_))
mapeo['diag'] = diag


df['medical_specialty'] = encoder.fit_transform(df['medical_specialty']) +1
med = dict(zip(range(1, len(encoder.classes_)+1), encoder.classes_))
mapeo['medical_specialty'] = med


df['gender'] = encoder.fit_transform(df['gender'])+1
gender  = dict(zip(range(1, len(encoder.classes_)+1), encoder.classes_))
mapeo['gender'] = gender
df['race'] = encoder.fit_transform(df['race'])+1
race = dict(zip(range(1, len(encoder.classes_)+1), encoder.classes_))
mapeo['race'] = race


####discharge_disposition_id
def ppDischarge_disposition_id(e):
    if e in [1,2,3,4,5,6,8,15,16,17,22,23,24,30,27,28,29]:
        return "Discharged"
    elif e == 26:
        return 'Unknown'
    else :
        return "Otherwise"
df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(ppDischarge_disposition_id)
df['discharge_disposition_id'] = encoder.fit_transform(df['discharge_disposition_id'])+1

discharge_disposition_id = dict(zip(range(1, len(encoder.classes_)+1), encoder.classes_))
mapeo['discharge_disposition_id'] = discharge_disposition_id
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
df['admission_source_id'] = encoder.fit_transform(df['admission_source_id'])+1

admission_source_id = dict(zip(range(1, len(encoder.classes_)+1), encoder.classes_))
mapeo['admission_source_id'] = admission_source_id
##admission_type_id
admission_type_id = {
    1: 'Emergency',
    2: 'Urgency',
    3: 'Elective',
    4: 'Newborn',
    5: 'Not Available',
    6: 'Not Available',
    7: 'Trauma',
    8: 'Not Available'
}
df['admission_type_id'] = df['admission_type_id'].map(admission_type_id)
df['admission_type_id'] = encoder.fit_transform(df['admission_type_id'])+1

admission_type_id = dict(zip(range(1, len(encoder.classes_)+1), encoder.classes_))
mapeo['admission_type_id'] = admission_type_id

mapeo_personalizado={'No': 0, 'Yes': 1,'Ch':1}

df['change'] = df['change'].map(mapeo_personalizado)

df['diabetesMed'] = df['diabetesMed'].map(mapeo_personalizado)

mapeo_personalizado={'none': 0, 'Norm': 1,'>200':2, '>300':3}
df['max_glu_serum'] = df['max_glu_serum'].map(mapeo_personalizado)
mapeo_personalizado={'none': 0, 'Norm': 1,'>7':2, '>8':3}
df['A1Cresult'] = df['A1Cresult'].map(mapeo_personalizado)


print(mapeo)
model_pkl_file = "map.pkl"  
joblib.dump(mapeo, model_pkl_file)


'''
def tHospital(value):
    if(value <= 4):
        return "Poco"
    else:
        return "Mucho"

df['time_in_hospital'] = df['time_in_hospital'].apply(tHospital)
df['time_in_hospital'] = encoder.fit_transform(df['time_in_hospital'])+1
'''
df.to_csv(dir + 'diabetic_dataPP.csv',index=False)



def ppCategorias(df):
        # Etiquetar los valores faltantes como 'Other'
        df['race'].fillna('Other', inplace=True)

        #rellenar campos vacios con none
        df['max_glu_serum'].fillna('none', inplace=True)
        df['A1Cresult'].fillna('none', inplace=True)

        # remplazar los valores ? 
        df['medical_specialty'] = df['medical_specialty'].replace('?', 'Other')
        df['race'] = df['race'].replace('?', 'Other')
        df['diag_1'] = df['diag_1'].replace('?', 0)
        df['diag_2'] = df['diag_2'].replace('?', 0)
        df['diag_3'] = df['diag_3'].replace('?', 0)
        def convertir_rango_a_valor(rango):
            inicio, fin = map(int, rango.strip('[]()').split('-'))
            if fin <= 30 :
                return "younger"
            elif inicio > 30 and fin <= 60:
                return "middle"
            else :
                return "older"
        #Eliminar filas de gender
        df = df[df['gender'] != 'Unknown/Invalid']

        # Aplicar la función a la columna 'edad'
        df['age'] = df['age'].apply(convertir_rango_a_valor)



        # eliminar los pacientes en hospicio o muertos (11,13,14,19,20,21)
        discharge_id = [11,13,14,19,20,21]
        df = df[~df['discharge_disposition_id'].isin(discharge_id)]



        # Eliminar elementos repetidos en la columna 'patient_nbr'
        df = df.drop_duplicates(subset=['patient_nbr'])

        #eliminar la columna weight, payer_code,medical_specialty ya que está (97%, 40%, 47%) incompleta
        df = df.drop(['weight','payer_code','encounter_id','patient_nbr'], axis=1)

        #df = df.loc[df['readmitted'] == "NO"]  #11315 52528 35503

        #asignar clasificacion (si el pasiente reingresa dentro de 30 dias o no)
        def ppReadmitted(element):
            if element == '<30':
                return 1
            else :
                return -1
        # Aplicar la función a la columna 'readmitted'
        df['readmitted'] = df['readmitted'].apply(ppReadmitted)
        #Eliminar columnas con varianza < 0.5

        denominator = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
        
        # Calcular el Health_index y redondear a 4 lugares decimales
        df['Health_index'] = np.where(denominator != 0,np.round(1 / denominator, 4),0)
        df['severity_disease'] = df['time_in_hospital']+df['num_procedures']+df['num_medications']+df['num_lab_procedures']+df['number_diagnoses']
        #Health_index = ( 1 / (number_emergency + number_inpatient + number_outpatient) )
        #severity_of_disease  = (time_in_hospital + num_procedures + num_medications + num_lab_procedures + number_of_diagnoses)
        df = df.drop(['number_outpatient','number_emergency','number_inpatient',
        'time_in_hospital','num_procedures','num_medications','num_lab_procedures','number_diagnoses',
        'repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
        'tolbutamide','acarbose','miglitol','troglitazone',
        'tolazamide','examide','citoglipton','glyburide-metformin','glipizide-metformin',
        'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone'], axis=1)

        ## Diagnosticos a categorias
        mapeo_icd9 = {
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
            for rango, categoria in mapeo_icd9.items():
                if 'V' in str(codigo) or 'E' in str(codigo) :
                    return 'Other'
                else : 
                    inicio, fin = rango.split('-')
                    if float(inicio) <= float(codigo) <= float(fin):
                        return categoria
            return 'Other'
        df['diag_1'] = df['diag_1'].apply(mapear_icd9_a_categoria)
        df['diag_2'] = df['diag_2'].apply(mapear_icd9_a_categoria)
        df['diag_3'] = df['diag_3'].apply(mapear_icd9_a_categoria)
        ####discharge_disposition_id
        def ppDischarge_disposition_id(e):
            if e in [1,2,3,4,5,6,8,15,16,17,22,23,24,30,27,28,29]:
                    return "Discharged"
            elif e == 26:
                    return 'Unknown'
            else :
                return "Otherwise"
        df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(ppDischarge_disposition_id)
        
        
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
        
        
        ##admission_type_id
        admission_type_id = {
                1: 'Emergency',
                2: 'Urgency',
                3: 'Elective',
                4: 'Newborn',
                5: 'Not Available',
                6: 'Not Available',
                7: 'Trauma',
                8: 'Not Available'
        }
        df['admission_type_id'] = df['admission_type_id'].map(admission_type_id)

        df.to_csv(dir +  'diabetic_dataPP_Categorias.csv',index=False)        


dfCategories = pd.read_csv(dir + 'diabetic_data.csv')

ppCategorias(dfCategories)