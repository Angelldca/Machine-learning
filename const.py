import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler




dir = 'C:/Angel/Programacion/project_Python/AA_Python/diabetes+130-us+hospitals+for+years+1999-2008/'

colMin=['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
'time_in_hospital','num_lab_procedures','num_procedures','num_medications','service_utilization',
'diag_1','diag_2','diag_3','number_diagnoses','max_glu_serum','A1Cresult','metformin','glipizide','glyburide',
'pioglitazone','rosiglitazone','insulin','change','diabetesMed']

miniMaxCol = ['service_utilization','time_in_hospital','num_lab_procedures',
'num_procedures','num_medications','number_diagnoses']


colMinResult = ['age','discharge_disposition_id','admission_source_id',
'time_in_hospital','diag_1','max_glu_serum','A1Cresult','diabetesMed','gender__Female',
'gender__Male','gender__Unknown/Invalid','race__AfricanAmerican','race__Asian',
'race__Caucasian','race__Hispanic','race__Other']

colMax = ['age','admission_type_id','discharge_disposition_id','admission_source_id',
'time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient',
'number_emergency','number_inpatient','diag_1','diag_2','diag_3','number_diagnoses','max_glu_serum',
'A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone',
'tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin',
'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','change', 'diabetesMed']



def loadTrain(knn,perceptron,decisionTree):
    knn = joblib.load('Knn_classifier_model.pkl')
    perceptron = joblib.load('Perceptron_classifier_model.pkl')
    decisionTree = joblib.load('DecisionTree_classifier_model.pkl')
    print("Entrenamiento cargado")

def ppDecicionTree(df,columnas):
    encoder = LabelEncoder()
    for element in columnas:
        df[element] = df[element].astype(int)

mapeo = joblib.load("map.pkl")

def ppDataSet(df):
    
    mapeo_personalizado = {'No': 0, 'Steady': 1,'Down':1, 'Up': 1}
    # Aplicar el mapeo personalizado a las columnas de los medicamentos
    df['metformin'] = df['metformin'].map(mapeo_personalizado)
    df['glipizide'] = df['glipizide'].map(mapeo_personalizado)
    df['glyburide'] = df['glyburide'].map(mapeo_personalizado)
    df['pioglitazone'] = df['pioglitazone'].map(mapeo_personalizado)
    df['rosiglitazone'] = df['rosiglitazone'].map(mapeo_personalizado)
    df['insulin'] = df['insulin'].map(mapeo_personalizado)
    
    
    # Aplicar la transformación a la columnas 20-1 readmitted
    diag = {v: k for k, v in mapeo['diag'].items()}     
    df['diag_1'] = df['diag_1'].replace(diag)
    df['diag_2'] = df['diag_2'].replace(diag)
    df['diag_3'] = df['diag_3'].replace(diag)

    df['gender'] = df['gender'].replace({v: k for k, v in mapeo['gender'].items()})
    #df['age'] = df['age'].replace(mapeo['age'])
    df['discharge_disposition_id'] = df['discharge_disposition_id'].replace({v: k for k, v in mapeo['discharge_disposition_id'].items()})
    df['admission_source_id'] = df['admission_source_id'].replace({v: k for k, v in mapeo['admission_source_id'].items()})
    df['admission_type_id'] = df['admission_type_id'].replace({v: k for k, v in mapeo['admission_type_id'].items()})
    df['race'] = df['race'].replace({v: k for k, v in mapeo['race'].items()})
    
    mapeo_personalizado={'younger': 1, 'middle': 1.5,'older':2}
    df['age'] = df['age'].map(mapeo_personalizado)
    mapeo_personalizado={'No': 0, 'Yes': 1,'Ch':1}

    df['change'] = df['change'].map(mapeo_personalizado)

    df['diabetesMed'] = df['diabetesMed'].map(mapeo_personalizado)
    mapeo_personalizado={'none': 0, 'Norm': 1,'>200':3, '>300':3}
    df['max_glu_serum'] = df['max_glu_serum'].map(mapeo_personalizado)
    mapeo_personalizado={'none': 0, 'Norm': 1,'>7':3, '>8':3}
    df['A1Cresult'] = df['A1Cresult'].map(mapeo_personalizado)