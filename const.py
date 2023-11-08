import joblib
from sklearn.preprocessing import LabelEncoder




dir = 'C:/Angel/Programacion/project_Python/AA_Python/diabetes+130-us+hospitals+for+years+1999-2008/'

colMin=['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
'time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient',
'diag_1','diag_2','diag_3','number_diagnoses','max_glu_serum','A1Cresult','metformin','glipizide','glyburide',
'pioglitazone','rosiglitazone','insulin','change','diabetesMed']

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
        df[element] = encoder.fit_transform(df[element])

