import joblib


dir = 'C:/Angel/Programacion/project_Python/AA_Python/diabetes+130-us+hospitals+for+years+1999-2008/'

colMin = ['race','gender','age','discharge_disposition_id','admission_source_id',
'time_in_hospital','diag_1','max_glu_serum','A1Cresult','diabetesMed']

colMax = ['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
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
