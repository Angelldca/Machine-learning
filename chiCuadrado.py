import pandas as pd
from scipy.stats import chi2_contingency,spearmanr
from const import colMin,colMax,loadTrain,dir,ppDecicionTree,ppDataSet
dir = 'C:/Angel/Programacion/project_Python/AA_Python/diabetes+130-us+hospitals+for+years+1999-2008/'
df = pd.read_csv(dir + 'diabetic_dataPP_Categorias.csv')

categorical_features = ['gender','age', 'admission_type_id', 'discharge_disposition_id', 
'admission_source_id', 'medical_specialty', 'diag_1', 'diag_2','diag_3', 'readmitted']
rejected_features = []

for col in df.columns : 
    data_crosstab = pd.crosstab(df['readmitted'],  
                                df[col], 
                                margins = False) 

    stat, p, dof, expected = chi2_contingency(data_crosstab)
    if p < 0.4 :
	    print(p, col, 'is significant')
    else:
        print(p, col, 'is not significant')
        rejected_features.append(col)

print("_____________________________________________")
for col in colMin :
    rho , pval = spearmanr(df['readmitted'], df[col])
    if pval < 0.4 : 
        print(col, 'is significant1')
    else : 
        print(col, 'is not significant1')
        rejected_features.append(col)