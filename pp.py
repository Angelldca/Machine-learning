import pandas as pd

#C:/Angel/Programacion/project_Python/diabetes+130-us+hospitals+for+years+1999-2008/
dir = 'C:/Angel/Programacion/project_Python/diabetes+130-us+hospitals+for+years+1999-2008/'
df = pd.read_csv(dir + 'diabetic_data.csv')
#datos.to_csv('C:/Angel/Programacion/project_Python/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_dataPP.csv')
##print(datos.head(3))

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
df['diag_1'] = df['diag_1'].replace('?', 0)
df['diag_2'] = df['diag_2'].replace('?', 0)
df['diag_3'] = df['diag_3'].replace('?', 0)

df['gender'] = df['gender'].replace('?', 'unknown')



#df['medical_specialty'] = df['medical_specialty'].replace('?','missing')


# Definir una función para convertir los rangos a valores numéricos(usando el valor medio)
def convertir_rango_a_valor(rango):
    inicio, fin = map(int, rango.strip('[]()').split('-'))
    if fin <= 30 :
      return "younger"
    elif inicio > 30 and fin <= 60:
        return "middle"
    else :
        return "older"
   

# Aplicar la función a la columna 'edad'
df['age'] = df['age'].apply(convertir_rango_a_valor)



# eliminar los pacientes en hospicio o muertos (11,13,14,19,20,21)
discharge_id = [11,13,14,19,20,21]
df = df[~df['discharge_disposition_id'].isin(discharge_id)]



# Eliminar elementos repetidos en la columna 'patient_nbr'
df = df.drop_duplicates(subset=['patient_nbr'])

#eliminar la columna weight, payer_code,medical_specialty ya que está (97%, 40%, 47%) incompleta
df = df.drop(['weight','payer_code','medical_specialty','encounter_id','patient_nbr'], axis=1)

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
print(porcentaje)  #69,984

df.to_csv(dir + 'diabetic_dataPP.csv',index=False)