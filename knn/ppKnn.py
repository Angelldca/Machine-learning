import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from const import dir,colMax,colMin


df = pd.read_csv(dir + 'diabetic_dataPP_PSimple.csv')
   

# Extraer las características del DataFrame
X = df.values

# Crear un objeto de escalador MinMax
scaler = MinMaxScaler()

# Aplicar la transformación de normalización a los datos
X_normalized = scaler.fit_transform(X)


df = pd.DataFrame(X_normalized, columns=df.columns)


df.to_csv(dir + 'diabetic_dataPP_Knn.csv',index=False)