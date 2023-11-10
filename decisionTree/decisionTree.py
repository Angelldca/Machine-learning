from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import graphviz
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from const import dir,colMin, ppDecicionTree



# conjunto de datos 

df = pd.read_csv(dir + 'diabetic_dataPP.csv')

columnas  = colMin




X = df[columnas]
y = df['readmitted']
#Equilibrar datos SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

clf = DecisionTreeClassifier()


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)



clf = clf.fit(X_train, y_train)

feature_importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Ordenar las características por importancia descendente
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

feature_importance_df.to_csv(dir +  'import.csv',index=False)  

y_pred = clf.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy}')

######

class_column = 'readmitted'
# Obtener los nombres únicos de las clases
class_names = df[class_column].unique().tolist()

# Obtener los nombres de las características
feature_names = X.columns.tolist()
#feature_names.remove(class_column)
#print(feature_names)
## Guardar imajen jpg
def saveImage():
    plt.figure(figsize=(200, 100))
    tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=feature_names, rounded=True, fontsize=10)
    plt.savefig("arbol_decision.png")
    #tree.plot_tree(clf,max_depth=2)
#saveImage()
#Crear el arbol en pdf
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render(dir + "arbol")



# guardar dt

model_pkl_file = "DecisionTree_classifier_model.pkl"  

joblib.dump(clf, model_pkl_file)
