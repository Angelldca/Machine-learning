from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (confusion_matrix,
precision_score,recall_score, f1_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN
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






X = df[colMin]
y = df['readmitted']

'''
smote = SMOTE(sampling_strategy=0.5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

over_sampler = RandomOverSampler(sampling_strategy=0.5)
X_resampled, y_resampled = over_sampler.fit_resample(X_train, y_train)

under_sampler = RandomUnderSampler(sampling_strategy=0.5)
X_resampled, y_resampled = under_sampler.fit_resample(X, y)


smote = SMOTEENN(sampling_strategy=0.4)
X_resampled, y_resampled = smote.fit_resample(X, y)

tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X, y)

'''

#Equilibrar datos 
over_sampler = RandomOverSampler(sampling_strategy=1)
X_resampled, y_resampled = over_sampler.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)



clf = DecisionTreeClassifier()
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



matriz_confusion = confusion_matrix(y_test, y_pred)

print("Matriz de Confusión:")
print(matriz_confusion)


#y = df.iloc[:, -1]

print(f'Precision Score: {precision_score(y_test, y_pred)}')
print(f'Recall Score: {recall_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred)}')


from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt 

probabilidades_positivas = clf.predict_proba(X_test)[:, 1]
# Supongamos que 'y_test' son las etiquetas reales y 'probabilidades_positivas' son las probabilidades predichas
fpr, tpr, umbrales = roc_curve(y_test, probabilidades_positivas)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea Base')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR) o Sensibilidad')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()



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
