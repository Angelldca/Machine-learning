import tkinter as tk
from tkinter import ttk
from const import colMin,colMax,loadTrain,dir,ppDecicionTree
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(dir + 'diabetic_dataPP_Categorias.csv')


ventana = tk.Tk()
ventana.title("Aprendizaje Automatico")


def obtener_algoritmo_seleccionado():
    print(algoritmo.get())




var1 = tk.BooleanVar()
var2 = tk.BooleanVar()
var3 = tk.BooleanVar()

var1.set(True)
var2.set(True)
var3.set(True)
algoritmo = tk.StringVar()
# Configurar los botones de opción
tk.Checkbutton(ventana, text="Knn", variable=var1).grid(row=0, column=0, sticky="w")
tk.Checkbutton(ventana, text="Árbol de decisión", variable=var2).grid(row=0, column=1, sticky="w")
tk.Checkbutton(ventana, text="Perceptrón simple", variable=var3).grid(row=0, column=2, sticky="w")

# Campos de entrada
entradas = []
i = 3
col = 0
for element in colMin:
    i += 1  
    values = [str(valor) for valor in df[element].unique()]
    ttk.Label(ventana, text=f'{element}:').grid(row=i, column=col, padx=10, pady=5, sticky='w')
    lista_desplegable = ttk.Combobox(ventana, values=values)
    lista_desplegable.grid(row=i, column=col+1, padx=10, pady=5, sticky='e')
    entradas.append(lista_desplegable)
    if i == 10:
        i=3
        col+=2

#resultado esperado
i = 12
etiqueta = ttk.Label(ventana, text=f'Resultado esperado')
etiqueta.grid(row=i, column=0, padx=10, pady=5, sticky='w')
entrada = ttk.Entry(ventana)
entrada.grid(row=i, column=1, padx=10, pady=5, sticky='e')
#entradas.append(entrada)


perceptron = joblib.load("Perceptron_classifier_model.pkl")
knn = joblib.load("Knn_classifier_model.pkl")
decicionTree = joblib.load("DecisionTree_classifier_model.pkl")

#Obtener elementos de las listas desplegables
seleccion_dict = {}
def obtener_seleccion():
    for element, lista_desplegable in zip(colMin, entradas):
        seleccion = lista_desplegable.get()
        seleccion_dict[element] = seleccion
    
    resultFinal()


y_pred = 0
diccionarioResult = {
    'Knn':0,
    'DecicionTree':0,
    'Perceptron':0
}


def knnResult():
    df = pd.DataFrame([seleccion_dict])
    ppDecicionTree(df,colMin)
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df.values)
    df = pd.DataFrame(X_normalized, columns=df.columns)
    print(knn.predict(df)[0])
    if knn.predict(df)[0] == 1:
        return 1
    else :
        return -1

def decisionTreeResult():
    df = pd.DataFrame([seleccion_dict])
    ppDecicionTree(df,colMin)
    print(decicionTree.predict(df)[0])
    if decicionTree.predict(df)[0] == 1:
        
        return 1
    else :
        return -1


def perceptronResult():
    df = pd.DataFrame([seleccion_dict])
    ppDecicionTree(df,colMin)
    print(perceptron.predict(df)[0])
    if perceptron.predict(df) == "Si":
       
        return 1
    else :
        return -1



def resultFinal():
    if var1.get():
       diccionarioResult['Knn'] = knnResult()
    if var2.get():
      diccionarioResult['DecicionTree']  =decisionTreeResult()
    if var3.get():
       diccionarioResult['Perceptron']= perceptronResult()
#y_pred = decicionTree.predict(X_test)


# Botón para ejecutar el algoritmo
i += 1

boton_ejecutar = ttk.Button(ventana, text="Ejecutar", command=obtener_seleccion)
boton_ejecutar.grid(row=i, column=0, columnspan=2, pady=10)

#Cargar entrenamiento


#loadTrain(knn,perceptron,decicionTree)

ventana.mainloop()