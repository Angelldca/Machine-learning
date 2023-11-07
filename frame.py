import tkinter as tk
from tkinter import ttk
from const import colMin,loadTrain,dir
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd




df = pd.read_csv(dir + 'diabetic_dataPP_Categorias.csv')
print(df['age'].unique()[0])

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
for element in colMin:
    i += 1  
    values = [str(valor) for valor in df[element].unique()]
    ttk.Label(ventana, text=f'{element}:').grid(row=i, column=0, padx=10, pady=5, sticky='w')
    lista_desplegable = ttk.Combobox(ventana, values=values)
    lista_desplegable.grid(row=i, column=1, padx=10, pady=5, sticky='e')


#resultado esperado
i += 1
etiqueta = ttk.Label(ventana, text=f'Resultado esperado')
etiqueta.grid(row=i, column=0, padx=10, pady=5, sticky='w')
entrada = ttk.Entry(ventana)
entrada.grid(row=i, column=1, padx=10, pady=5, sticky='e')
entradas.append(entrada)


perceptron = Perceptron
knn = KNeighborsClassifier
decicionTree = DecisionTreeClassifier
# Botón para ejecutar el algoritmo
i += 1
boton_ejecutar = ttk.Button(ventana, text="Ejecutar", command=loadTrain(knn,perceptron,decicionTree))
boton_ejecutar.grid(row=i, column=0, columnspan=2, pady=10)

#Cargar entrenamiento


#loadTrain(knn,perceptron,decicionTree)

ventana.mainloop()