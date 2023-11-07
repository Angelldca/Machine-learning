import tkinter as tk
from tkinter import ttk
from const import colMin


def ejecutar_algoritmo():
    algoritmo_seleccionado = seleccion.get()
    print(f'Algoritmo seleccionado: {algoritmo_seleccionado}')
    datos = [entry.get() for entry in entradas]
    sexo = lista_desplegable.get()
    print(f'Datos ingresados: {datos}')
    print(f'Sexo seleccionado: {sexo}')

ventana = tk.Tk()
ventana.title("Interfaz de Selección")

# Elementos seleccionables
seleccion = ttk.Combobox(ventana, values=["Knn", "Árbol de decisión", "Perceptrón simple"])
seleccion.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

# Campos de entrada
entradas = []
i = 0
for element in colMin:  
    i += 1
    etiqueta = ttk.Label(ventana, text=f'{element}:')
    etiqueta.grid(row=i, column=0, padx=10, pady=5, sticky='w')
    entrada = ttk.Entry(ventana)
    entrada.grid(row=i, column=1, padx=10, pady=5, sticky='e')
    entradas.append(entrada)

# Lista desplegable para el sexo
i += 1
ttk.Label(ventana, text="Sexo:").grid(row=i, column=0, padx=10, pady=5, sticky='w')
lista_desplegable = ttk.Combobox(ventana, values=["Masculino", "Femenino"])
lista_desplegable.grid(row=i, column=1, padx=10, pady=5, sticky='e')

#resultado esperado
i += 1
etiqueta = ttk.Label(ventana, text=f'Resultado esperado')
etiqueta.grid(row=i, column=0, padx=10, pady=5, sticky='w')
entrada = ttk.Entry(ventana)
entrada.grid(row=i, column=1, padx=10, pady=5, sticky='e')
entradas.append(entrada)

# Botón para ejecutar el algoritmo
i += 1
boton_ejecutar = ttk.Button(ventana, text="Ejecutar", command=ejecutar_algoritmo)
boton_ejecutar.grid(row=i, column=0, columnspan=2, pady=10)



print(colMin)
ventana.mainloop()