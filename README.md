# Trabajo de Grado.
Implementación de algoritmos de inteligencia artificial para La categorización de retinopatía diabética, en el trabajo de grado de ingeniería de sistemas.

## Pruebas. 
las Pruebas realizadas son los de algoritmos evolutivos, backpropagation y SVM,

### Pruebas de backpropagation.
Las Pruebas de backpropagation se pueden encontrar en la carpeta "Pruebas Backpropagation", se realizaron pruebas con 19 y 38 neuronas.

En cada archivo de texto, se puede encontrar todo el modelo, pesos de la capa oculta, pesos de la capa de salida, y la precisión del modelo con los datos de prueba.

### Pruebas de algoritmo genéticos.
Las Pruebas de algoritmos genéticos se encuentran en la carpeta "resultados evolutivos", como en los algoritmos genéticos se realizan varias pruebas, se divide en dos carpetas con los conjuntos de pruebas.

En el nombre de cada archivo de texto se encuentra el número que identifica las características de la prueba y después del guión "-" el número de la pruebas, donde 0 será la primer prueba y 1 la segunda prueba.

Las carácteristicas de cada prueba se puede encontrar en el documento pdf. donde la prueba 1 sería:

Pruebas 1, Población: 50, Mutación: 8%, Reemplazo: Inserción, Escalado: No.

En cada archivo de texto se podrá encontrar el modelo de los pesos. pesos de la capa oculta, pesos de la capa de salida, bias capa oculta, bias capa de salida, además de algunos datos importantes como fecha y hora del inicio de la ejecución, fecha y hora en que se encontró el modelo, generación en la que se encontró, el valor de la evaluación del cromosoma y por último el valor de la precisión,

## Ejecución.
Para la ejecución del los algoritmos se necesita :
 * Python 3.7 
 * Numpy.
 * PyQt5
 * Matplotlib
 * Seaborn
 * SKlearn
 * Pickle
 * Pandas

Los datos de entrenamiento ya están organizados en el proyecto, en el archivo datos_retinopatia.arff.

Para la ejecución del proyecto, se debe ejecutar el archivo Vistas/Main.py

