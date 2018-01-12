'''
Created on 25/10/2017

@author: hector
'''
import sys
from time import gmtime
sys.path.append('../')
sys.path.append('/home/ec2-user/.local/lib/python3.6/site-packages')
from pylab import *
import math
import numpy as np
import random
import threading
import time
from Evolucion.individuo import Individuo
from backpropagation.Prueba import RedNeuronal

class Algoritmoevolutivo(threading.Thread):
    '''
    classdocs
    '''

    def __init__(self, nombrePrueba ,tamanoPoblacion,mutacion,porcentajeReemplazo,escalado, numeroNeuronas,numeroEntradas,segundos):
        self.poblacion = []
        self.redNeuronal = RedNeuronal()
        self.tamanoPoblacion = tamanoPoblacion
        self.mutacion= mutacion
        self.porcentajeReemplazo = porcentajeReemplazo
        self.escalado = escalado
        self.numeroNeuronas = numeroNeuronas
        self.numeroEntradas = numeroEntradas
        self.nombrePrueba = nombrePrueba
        self.segundos = segundos
        threading.Thread.__init__(self, name=nombrePrueba)
    
    def run(self):
        self.evolucion()
        
    def evolucion(self):
        #generar la poblacion inicial random
        for x in range(self.tamanoPoblacion):
            #Generacion de indiviuos simple, todo de 0 a 1
            #self.poblacion.append(Individuo(list(np.random.rand(self.numeroNeuronas*(self.numeroEntradas+1)+2))))
            #Generacion de individuos: la primera capa de -1 a 1, la segunda capa de -10 a 10 y el bias de 0 a 1 
            capaOculta = [(random.random()-0.5)*2 for x in range(self.numeroNeuronas*self.numeroEntradas)]
            capaSalida = [(random.random()-0.5)*14 for x in range(self.numeroNeuronas)]
            bias = [random.random(),random.random()]# para que el bias se volucione con el cromosoma 
            cromosoma = capaOculta+capaSalida+bias
            self.poblacion.append(Individuo(cromosoma))
        generacion = 0
        mejorError = 9999999
        inicial = time.time()
        actual = time.time()
        limite = inicial + self.segundos
        errorGrafica = [] # guarda el mejor error de cada generación
        mejorErrorGrafica = [] #guarda el mejor error obtenido
        #evaluar la poblacionInicial
        G = open("../Generaciones/generacion"+str(self.nombrePrueba)+".txt","w")
        G.close()
        while actual<=limite:
            G = open("../Generaciones/generacion"+str(self.nombrePrueba)+".txt","r+")
            G.write("Generacion numero"+str(generacion)+"\n")
            G.close()
            print("generacion: {}  de la prueba: {}".format(generacion,self.nombrePrueba))
            sumatoriaAptitud = 0
            for ind in self.poblacion:
                pesosCapaOculta = np.array([ind.cromosoma[(x*self.numeroEntradas):((x*self.numeroEntradas)+self.numeroEntradas)] for x in range(self.numeroNeuronas)])
                pesosCapaSalida = np.array(ind.cromosoma[(55*self.numeroNeuronas):((self.numeroEntradas*self.numeroNeuronas)+self.numeroNeuronas)])
                biasCapaOculta = 1 #ind.cromosoma[-2]
                biasCapaSalida = 1 #ind.cromosoma[-1]
                ind.evaluacion = self.redNeuronal.ProbarModelo(pesosCapaOculta, pesosCapaSalida, biasCapaOculta, biasCapaSalida)
                ind.aptitud = (1/(ind.evaluacion))*100
                ind.precision = self.redNeuronal.precision
                # donde se guarda aparte cada modelo que va siendo mejor que el anterior
                if(ind.evaluacion<mejorError):
                    mejorError = ind.evaluacion
                    mejorErrorGrafica.append(mejorError)
                    F = open("../resultadosEvolutivos/mejorResultado"+self.nombrePrueba+".txt","w")
                    F.write("\n fecha de inicio {}".format(time.strftime("%a, %d %b %Y %H:%M:%S ", gmtime(inicial))))
                    F.write("\n fecha de la generacion {}".format(time.strftime("%a, %d %b %Y %H:%M:%S ", gmtime(actual))))
                    F.write("\n generacion: {}".format(generacion))
                    F.write("\n pesos capa oculta 1\n")
                    for m in range(len(pesosCapaOculta)):
                            F.write(str(pesosCapaOculta[m]))
                    F.write("\n pesos capa salida\n")
                    F.write(str(pesosCapaSalida))
                    F.write("\n biasCapaoculta\n")
                    F.write(str(biasCapaOculta))
                    F.write("\n biasCapaoSalida\n")
                    F.write(str(biasCapaSalida))
                    F.write("\n mejor error\n")
                    F.write(str(mejorError))
                    F.write("\n Precision \n")
                    F.write(str(ind.precision))
                    F.close()
                    print(ind.precision)
                # fin if
                sumatoriaAptitud += ind.aptitud
            #fin evaluacion
            #for donde se guarda toda la generacion
            ''''F = open("../Generaciones/generacion"+str(generacion)+".txt","w")
            F.write("Generacion numero"+str(generacion)+"\n")
            for ind in self.poblacion:
                    F.write("Precision:  ")
                    F.write(str(ind.precision))
                    F.write("   Error: ")
                    F.write(str(ind.evaluacion))
                    F.write("   Aptitud: ")
                    F.write(str(ind.aptitud))
                    F.write("   Cromosoma:  ")
                    F.write(str(ind.cromosoma))
                    F.write("\n")
            F.close()'''
            #fin for 
            seleccionados = []# arreglo con los individuos que seran seleccionados
            #for para seleccionar los individuos
            for x in range(math.ceil(self.tamanoPoblacion*(self.porcentajeReemplazo))):
                ruleta = random.random()*sumatoriaAptitud 
                sumaBusqueda = 0
                for ind in self.poblacion:
                    sumaBusqueda += ind.aptitud
                    if ruleta < sumaBusqueda:
                        seleccionados.append(ind)
                        break
            #fin de seleccionar los individuos
            #cruce de individuos
            hijos = []
            for x in range(int(len(seleccionados)/2)):
                padre1 = seleccionados[(2*x)]
                padre2 = seleccionados[(2*x)+1]
                hijo1=[]
                hijo2=[]
                for x in range(len(padre1.cromosoma)):
                    if random.random()<0.5:# la probabilidad igual para hacer el cruce o dejar igual
                        hijo1.append(padre1.cromosoma[x])
                        hijo2.append(padre2.cromosoma[x])
                    else:
                        hijo2.append(padre1.cromosoma[x])
                        hijo1.append(padre2.cromosoma[x])
                        
                    if random.random()<self.mutacion:# probabilidad de mutacion para el hijo1 en el gen x
                        hijo1[x] += (np.random.standard_normal())
                        
                    if random.random()<self.mutacion:# probabilidad de mutacion para el hijo2 en el gen x
                        hijo2[x] += (np.random.standard_normal())
    
                hijos.append(Individuo(hijo1))
                hijos.append(Individuo(hijo2))
            # fin del cruce y creacion de los hijos
            self.poblacion.sort(key=lambda ind: ind.aptitud, reverse=True)# ordenar la poblacion para poder hacer el reemplazo
            errorGrafica.append(self.poblacion[0].evaluacion)
            # reemplazo
            self.poblacion =  self.poblacion[:(math.floor(self.tamanoPoblacion*(1-self.porcentajeReemplazo)))]+hijos# se elimina una mitad y se le agrega el arreglo de hijos
            #escalado no entra si el escalado es 1 ya que sera lo mismo
            if self.escalado != 1 :
                for ind in self.poblacion[:(math.floor(self.tamanoPoblacion*(1-self.porcentajeReemplazo)))]:
                    ind.cromosoma = [x*self.escalado for x in ind.cromosoma] # realizar el escalado 
            #fin escalado
            generacion += 1
            actual = time.time()
        print("fin")
        #grafica del mejor error de cada generacion
        plt.plot(errorGrafica)
        plt.title("Error de cada generación")
        plt.xlabel("Generación")
        plt.ylabel("ECM")
        plt.savefig("../imagenesPruebas/error"+self.nombrePrueba+".png")
        plt.cla()   # Borrar información de los ejes
        plt.clf()   # Borrar un gráfico completo
        
        #grafica del mejor error obtenido hasta el momento
        plt.plot(mejorErrorGrafica)
        plt.title("Mejor error obtenido ")
        plt.ylabel("ECM")
        plt.savefig("../imagenesPruebas/mejorError"+self.nombrePrueba+".png")
        plt.cla()   # Borrar información de los ejes
        plt.clf()   # Borrar un gráfico completo
         
        #fin while
                  

'''
for x in range(10):
    print(" iteracion numero: {}".format(x))
    listaObjetos = []
    listaObjetos.append(Algoritmoevolutivo('Prueba 1{}'.format(x),50,0.08,0.5,1.1 ,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 2{}'.format(x),50,0.08,0.5,1,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 3{}'.format(x),50,0.08,0.88,1.1 ,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 4{}'.format(x),50,0.08,0.88,1,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 5{}'.format(x),50,0.2,0.5,1.1 ,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 6{}'.format(x),50,0.2,0.5,1,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 7{}'.format(x),50,0.2,0.88,1.1 ,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 8{}'.format(x),50,0.2,0.88,1,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 9{}'.format(x),100,0.08,0.5,1.1 ,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 10{}'.format(x),100,0.08,0.5,1,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 11{}'.format(x),100,0.08,0.94,1.1 ,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 12{}'.format(x),100,0.08,0.94,1,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 13{}'.format(x),100,0.2,0.5,1.1 ,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 14{}'.format(x),100,0.2,0.5,1,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 15{}'.format(x),100,0.2,0.94,1.1 ,19,55,172800))
    listaObjetos.append(Algoritmoevolutivo('Prueba 16{}'.format(x),100,0.2,0.94,1,19,55,172800))
    for evolucion in listaObjetos:
        evolucion.start()
    time.sleep(172800)'''

if __name__ == '__main__':
    e = Algoritmoevolutivo('Prueba 1{}'.format(3),50,0.08,0.5,0.999999 ,19,55,21600)
    e.run()
    e = Algoritmoevolutivo('Prueba 2{}'.format(3),50,0.08,0.5,1,19,55,21600)
    e.run()
    e = Algoritmoevolutivo('Prueba 3{}'.format(3),50,0.08,0.88,0.999999 ,19,55,21600)
    e.run()
    e = Algoritmoevolutivo('Prueba 4{}'.format(3),50,0.08,0.88,1,19,55,21600)
    e.run()
    e = Algoritmoevolutivo('Prueba 5{}'.format(3),50,0.2,0.5,0.999999 ,19,55,21600)
    e.run()
    e = Algoritmoevolutivo('Prueba 6{}'.format(3),50,0.2,0.5,1,19,55,21600)
    e.run()
    e = Algoritmoevolutivo('Prueba 7{}'.format(3),50,0.2,0.88,0.999999 ,19,55,21600)
    e.run()
    e = Algoritmoevolutivo('Prueba 8{}'.format(3),50,0.2,0.88,1,19,55,21600)
    e.run()
    e = Algoritmoevolutivo('Prueba 9{}'.format(3),100,0.08,0.5,0.999999 ,19,55,21600)
    e.run()



