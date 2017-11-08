'''
Created on 25/10/2017

@author: hector
'''
import sys
sys.path.append('../')
sys.path.append('/home/ec2-user/.local/lib/python3.6/site-packages')
import math
import numpy as np
import random
from Evolucion.individuo import Individuo
from backpropagation.Prueba import RedNeuronal

class Algoritmoevolutivo(object):
    '''
    classdocs
    '''
    poblacion = []
    tamanoPoblacion = 0
    numeroNeuronas = 0
    redNeuronal = RedNeuronal()
    numeroEntradas = 0
    

    def __init__(self, nombrePrueba ,tamanoPoblacion,mutacion,porcentajeReemplazo,escalado, numeroNeuronas,numeroEntradas):
        self.tamanoPoblacion = tamanoPoblacion
        self.mutacion= mutacion
        self.porcentajeReemplazo = porcentajeReemplazo
        self.escalado = escalado
        self.numeroNeuronas = numeroNeuronas
        self.numeroEntradas = numeroEntradas
        self.nombrePrueba = nombrePrueba
        self.evolucion()
        
    def evolucion(self):
        #generar la poblacion inicial random
        for x in range(self.tamanoPoblacion):
            #Generacion de indiviuos simple, todo de 0 a 1
            #self.poblacion.append(Individuo(list(np.random.rand(self.numeroNeuronas*(self.numeroEntradas+1)+2))))
            #Generacion de individuos: la primera capa de -1 a 1, la segunda capa de -10 a 10 y el bias de 0 a 1 
            capaOculta = [(random.random()-0.5)*2 for x in range(self.numeroNeuronas*self.numeroEntradas)]
            capaSalida = [(random.random()-0.5)*14 for x in range(self.numeroNeuronas)]
            bias = [random.random(),random.random()]
            cromosoma = capaOculta+capaSalida+bias
            self.poblacion.append(Individuo(cromosoma))
        generacion = 0
        mejorError = 9999999
        #evaluar la poblacionInicial
        while True:
            print("generacion: {}".format(generacion))
            sumatoriaAptitud = 0
            for ind in self.poblacion:
                pesosCapaOculta = np.array([ind.cromosoma[(x*self.numeroEntradas):((x*self.numeroEntradas)+self.numeroEntradas)] for x in range(self.numeroNeuronas)])
                pesosCapaSalida = np.array(ind.cromosoma[(55*self.numeroNeuronas):((self.numeroEntradas*self.numeroNeuronas)+self.numeroNeuronas)])
                biasCapaOculta = ind.cromosoma[-2]
                biasCapaSalida = ind.cromosoma[-1]
                ind.evaluacion = self.redNeuronal.ProbarModelo(pesosCapaOculta, pesosCapaSalida, biasCapaOculta, biasCapaSalida)
                ind.aptitud = (1/(ind.evaluacion))*100
                ind.precision = self.redNeuronal.precision
                # donde se guarda aparte cada modelo que va siendo mejor que el anterior
                if(ind.evaluacion<mejorError):
                    mejorError = ind.evaluacion
                    F = open("../resultadosEvolutivos/mejorResultado"+self.nombrePrueba+".txt","w")
                    F.write("generacion: {}".format(generacion))
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
            ''''F = open("../generaciones/generacion"+str(generacion)+".txt","w")
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
            # reemplazo
            self.poblacion =  self.poblacion[:(math.floor(self.tamanoPoblacion*(1-self.porcentajeReemplazo)))]+hijos# se elimina una mitad y se le agrega el arreglo de hijos
            #escalado no entra si el escalado es 1 ya que sera lo mismo
            if self.escalado != 1 :
                for ind in self.poblacion[:(math.floor(self.tamanoPoblacion*(1-self.porcentajeReemplazo)))]:
                    ind.cromosoma = [x*self.escalado for x in ind.cromosoma] # realizar el escalado 
            #fin escalado
            generacion += 1
        #fin while
                  
  
e = Algoritmoevolutivo("PruebaServer",100,0.08,0.94,1,19,55)