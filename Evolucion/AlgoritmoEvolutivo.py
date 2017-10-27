'''
Created on 25/10/2017

@author: hector
'''
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
    

    def __init__(self, tamanoPoblacion, numeroNeuronas,numeroEntradas):
        self.tamanoPoblacion = tamanoPoblacion
        self.numeroNeuronas = numeroNeuronas
        self.numeroEntradas = numeroEntradas
        
    def evolucion(self):
        #generar la poblacion inicial random
        fMin = 0.006
        for x in range(self.tamanoPoblacion):
            self.poblacion.append(Individuo(np.random.rand(self.numeroNeuronas*(self.numeroEntradas+1)+2)))
        #evaluar la poblacionInicial
        sumatoriaAptitud = 0
        for ind in self.poblacion:
            pesosCapaOculta = np.array([ind.cromosoma[(x*self.numeroEntradas):((x*self.numeroEntradas)+self.numeroEntradas)] for x in range(self.numeroNeuronas)])
            pesosCapaSalida = np.array(ind.cromosoma[(55*self.numeroNeuronas):((self.numeroEntradas*self.numeroNeuronas)+self.numeroNeuronas)])
            biasCapaOculta = ind.cromosoma[-2]
            biasCapaSalida = ind.cromosoma[-1]
            ind.evaluacion = self.redNeuronal.ProbarModelo(pesosCapaOculta, pesosCapaSalida, biasCapaOculta, biasCapaSalida)
            ind.aptitud = (1/(ind.evaluacion))*100
            sumatoriaAptitud += ind.aptitud
        #fin evaluacion
        seleccionados = []# arreglo con los individuos que seran seleccionados
        #for para seleccionar los individuos
        for x in range(int(self.tamanoPoblacion/2)):
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
                print(np.random.standard_normal())
                if random.random()<0.5:
                    hijo1.append(padre1.cromosoma[x])
                    hijo2.append(padre2.cromosoma[x])
                else:
                    hijo2.append(padre1.cromosoma[x])
                    hijo1.append(padre2.cromosoma[x])
            hijos.append(Individuo(hijo1))
            hijos.append(Individuo(hijo2))
        # fin del cruce y creacion de los hijos
        self.poblacion.sort(key=lambda ind: ind.aptitud, reverse=True)# ordenar la poblacion para poder hacer el reemplazo
        # reemplazo
        self.poblacion =  self.poblacion[:int(self.tamanoPoblacion/2)]+hijos# se elimina una mitad y se le agrega el arreglo de hijos

        

        

            
  
e = Algoritmoevolutivo(24,3,55)
e.evolucion()