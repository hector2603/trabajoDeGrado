'''
Created on 12/10/2017

@author: hector
'''
import math
import numpy as np

class RedNeuronal(object):
    '''
    classdocs
    '''


    def __init__(self):
        factorEntrenamiento = 0.1;
        errorPermitido = 0.1;
        #entradaDeseada = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]];
        #salidaDeseada = [0,1,1,0,1,0,0,1];
        entradaDeseada = [[-1,-1],[-1,1],[1,-1],[1,1]];
        salidaDeseada = [0,1,1,0];
        pesosCapaOculta,pesosCapaSalida = self.backpropagation(factorEntrenamiento, errorPermitido, entradaDeseada, salidaDeseada)
        print("pesos optimos son: pesos capa oculta: {}  \t pesos capa salida: {} ".format(pesosCapaOculta, pesosCapaSalida))
        while True:
            entrada1 = int(input("entrada1"))
            entrada2 = int(input("entrada2"))
            salida,a,b,c = self.RedNeuronal([entrada1,entrada2], pesosCapaOculta, pesosCapaSalida)
            print("salida: {}".format(salida))
    def sigmoide(self,x):
        return 1 / (1 + math.exp(-x))
    def derivadaSigmoide(self,x):
        return self.sigmoide(x)*(1-self.sigmoide(x))
    def funcionDeActivacion(self, x):
        return self.sigmoide(x)
    def derivada(self, x):
        return self.derivadaSigmoide(x)
    
    def RedNeuronal(self,entrada,pesosCapa2,PesosCapa3):
        #capa oculta
        tendenciaCapa1= 1
        entradaNetaCapaOculta = pesosCapa2.dot(entrada)
        entradaNetaCapaOculta = [x+tendenciaCapa1 for x in entradaNetaCapaOculta]
        salidaCapaOculta = [self.funcionDeActivacion(x) for x in entradaNetaCapaOculta]
        #capa de salida
        tendenciaCapa2 = 1
        entradaNetaCapaSalida = PesosCapa3.dot(salidaCapaOculta)
        entradaNetaCapaSalida = entradaNetaCapaSalida+tendenciaCapa2
        salidaTotal =   self.funcionDeActivacion(entradaNetaCapaSalida)
        #print(" entrada: {} \n entrada capa oculta: {} \n salida capa oculta: {} \n entrada neta capa salida: {} \n salida total: {}".format(entrada,entradaNetaCapaOculta,salidaCapaOculta,entradaNetaCapaSalida,salidaTotal))
        return salidaTotal,salidaCapaOculta,entradaNetaCapaOculta,entradaNetaCapaSalida
    
    def backpropagation(self,entrenamiento,errorPermitido,entradaDeseada,salidaDeseada):
        
        pesosCapaOculta = np.random.rand(5,2) #np.array([[0.3568718 ,  0.6821255],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969]])#pesos de la capa oculta, deben ser aleatorios
        pesosCapaOculta2 = np.random.rand(5,5) #pesos de la capa oculta 2
        pesosCapaSalida = np.random.rand(5)  #np.array([0.69900,0.50459,0.5859,0.4789])# pesos de la capa de salida
        
        elementos = np.random.permutation(math.ceil(len(entradaDeseada)*0.8)) # lista desordenada para el entrenamiento
        #elementos = [0,1,2,3] # orden temporal mientras las pruebas
        #print (elementos)
        
        it = 0
        seguir = True
        while seguir:
            for i in elementos:
                it +=1
                pesosCapaOcultaActual = pesosCapaOculta
                pesosCapaSalidaActual = pesosCapaSalida
                patronIn = entradaDeseada[i]
                patronOut = salidaDeseada[i]
                #print(patronIn,patronOut)
                salidaObtenida,salidaCapaOculta,entradaNetaCapaOculta,entradaNetaCapaSalida = self.RedNeuronal(patronIn, pesosCapaOcultaActual, pesosCapaSalidaActual)
                #salidaCapaOculta es la salida de la capa oculta
                #salidaObtenida es la salida de la cada de entrada
                
                #Miramos la neurona de la capa de salida. Como es una solo la miramos a ella
                errorCapaDeSalida = (patronOut - salidaObtenida)*self.derivada(entradaNetaCapaSalida)
                #Actualizamos los pesos de la capa de salida
                
                for j in range(len(pesosCapaSalida)): #actializacion de los pesos en la capa de salida 
                    pesosCapaSalida[j] += entrenamiento*errorCapaDeSalida*salidaCapaOculta[j]
                
                #calculo de y actualizacion de pesos de la capa oculta
                # en estos errores no hay que hacer una sumatoria? 
                for j in range(len(entradaNetaCapaOculta)):
                    errorNeuronacapaOculta = self.derivada(entradaNetaCapaOculta[j])*pesosCapaSalidaActual[j]*errorCapaDeSalida
                    for k in range(len(pesosCapaOculta[j])):
                        pesosCapaOculta[j][k] += entrenamiento*errorNeuronacapaOculta*patronIn[k]
                        pesosCapaOculta[j][k] += entrenamiento*errorNeuronacapaOculta*patronIn[k]
                
    
                #Calculamos el error
                error = 0
                for j in range(len(entradaDeseada)):
                    entrada = entradaDeseada[j]
                    salidaEsperada = salidaDeseada[j]
                    salida,a,b,c = self.RedNeuronal(entrada, pesosCapaOculta, pesosCapaSalida)
                    error += 0.5*(salida - salidaEsperada)**2
                    
                print("error global: {}".format(error))
                
                if(error <errorPermitido):
                    print("salio")
                    seguir = False
                    break
                    
                #print("error capa de salida: {}".format(errorCapaDeSalida))
                #print("pesos capa de salida actualizados: {}".format(pesosCapaSalida))
                #print("pesos capa de oculta actualizados: {}".format(pesosCapaOculta))

        return pesosCapaOculta,pesosCapaSalida
RedNeuronal()
