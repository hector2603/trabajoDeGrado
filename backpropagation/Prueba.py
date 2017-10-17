'''
Created on 12/10/2017

@author: hector
'''
import math
import numpy as np
from Datos.Datos import Datos

class RedNeuronal(object):
    '''
    classdocs
    '''


    def __init__(self):
        factorEntrenamiento = 0.1;
        errorPermitido = 0.1;
        #entradaDeseada = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]];
        #salidaDeseada = [0,1,1,0,1,0,0,1];
        #entradaDeseada = [[-1,-1],[-1,1],[1,-1],[1,1]];
        #salidaDeseada = [0,1,1,0];
        datos = Datos()
        entradaDeseada = datos.Datos
        salidaDeseada = datos.Resultado
        #pesosCapaOculta,pesosCapaSalida = self.backpropagation(factorEntrenamiento, errorPermitido, entradaDeseada, salidaDeseada,60,60)
        pesosCapaOculta,pesosCapaSalida = self.backpropagation1CapaOculta(factorEntrenamiento, errorPermitido, entradaDeseada, salidaDeseada,19)

    def sigmoide(self,x):
        return 1 / (1 + math.exp(-x))
    def derivadaSigmoide(self,x):
        return self.sigmoide(x)*(1-self.sigmoide(x))
    def funcionDeActivacion(self, x):
        return self.sigmoide(x)
    def derivada(self, x):
        return self.derivadaSigmoide(x)
    
    def RedNeuronal(self,entrada,pesosCapaOculta1,pesosCapaOculta2,pesosCapaSalida):
        #capa oculta 1
        tendenciaCapa1= 1
        entradaNetaCapaOculta1 = pesosCapaOculta1.dot(entrada)
        entradaNetaCapaOculta1 = [x+tendenciaCapa1 for x in entradaNetaCapaOculta1]
        salidaCapaOculta1 = [self.funcionDeActivacion(x) for x in entradaNetaCapaOculta1]
        
        #capa oculta 2
        tendenciaCapa2 = 1
        entradaNetaCapaOculta2 = pesosCapaOculta2.dot(salidaCapaOculta1)
        entradaNetaCapaOculta2 = [x+tendenciaCapa2 for x in entradaNetaCapaOculta2]
        salidaCapaOculta2 = [self.funcionDeActivacion(x) for x in entradaNetaCapaOculta2]
        
        #capa de salida
        tendenciaCapaSalida = 1
        entradaNetaCapaSalida = pesosCapaSalida.dot(salidaCapaOculta2)
        entradaNetaCapaSalida = entradaNetaCapaSalida+tendenciaCapaSalida
        salidaTotal =   self.funcionDeActivacion(entradaNetaCapaSalida)
        #print(" entrada: {} \n entrada capa oculta: {} \n salida capa oculta: {} \n entrada neta capa salida: {} \n salida total: {}".format(entrada,entradaNetaCapaOculta1,salidaCapaOculta1,entradaNetaCapaSalida,salidaTotal))
        return salidaTotal,salidaCapaOculta1,salidaCapaOculta2,entradaNetaCapaSalida,entradaNetaCapaOculta1,entradaNetaCapaOculta2
    
    def backpropagation(self,entrenamiento,errorPermitido,entradaDeseada,salidaDeseada,numeroNeuronasCapaOculta1,numeroNeuronasCapaOculta2):
        
        pesosCapaOculta = np.random.rand(numeroNeuronasCapaOculta1,len(entradaDeseada[0])) #np.array([[0.3568718 ,  0.6821255],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969]])#pesos de la capa oculta, deben ser aleatorios
        pesosCapaOculta2 = np.random.rand(numeroNeuronasCapaOculta2,numeroNeuronasCapaOculta1) #pesos de la capa oculta 2
        pesosCapaSalida = np.random.rand(numeroNeuronasCapaOculta2)  #np.array([0.69900,0.50459,0.5859,0.4789])# pesos de la capa de salida
        
        elementos = np.random.permutation(math.ceil(len(entradaDeseada)*0.8)) # lista desordenada para el entrenamiento
        #elementos = [0,1,2,3] # orden temporal mientras las pruebas
        #print (elementos)
        
        it = 0
        seguir = True
        while seguir:
            for i in elementos:
                it +=1
                pesosCapaOcultaActual = pesosCapaOculta
                pesosCapaOculta2Actual = pesosCapaOculta2
                pesosCapaSalidaActual = pesosCapaSalida
                patronIn = entradaDeseada[i]
                patronOut = salidaDeseada[i]
                #print(patronIn,patronOut)
                salidaObtenida,salidaCapaOculta,salidaCapaOculta2,entradaNetaCapaSalida,entradaNetaCapaOculta,entradaNetaCapaOculta2 = self.RedNeuronal(patronIn, pesosCapaOcultaActual,pesosCapaOculta2Actual, pesosCapaSalidaActual)
                #salidaCapaOculta es la salida de la capa oculta
                #salidaObtenida es la salida de la cada de entrada
                
                #Miramos la neurona de la capa de salida. Como es una solo la miramos a ella
                errorCapaDeSalida = (patronOut - salidaObtenida)*self.derivada(entradaNetaCapaSalida)
                #Actualizamos los pesos de la capa de salida
                
                for j in range(len(pesosCapaSalida)): #actializacion de los pesos en la capa de salida 
                    pesosCapaSalida[j] += entrenamiento*errorCapaDeSalida*salidaCapaOculta2[j]
                
                #calculo de y actualizacion de pesos de la capa oculta
                # en estos errores no hay que hacer una sumatoria? 
                errorNeuronasCapaOculta2 = []
                for j in range(len(entradaNetaCapaOculta2)):
                    errorNeuronasCapaOculta2.append(self.derivada(entradaNetaCapaOculta2[j])*pesosCapaSalidaActual[j]*errorCapaDeSalida)
                    for k in range(len(pesosCapaOculta2[j])):
                        pesosCapaOculta2[j][k] += entrenamiento*errorNeuronasCapaOculta2[j]*salidaCapaOculta[k]

                for j in range(len(entradaNetaCapaOculta)):
                    errorNeuronaCapaOculta1 = 0
                    for k in range(len(errorNeuronasCapaOculta2)):
                        errorNeuronaCapaOculta1 += pesosCapaOculta2Actual[k][j]*errorNeuronasCapaOculta2[k]
                    errorNeuronaCapaOculta1 = errorNeuronaCapaOculta1*self.derivada(entradaNetaCapaOculta[j])
                    for k in range(len(pesosCapaOculta[j])):
                        pesosCapaOculta[j][k] += entrenamiento*errorNeuronaCapaOculta1*patronIn[k]
                
    
                #Calculamos el error
                error = 0
                mejorError=99999999
                for j in range(len(entradaDeseada)):
                    entrada = entradaDeseada[j]
                    salidaEsperada = salidaDeseada[j]
                    salida,a,b,c,d,e = self.RedNeuronal(entrada, pesosCapaOculta,pesosCapaOculta2, pesosCapaSalida)
                    error += 0.5*(salida - salidaEsperada)**2
                    
                print("error global: {}".format(error))
                if(error<mejorError):
                    F = open("matriz","w")
                    F.write("pesos capa oculta 1")
                    F.write(pesosCapaOculta)
                    F.write("pesos capa oculta 2")
                    F.write(pesosCapaOculta2)
                    F.write("pesos capa salida")
                    F.write(pesosCapaSalida)
                    F.write("mejor error")
                    F.write(mejorError)
                    F.close()
                if(error <errorPermitido):
                    print("salio")
                    seguir = False
                    break
        print("numero de iteraciones: {}".format(it))        
        print("pesos optimos son: pesos capa oculta1: {} \n pesos capa oculta2: {}  \n pesos capa salida: {} ".format(pesosCapaOculta,pesosCapaOculta2, pesosCapaSalida))
        while True:
            entrada1 = int(input("entrada1"))
            entrada2 = int(input("entrada2"))
            salida,a,b,c,d,e = self.RedNeuronal([entrada1,entrada2], pesosCapaOculta,pesosCapaOculta2, pesosCapaSalida)
            print("salida: {}".format(salida))
        return pesosCapaOculta,pesosCapaOculta2,pesosCapaSalida

    def RedNeuronal1CapaOculta(self,entrada,pesosCapa2,PesosCapa3):
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
    
    def backpropagation1CapaOculta(self,entrenamiento,errorPermitido,entradaDeseada,salidaDeseada,numeroNeuronasCapaOculta):
        
        pesosCapaOculta = np.random.rand(numeroNeuronasCapaOculta,len(entradaDeseada[0])) #np.array([[0.3568718 ,  0.6821255],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969]])#pesos de la capa oculta, deben ser aleatorios
        pesosCapaSalida = np.random.rand(numeroNeuronasCapaOculta)  #np.array([0.69900,0.50459,0.5859,0.4789])# pesos de la capa de salida
        
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
                salidaObtenida,salidaCapaOculta,entradaNetaCapaOculta,entradaNetaCapaSalida = self.RedNeuronal1CapaOculta(patronIn, pesosCapaOcultaActual, pesosCapaSalidaActual)
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
                mejorError=99999
                for j in range(len(entradaDeseada)):
                    entrada = entradaDeseada[j]
                    salidaEsperada = salidaDeseada[j]
                    salida,a,b,c = self.RedNeuronal1CapaOculta(entrada, pesosCapaOculta, pesosCapaSalida)
                    error += 0.5*(salida - salidaEsperada)**2
                    
                print("error global: {}".format(error))
                if(error<mejorError):
                    mejorError = error
                    F = open("matriz.txt","w")
                    F.write("pesos capa oculta 1")
                    for m in range(len(pesosCapaOculta)):
                            F.write(str(pesosCapaOculta[m]))
                    F.write("pesos capa salida")
                    F.write(str(pesosCapaSalida))
                    F.write("mejor error")
                    F.write(str(mejorError))
                    F.close()
                if(error <errorPermitido):
                    print("salio")
                    seguir = False
                    break
                    
                #print("error capa de salida: {}".format(errorCapaDeSalida))
                #print("pesos capa de salida actualizados: {}".format(pesosCapaSalida))
                #print("pesos capa de oculta actualizados: {}".format(pesosCapaOculta))
        print("numero de iteraciones: {}".format(it))
        print("pesos optimos son: pesos capa oculta: {}  \t pesos capa salida: {} ".format(pesosCapaOculta, pesosCapaSalida))
        while True:
            entrada1 = int(input("entrada1"))
            entrada2 = int(input("entrada2"))
            salida,a,b,c = self.RedNeuronal1CapaOculta([entrada1,entrada2], pesosCapaOculta, pesosCapaSalida)
            print("salida: {}".format(salida))
        return pesosCapaOculta,pesosCapaSalida
    
RedNeuronal()
