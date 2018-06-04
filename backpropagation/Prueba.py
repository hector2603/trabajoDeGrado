'''
Created on 12/10/2017

@author: hector
'''
import sys
sys.path.append('../')
sys.path.append('/home/ec2-user/.local/lib/python3.6/site-packages')
from pylab import *
import math
import random
import numpy as np
import time
from Datos.Datos import Datos

class RedNeuronal(object):
    '''
    classdocs
    '''
    '''
    factorEntrenamiento = 0.1 # factor de entramiento de la red
    errorPermitido = 0.1 # error permitido para la condicion de parada 
    datos = 0 # clase Datos, la cual guarda el archivo de texto con los datos
    entradaDeseada = 0 # matriz con las entradas de la red
    salidaDeseada = 0 # arreglo con las salidas de las entradas de la red
    pesosCapaOculta = 0 # pesos de la primer capa oculta 
    pesosCapaOculta2 = 0 # pesos de la segunda capa oculta
    pesosCapaSalida = 0 # pesos de la capa de salida
    matrizDeConfusion = [] # matriz de confusion
    error = 0 # error global en la iteracion 
    
    '''

    def __init__(self):
        self.factorEntrenamiento = 0.1;
        self.errorPermitido = 0.1;
        self.datos = Datos()
        self.datos.datosConBinarios()
        self.entradaDeseada = self.datos.Datos
        self.salidaDeseada = self.datos.Resultado
        self.biasCapaOculta = 1
        self.biasCapaSalida = 1
        #self.elementosPrueba = [int(random.random()*len(self.entradaDeseada)) for x in range(math.ceil(len(self.entradaDeseada)*0.2))]
        self.elementosPrueba = self.entradaDeseada
        #print(self.sigmoide(-4*1000000*-0.0641515994108))
        #pesosCapaOculta,pesosCapaSalida = self.backpropagation(self.factorEntrenamiento, self.errorPermitido, self.entradaDeseada, self.salidaDeseada,19,10)
        #pesosCapaOculta,pesosCapaSalida = self.backpropagation1CapaOculta(self.factorEntrenamiento, self.errorPermitido, self.entradaDeseada, self.salidaDeseada,38)
        #self.ProbarModelo()

    def sigmoide(self,x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.000000001
    def derivadaSigmoide(self,x):
        return self.sigmoide(x)*(1-self.sigmoide(x))
    def relu(self,x):
        return x if x>0 else 0
    def derivadaRelu(self,x):
        return 1 if x>0 else 0
    def funcionDeActivacion(self, x):
        return self.sigmoide(x)
    def derivada(self, x):
        return self.derivadaSigmoide(x)
    
    def RedNeuronal(self,entrada):
        #capa oculta 1
        tendenciaCapa1= 1
        entradaNetaCapaOculta1 = self.pesosCapaOculta.dot(entrada)
        entradaNetaCapaOculta1 = [x+tendenciaCapa1 for x in entradaNetaCapaOculta1]
        salidaCapaOculta1 = [self.funcionDeActivacion(x) for x in entradaNetaCapaOculta1]
        
        #capa oculta 2
        tendenciaCapa2 = 1
        entradaNetaCapaOculta2 = self.pesosCapaOculta2.dot(salidaCapaOculta1)
        entradaNetaCapaOculta2 = [x+tendenciaCapa2 for x in entradaNetaCapaOculta2]
        salidaCapaOculta2 = [self.funcionDeActivacion(x) for x in entradaNetaCapaOculta2]
        
        #capa de salida
        tendenciaCapaSalida = 1
        entradaNetaCapaSalida = self.pesosCapaSalida.dot(salidaCapaOculta2)
        entradaNetaCapaSalida = entradaNetaCapaSalida+tendenciaCapaSalida
        salidaTotal =   self.funcionDeActivacion(entradaNetaCapaSalida)
        #print(" entrada: {} \n entrada capa oculta: {} \n salida capa oculta: {} \n entrada neta capa salida: {} \n salida total: {}".format(entrada,entradaNetaCapaOculta1,salidaCapaOculta1,entradaNetaCapaSalida,salidaTotal))
        return salidaTotal,salidaCapaOculta1,salidaCapaOculta2,entradaNetaCapaSalida,entradaNetaCapaOculta1,entradaNetaCapaOculta2
    
    def backpropagation(self,entrenamiento,errorPermitido,entradaDeseada,salidaDeseada,numeroNeuronasCapaOculta1,numeroNeuronasCapaOculta2):
        
        self.pesosCapaOculta = pesosCapaOculta = np.random.rand(numeroNeuronasCapaOculta1,len(entradaDeseada[0])) #np.array([[0.3568718 ,  0.6821255],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969]])#pesos de la capa oculta, deben ser aleatorios
        self.pesosCapaOculta2 = pesosCapaOculta2 = np.random.rand(numeroNeuronasCapaOculta2,numeroNeuronasCapaOculta1) #pesos de la capa oculta 2
        self.pesosCapaSalida = pesosCapaSalida = np.random.rand(numeroNeuronasCapaOculta2)  #np.array([0.69900,0.50459,0.5859,0.4789])# pesos de la capa de salida
        
        elementos = np.random.permutation(math.ceil(len(entradaDeseada)*1)) # lista desordenada para el entrenamiento
        #elementos = [0,1,2,3] # orden temporal mientras las pruebas
        #print (elementos)
        mejorError=99999999
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
                salidaObtenida,salidaCapaOculta,salidaCapaOculta2,entradaNetaCapaSalida,entradaNetaCapaOculta,entradaNetaCapaOculta2 = self.RedNeuronal(patronIn)
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
                    errorNeuronaCapaOculta1 = pesosCapaOculta2Actual.T.dot(errorNeuronasCapaOculta2)
                    '''for k in range(len(errorNeuronasCapaOculta2)):
                        errorNeuronaCapaOculta1 += pesosCapaOculta2Actual[k][j]*errorNeuronasCapaOculta2[k]
                    errorNeuronaCapaOculta1 = errorNeuronaCapaOculta1*self.derivada(entradaNetaCapaOculta[j])'''
                    for k in range(len(pesosCapaOculta[j])):
                        pesosCapaOculta[j][k] += entrenamiento*errorNeuronaCapaOculta1[j]*patronIn[k]
                
                self.pesosCapaOculta = pesosCapaOculta
                self.pesosCapaOculta2 = pesosCapaOculta2
                self.pesosCapaSalida = pesosCapaSalida
                
                #Calculamos el error
                #Calculamos el error
                self.generarMatrizDeConfusion(1,2)
                self.calcularPrecision(1)
                    
                print("error global: {}".format(self.error))
                #print("matriz de confusion")
                #print(self.matrizDeConfusion[0])
                #print(self.matrizDeConfusion[1])
                print("precision: {}".format(self.precision))

                
                if(True):
                    mejorError = self.error
                    F = open("matriz.txt","w")
                    F.write("pesos capa oculta 1")
                    F.write(str(pesosCapaOculta))
                    F.write("pesos capa oculta 2")
                    F.write(str(pesosCapaOculta2))
                    F.write("pesos capa salida")
                    F.write(str(pesosCapaSalida))
                    F.write("mejor error")
                    F.write(str(mejorError))
                    F.close()
                if(self.error <errorPermitido):
                    print("salio")
                    seguir = False
                    break
        print("numero de iteraciones: {}".format(it))        
        print("pesos optimos son: pesos capa oculta1: {} \n pesos capa oculta2: {}  \n pesos capa salida: {} ".format(pesosCapaOculta,pesosCapaOculta2, pesosCapaSalida))
        while True:
            entrada1 = int(input("entrada1"))
            entrada2 = int(input("entrada2"))
            salida,a,b,c,d,e = self.RedNeuronal([entrada1,entrada2])
            print("salida: {}".format(salida))
        return pesosCapaOculta,pesosCapaOculta2,pesosCapaSalida

    def RedNeuronal1CapaOculta(self,entrada):
        #capa oculta
        #tendenciaCapa1= 1
        entradaNetaCapaOculta = self.pesosCapaOculta.dot(entrada)
        entradaNetaCapaOculta = [x+self.biasCapaOculta for x in entradaNetaCapaOculta]
        salidaCapaOculta = [self.funcionDeActivacion(x) for x in entradaNetaCapaOculta]
        #capa de salida
        #tendenciaCapa2 = 1
        entradaNetaCapaSalida = self.pesosCapaSalida.dot(salidaCapaOculta)
        entradaNetaCapaSalida = entradaNetaCapaSalida+self.biasCapaSalida
        salidaTotal =   self.funcionDeActivacion(entradaNetaCapaSalida)
        #print(" entrada: {} \n entrada capa oculta: {} \n salida capa oculta: {} \n entrada neta capa salida: {} \n salida total: {}".format(entrada,entradaNetaCapaOculta,salidaCapaOculta,entradaNetaCapaSalida,salidaTotal))
        return salidaTotal,salidaCapaOculta,entradaNetaCapaOculta,entradaNetaCapaSalida
    
    def backpropagation1CapaOculta(self,entradaDeseada,salidaDeseada,numeroNeuronasCapaOculta,entrenamiento=0.1,errorPermitido=0.1):
        
        self.pesosCapaOculta = pesosCapaOculta = np.random.rand(numeroNeuronasCapaOculta,len(entradaDeseada[0])) #np.array([[0.3568718 ,  0.6821255],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969]])#pesos de la capa oculta, deben ser aleatorios
        self.pesosCapaSalida = pesosCapaSalida = np.random.rand(numeroNeuronasCapaOculta)  #np.array([0.69900,0.50459,0.5859,0.4789])# pesos de la capa de salida
        
        elementos = [int(random.random()*len(self.entradaDeseada)) for x in range(math.ceil(len(self.entradaDeseada)*0.8))]
        #elementos = np.random.permutation(math.ceil(len(entradaDeseada)*0.8)) # lista desordenada para el entrenamiento
        #elementos = [0,1,2,3] # orden temporal mientras las pruebas
        #print (elementos)
        mejorError=99999
        it = 0
        inicial = time.time()
        actual = time.time()
        limite = inicial + 900
        errorGrafica = [] # guarda el error
        precisionGrafica = [] # guarda la precision
        while actual<=limite:
            for i in elementos:
                it +=1
                pesosCapaOcultaActual = pesosCapaOculta
                pesosCapaSalidaActual = pesosCapaSalida
                patronIn = entradaDeseada[i]
                patronOut = salidaDeseada[i]
                #print(patronIn,patronOut)
                salidaObtenida,salidaCapaOculta,entradaNetaCapaOculta,entradaNetaCapaSalida = self.RedNeuronal1CapaOculta(patronIn)
                #salidaObtenida es la salida de la capa oculta
                #salidaCapaOculta es la salida de cada de entrada
                
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
                        #pesosCapaOculta[j][k] += entrenamiento*errorNeuronacapaOculta*patronIn[k]
                
                self.pesosCapaOculta = pesosCapaOculta
                self.pesosCapaSalida = pesosCapaSalida
                
                #Calculamos el error
                self.generarMatrizDeConfusion()
                self.calcularPrecision(1)
                #print("precision todos los datos ")
                pricisionTotal =  self.precision
                #print(pricisionTotal)
                precisionGrafica.append(pricisionTotal)
                self.generarMatrizDeConfusion()
                self.calcularPrecision(0.2)
                    
                print("iteracion: {} -- error global: {} -- precisión: {}".format(it,self.error,pricisionTotal))
                if((it%100)==0):
                    errorGrafica.append(self.error)
                #print("matriz de confusion")
                #print(self.matrizDeConfusion[0])
                #print(self.matrizDeConfusion[1])
                #print("precision: {}".format(pricisionTotal))
                
                if(self.error<mejorError):
                    mejorError = self.error
                    F = open("matrizBackpropagation"+str(numeroNeuronasCapaOculta)+".txt","w")
                    F.write("pesos capa oculta 1")
                    for m in range(len(pesosCapaOculta)):
                            F.write(str(pesosCapaOculta[m]))
                    F.write("\n pesos capa salida")
                    F.write(str(pesosCapaSalida))
                    F.write("\n mejor error")
                    F.write(str(mejorError))
                    F.write("\n precision con todos los datos:")
                    F.write(str(pricisionTotal))
                    F.close()
                if(self.error <errorPermitido):
                    print("salio")
                    seguir = False
                    break
                    
                #print("error capa de salida: {}".format(errorCapaDeSalida))
                #print("pesos capa de salida actualizados: {}".format(pesosCapaSalida))
                #print("pesos capa de oculta actualizados: {}".format(pesosCapaOculta))
                actual = time.time()
                
        #grafica del mejor error de cada generacion
        plt.plot(errorGrafica)
        plt.title("Error")
        plt.xlabel("iteración")
        plt.ylabel("ECM")
        plt.savefig("error"+str(numeroNeuronasCapaOculta)+".png")
        plt.show()
        plt.cla()   # Borrar información de los ejes
        plt.clf()   # Borrar un gráfico completo
        
        #grafica de la precisión obtenida hasta el momento
        '''plt.plot(precisionGrafica)
        plt.title("Precisión con todos los datos ")
        plt.ylabel("ECM")
        plt.savefig("mejorError"+str(numeroNeuronasCapaOculta)+".png")
        plt.cla()   # Borrar información de los ejes
        plt.clf()   # Borrar un gráfico completo'''
         
        '''print("numero de iteraciones: {}".format(it))
        print("pesos optimos son: pesos capa oculta: {}  \t pesos capa salida: {} ".format(pesosCapaOculta, pesosCapaSalida))
        while True:
            entrada1 = int(input("entrada1"))
            entrada2 = int(input("entrada2"))
            salida,a,b,c = self.RedNeuronal1CapaOculta([entrada1,entrada2], pesosCapaOculta, pesosCapaSalida)
            print("salida: {}".format(salida))
        return pesosCapaOculta,pesosCapaSalida'''
        return errorGrafica
    
    # funcion que genera la matriz de confusion
    # porcentajePrueba, el porcentaje de datos del dataset que se van a usar para las pruebas
    # numeroCapas =  1 para cuando el numero de capas ocultas es una, 2 si el numero de capas ocultas son dos
    def generarMatrizDeConfusion(self,porcentajePrueba = 0,numeroCapas=1):
        self.matrizDeConfusion = [[0,0],[0,0]]
        self.y_resultado = []
        if porcentajePrueba == 0:
            elementos = self.elementosPrueba
        else:
            elementos = np.random.permutation(math.ceil(len(self.entradaDeseada)*porcentajePrueba)) # lista desordenada para el test
        self.error=0
        for (j,l) in enumerate(elementos):
            entrada = self.entradaDeseada[j]
            salidaEsperada = self.salidaDeseada[j]
            if numeroCapas == 1:
                salida,a,b,c = self.RedNeuronal1CapaOculta(entrada)
            else:
                salida,a,b,c,d,e = self.RedNeuronal(entrada)
            self.y_resultado.append(salida)
            self.error += 0.5*(salida - salidaEsperada)**2
            self.matrizDeConfusion[round(salida)][salidaEsperada] += 1
            #print("salida {} salidadeseada {}".format(salida,salidaEsperada))
            
    def calcularPrecision(self,porcentajePrueba):
        total = math.ceil(len(self.entradaDeseada)*porcentajePrueba)
        self.precision = (self.matrizDeConfusion[0][0] + self.matrizDeConfusion[1][1])/total
        self.specificity = self.matrizDeConfusion[0][0] / (self.matrizDeConfusion[0][0] + self.matrizDeConfusion[1][0])
        self.sensitivity = self.matrizDeConfusion[1][1] / (self.matrizDeConfusion[1][1] + self.matrizDeConfusion[0][1])
        
        
    def ProbarModelo(self,pesosCapaOculta,pesosCapaSalida,biasCapaOculta,biasCapaSalida):
        self.pesosCapaOculta = pesosCapaOculta
        self.pesosCapaSalida = pesosCapaSalida
        self.biasCapaOculta = biasCapaOculta
        self.biasCapaSalida = biasCapaSalida
        self.generarMatrizDeConfusion(1)
        self.calcularPrecision(1)
        '''print("error global: {}".format(self.error))
        print("matriz de confusion")
        print(self.matrizDeConfusion[0])
        print(self.matrizDeConfusion[1])
        print("precision")
        print(self.precision)'''
        return self.error
    
    def ProbarModeloBackpropagation(self):
        self.pesosCapaOculta = np.array([[0.62972763,-0.33723407,-1.08255794,0.02724721,0.57767761,-0.2487474,-0.69581832,1.45393275,0.67232272,0.17904853,-0.59724445,1.9636341,0.22386637,-1.18376154,0.90769228,1.53064995,0.52666167,0.397916,0.04875601,2.8269017,0.88177378,2.13350652,0.4562378,1.13073135,0.76204806,-0.2328692,1.61166143,-1.12962299,1.98955297,1.09249102,1.363435,0.53469893,1.01644712,2.73835149,1.77346195,-0.46331771,3.12786825,-0.15702667,2.72486875,-0.27585965,2.74799486,2.18629686,1.95426053,1.10296117,1.20602334,1.64577516,0.06937628,0.86685781,-1.02739663,0.61893413,2.4117933,0.7923125,0.85836492,0.48032376,2.20176826],[0.11002702,0.26355545,2.7755299,-0.11942971,1.40115458,0.1694512,-0.9970919,1.50814064,-1.08172959,2.2629862,0.71247446,-0.61198922,1.49034432,1.59147045,0.64901848,-0.47829422,0.74017721,2.14120731,0.25200953,0.45950122,1.72653862,1.52732597,0.0271832,0.93208941,2.40786394,-0.75393802,0.69587332,-0.31382507,-0.29799765,1.54206456,0.34039248,0.83011701,1.25519273,-0.43174128,1.66407018,1.25052775,0.40838819,0.32182125,1.84022993,-0.34464222,-0.09565023,1.05782243,0.51934702,1.1146469,-0.60725702,0.36412315,0.03909931,0.98619219,0.11432104,0.23907546,0.97502534,1.53553664,0.95120314,0.84394429,1.49509471],[-0.20007113,0.47518662,1.65679547,1.86611951,2.2328,-0.18619276,1.72817849,2.24173486,1.34957083,1.14849177,1.70313182,0.81186416,-1.16348676,3.14585685,1.0527425,-0.13138222,1.95382632,0.09298333,0.58411299,1.53184131,-1.34936091,0.84732175,2.36862402,1.07873451,-0.84114243,-0.02529284,0.68675924,-0.20288779,-0.23302494,2.72355292,0.08246072,0.73021548,-1.59688761,1.65937712,0.05160105,-0.21427971,0.56172815,0.67193657,0.11947396,0.67053609,0.51292926,2.79298937,-2.01411233,1.58356626,2.90491901,-0.09337754,0.55808316,0.7717971,0.33345068,0.81119372,0.48847137,1.71884915,0.87986419,0.2240618,0.56092157],[-0.48335391,0.89720804,1.33456048,1.59543292,1.24616661,0.3687664,1.08398288,1.15346702,2.32213568,1.39053511,0.80772711,0.10440265,0.75139024,0.88314889,-0.06541093,-0.54869444,0.61899383,1.04347926,-0.01135654,1.32774952,-0.32829638,1.68952593,-0.12195938,0.52088412,0.82956322,0.75525991,0.37684538,2.01749167,0.27113515,-1.59925246,0.85367277,0.51174563,2.6083158,0.18796788,-0.3367419,0.75131395,0.14682224,0.99699337,0.92184029,0.416978,-0.319526,0.56549156,1.6764084,0.36076527,0.10178246,-1.01709166,0.81574258,0.95292761,-0.62973477,0.39508574,0.95069075,1.32018319,0.32136882,0.46798864,1.29895503],[-0.72321154,0.77648174,2.70996005,0.41298562,-0.91573388,0.8062967,-0.69281056,-0.49396912,1.0409624,0.68995345,2.22683458,0.76108751,-0.01483403,1.31529747,-0.52164118,-0.4266642,0.75665064,0.09848374,1.16403503,-0.21652293,0.03217823,0.46727784,2.37191519,-0.10611002,-0.38458396,0.64572401,1.99112519,1.98839092,1.66210131,0.19786008,0.8348327,-1.38238952,0.60409842,-0.33980274,0.08599113,-1.48951183,0.77519512,0.70897432,0.38919703,0.85034977,0.75699055,-0.62457075,1.52882529,0.53712649,-0.69874511,-0.82765434,-0.5554365,1.45224308,1.08354984,1.05446045,0.61162248,0.74382059,0.91476151,0.82155781,0.38930058],[1.59081291,1.76074136,1.87087983,2.14642674,1.05799474,-0.85977135,1.94542949,1.6466357,1.77714612,0.66735662,1.54437663,0.99792642,1.46806566,-1.99213895,0.5723292,0.65313283,1.47062164,1.17161291,-0.30992064,0.61362338,-0.52596987,-0.36345721,1.64042908,1.01692951,-1.0006965,1.04121579,-0.00479114,2.64756791,-0.54829995,1.18612114,0.34269364,-1.04345464,1.47593906,-0.15815237,2.12080919,0.75981346,-0.1304503,-0.18487782,0.18723494,2.10879255,1.91631716,1.8163198,2.60312088,1.33988792,1.38665325,-0.01031081,0.28449583,-0.34996379,-0.18753698,-0.5150921,-0.19250171,0.79424164,0.70900762,0.8262412,0.93143438],[1.33956849,0.59497364,0.77886039,1.08081842,-0.77063863,-0.70688278,1.55235245,2.17086833,-0.18999941,1.44469166,0.28232654,-0.15239269,1.95360944,1.83249767,2.03101095,-0.3678401,1.64169679,-0.36348736,0.41394412,-1.04313542,-1.78743932,3.1924236,0.3264953,1.04233678,0.70727947,2.3326002,-0.18196411,1.14222918,1.16862835,0.1824333,1.39862425,0.43807343,0.62274741,0.3296913,-0.13873598,1.21273701,-0.92601648,0.16498764,0.10011755,2.32393981,-0.37646368,1.57786828,-0.95976609,-0.42067329,1.99389989,1.6253411,0.38983225,0.8365269,0.7359938,1.6482715,0.50980488,0.72619334,0.3538259,0.59127896,0.10707699],[0.80380272,2.02653561,-0.39726214,0.0446585,-1.12423855,0.85499735,1.56518337,0.10166213,1.50575345,-0.57107415,-0.44884482,0.61885402,2.23386227,0.32156441,-0.09341366,1.09413817,0.04590635,0.32538907,3.00267257,1.31262555,0.69691283,0.33691542,-0.32844524,0.9952698,1.48304903,0.55914917,1.07319727,1.33401799,-0.92409961,-1.37425054,1.44599835,0.77269967,0.1212827,-0.65104894,1.16848863,0.9875591,-1.9779509,0.5364244,1.72031785,1.17043161,2.24480145,1.2277942,2.71539999,2.289536,-0.77144885,2.42770833,1.60592057,1.12153161,0.57246076,-0.40491969,-0.00993519,0.28511704,0.51178899,0.6225254,-1.81519876],[0.63420613,-0.75080912,0.87066859,0.49635251,2.46815068,0.4098518,-1.62402862,1.30940206,-0.4221099,0.9859362,0.56607067,1.0387536,0.47348342,1.00224105,0.26048816,1.79947175,0.51940717,-0.25171927,0.54124555,2.41302782,1.96471269,-0.65584922,0.19996935,0.49277859,0.1100048,1.18771136,1.29327344,1.27251162,0.20575265,0.36889731,0.79439199,0.77141003,0.31221922,0.27921062,0.4380318,-2.10514603,0.607217,0.74616901,-0.02556221,1.71832931,0.98290328,0.56889853,-1.6656339,-0.00358854,1.36096782,0.42444006,0.1457114,0.6225656,-0.18626712,0.05271653,1.07551351,0.19253434,0.98118868,0.55701347,0.73758893],[-0.8278623,-0.17668815,-0.30701783,-0.03121358,-0.00359238,-0.06876695,0.95864154,0.69326578,1.03037488,0.15933364,0.57807784,0.91669418,0.70461556,0.64658529,0.00757632,3.30948762,-0.35005523,-0.92560166,0.15157444,1.62221497,1.84445172,0.52808757,2.57718274,0.89328396,0.42191722,0.35114258,1.03179241,2.02210344,-0.38679019,-0.52429388,0.9840805,1.78926434,1.49782942,1.51741235,0.94087476,1.89984007,0.6980188,0.9052104,1.2666005,1.8712523,0.14293573,0.56583509,0.89091332,1.96911818,-0.8463918,1.15614359,0.81528236,1.5880498,0.47277021,1.96237915,-0.86981329,-0.43034789,0.17615137,0.96678589,2.10136857],[1.34075033,-1.33715072,2.01126173,-0.00688274,0.28764881,-0.260743,2.39428275,2.07941422,-0.82979388,1.36346142,0.14625602,0.66237472,-0.34195018,1.73421789,-0.03933327,2.97833986,0.5883614,1.09629242,-1.3349231,1.71682815,-1.25157951,-1.54438058,-2.07684302,-0.52821224,1.16391747,1.04630803,-1.30932793,-0.47440596,2.68425567,0.38765448,0.31184629,1.23043691,-1.00160313,1.1712577,-0.10147886,2.18270879,0.45188533,0.44804453,0.3549314,0.49843628,-0.42009463,1.62670877,2.25656939,0.11811388,-0.14826327,0.15755337,-0.09590552,0.20970114,0.25396569,0.94650227,-0.27907858,0.16462798,0.53129268,0.97465731,0.78578396],[0.34592969,-0.20834099,0.96213641,0.93707031,-0.82333321,1.73751854,-0.15613746,0.27633407,3.8995425,0.40753841,1.38399932,0.99180078,0.04372506,0.64377366,0.31214858,-0.4185949,0.22968352,1.10689319,2.08406017,-0.09283607,2.54979857,2.20884545,-2.88686114,0.77290879,0.10976391,-1.42050928,0.67752166,3.15993606,0.65996168,0.76768141,0.06558643,0.92246347,-0.53298774,-0.14074604,0.62228049,-0.28559769,0.6722718,0.5707168,0.52500428,-0.46886363,0.4996984,0.65392159,2.02165963,1.16242968,3.68932869,-1.14551561,1.03562664,-2.65527214,-0.7713427,0.65527874,0.24559192,1.00651215,0.64905846,0.77108564,-0.34236371],[0.90297104,0.14143566,-0.29471783,1.68256678,0.7745197,1.74211845,-1.02882274,-0.46133599,-0.4769613,0.97422258,1.24328567,2.45545919,0.06267022,0.44251875,-0.0927202,-0.28209238,0.66825053,1.31619756,0.54202294,0.52537316,1.05709371,1.27170969,-1.72396232,1.29391847,0.68158832,0.70290869,-0.06869774,0.98844077,1.55972875,0.6210778,1.16188375,0.95915245,0.90269033,-0.29299538,0.08961142,-0.85651818,0.97053842,0.16231838,0.64271002,1.4190606,0.17438914,0.75430852,0.02935225,1.20147882,-0.15445821,0.22096343,0.57439529,-0.27815316,1.628692,0.82372855,-0.79541166,-0.09423679,0.91868981,0.10152649,0.71476458],[0.26772135,0.07042521,0.71813155,0.75683755,2.51830125,0.06091144,0.33721563,-0.36373065,0.98501453,0.52682579,1.14607215,1.87169123,1.86389015,0.96466463,-0.81276581,1.68155082,1.01291614,0.15788093,0.76004869,0.62848447,1.32536258,0.69460097,1.60067763,0.52168061,0.51918252,0.92392892,0.93840613,-0.06987614,-1.61115514,0.14287956,0.89023939,0.41235886,0.39786715,-0.18844928,2.06737305,0.39697418,0.70490279,0.43875149,-0.07421018,0.31251646,1.25433618,0.29044152,0.98506388,1.29431399,-0.36138301,-0.32950295,-0.37600338,0.49348481,0.11762595,-0.72447073,0.50262683,1.24687684,0.36073037,0.16967807,-0.11173837],[0.29251456,0.16717115,1.9919551,-1.08654902,0.21835299,1.73450039,0.34270423,-0.51782635,-1.37781377,1.43169096,-0.37252217,-0.40412067,0.12340524,-0.93280731,0.37346704,0.67230667,0.09203754,1.70390114,0.06196267,-0.90160776,-0.70031906,-0.63130626,2.17391107,0.53581672,1.53093382,-0.47412399,0.16555712,-1.83637857,1.56093048,-2.2384612,-0.51790635,-0.58897475,1.90002188,1.52589066,-1.1263222,1.00481078,-1.3874534,-0.41399405,1.42143185,1.28079228,0.47247451,0.67883513,0.84380946,0.97471161,0.84116826,-0.37550498,0.58939029,-0.00373572,0.08294536,0.33982994,2.65923209,0.50619328,0.72160415,0.08751195,0.30486405],[0.70434071,0.02625384,0.68340944,1.2841442,-0.98237799,-0.21521718,2.05292736,1.98984587,0.92574012,1.58270743,0.89178287,0.42220534,-0.5537176,1.65922071,0.35488821,0.78835043,0.92140881,-0.11036304,0.97370588,-0.02728739,0.38108425,-0.04401823,3.21655496,0.53577621,1.14504601,-0.04891532,2.62788569,1.58227265,0.70902083,-0.58982236,0.37530446,1.95448912,1.62399816,0.39816483,0.70186266,0.67263052,1.65901543,1.23267692,0.05873289,-0.25418087,0.20833793,-1.12575572,0.02953825,0.41245629,0.74874594,1.52068664,0.57231343,0.43158492,-0.02625321,0.76822803,0.51994109,1.62026104,0.49947618,0.49476606,-0.9360265,],[-0.23588812,-1.21876831,0.74401728,1.27654266,0.42119259,1.21689786,1.27627796,-1.16118015,-0.31326697,0.21020445,1.39964821,0.57405135,0.34782133,0.30190917,0.91741465,1.55575885,0.95531134,0.1138018,1.4610987,0.23180273,-1.90680742,3.11570147,-0.74488606,0.50267045,-0.7736957,0.72579407,0.56193328,0.36510859,-0.35555339,0.24664969,0.60967649,-0.85912145,0.89747686,0.63332119,-0.25664819,0.27439024,0.01163045,1.35243326,1.69464164,-1.74133494,-0.28508312,-1.25599039,-0.9746563,-0.468629,1.49331905,-0.4080355,0.38276094,1.65759231,1.08952836,1.6138047,0.17545872,0.31062929,0.56647155,0.00490186,-1.43724625],[0.44147631,3.88408863,-0.14553955,0.19787708,-0.98100461,-1.83632926,0.43096895,2.58356049,1.26404819,-0.30465665,-0.18236218,1.47302555,0.0649027,2.09706202,-3.52195437,-1.36971494,-0.69037603,1.60604968,-0.45232728,0.59427226,-2.0696163,0.50161793,-0.41164916,0.71301821,0.65875973,-0.02389045,-0.91802027,1.41757486,-2.18555514,5.95682588,1.22791847,1.22105835,-0.1427802,-1.12176142,-2.32219324,-1.51344766,-1.77199761,-0.10004421,1.76208001,0.53421353,-1.68853604,2.26837991,1.81632578,0.06576051,-0.72706655,1.98062788,-0.38241743,0.79673414,-4.83553076,0.16133476,-2.18583462,-2.07204554,0.64613666,0.60312798,0.68905415],[0.44583652,1.37697813,-0.71582944,1.21551862,0.53122446,0.431759,-1.08516982,-0.19133894,2.27269092,0.68161667,0.63909129,1.14897438,0.83613265,0.32343275,-0.58565989,1.01566185,1.20154615,0.76695403,1.62248673,-0.09695148,0.87479709,-1.23004118,0.14437555,2.40277865,0.8669818,0.14578078,-0.67093789,0.12926904,-0.41819837,1.89524541,1.91210376,1.05818059,2.64889132,0.7012283,1.18175579,0.15615029,-0.03303959,1.40863952,0.81138973,-0.81482309,-0.1466386,-0.3958516,1.96702951,0.27313907,0.03400407,1.69913033,1.41359028,1.45474843,0.93245488,1.05435722,-0.86334827,-0.06298241,0.23542136,0.95759291,0.30197674]])
        self.pesosCapaSalida = np.array([-6.94335683,5.45412349,5.78930716,3.92492698,5.19828945,5.74864534,-6.73635607,-6.30403354,4.31949674,-5.75124391,5.62215911,6.37520338,-5.3092763,3.71834992,5.20652386,-5.83688495,5.70589252,-4.15367808,-5.8127265 ])
        #self.pesosCapaOculta = np.array([[-0.82755118 , 0.58063252 , -0.95722717 , 0.41156061 , -1.01414162 , -0.00962569 , -0.86265106 , -0.39343142 , 0.35626582 , 0.24866342 , -0.20269003 , -0.1586937 , 0.88699744 , 0.06184265 , -1.61966886 , -2.35626876 , 0.40070654 , 1.42559335 , -1.48465041 , 0.3653684 , 1.62248508 , -1.65271526 , 3.30275202 , 0.59065149 , 1.11345969 , -1.63766089 , 0.53895026 , -1.23036962 , -1.44816084 , 3.32518476 , 1.22778767 , 0.49564142 , 0.79910494 , -0.31591065 , -0.17233996 , 0.63641138 , -0.71180835 , 1.28312204 , 0.42965358 , 1.20472648 , 0.04952242 , 1.29170874 , 2.01631696 , 0.66714979 , 1.08249921 , 2.45768355 , 0.61507577 , 0.9854264 , -1.85455643 , -3.3066681 , 0.66163235 , -0.30914254 , 0.83759063 , 0.6479144 , 0.91778127],[ 0.7827198 , -0.00877609 , 0.74056223 , 1.04086772 , 0.34579949 , 0.83814849 , 0.93128554 , 0.10532592 , 0.2684986 , 0.3707752 , 0.65503488 , 0.76353093 , 1.26824363 , 0.50898044 , 0.54509453 , -0.30414833 , 0.5817853 , 0.49949251 , 1.02682735 , 0.38905017 , 0.73311512 , 0.0046855 , 0.79873459 , 0.42603138 , 0.32318445 , 0.73048686 , 1.39810415 , 0.63531374 , 0.84146302 , 0.49674243 , 0.67582308 , 0.28540474 , 0.59640121 , 0.62315066 , 0.37439993 , 1.02643038 , 0.7921502 , 0.62727674 , 0.3207217 , 0.59055758 , 0.57483716 , 0.46558703 , -0.30502265 , 0.08643729 , 0.36625521 , 0.07751605 , 0.50644822 , 0.64746343 , 0.49617096 , 0.20505572 , 1.23224493 , 0.58798449 , 0.83159052 , 0.80435022 , 0.02285079],[-0.11221288 , -0.3245084 , 0.65695473 , 0.85997334 , 0.064703 , 0.30089382 , 1.2321828 , -0.76722366 , 0.95346215 , 0.13990539 , 0.65693962 , 1.1367724 , 0.34483213 , 1.35515079 , -0.36864283 , -0.046969 , 0.44314997 , 1.19328488 , 1.0023454 , -0.18260111 , 0.87189379 , 0.36764853 , -0.15457743 , 0.35730205 , 0.31079153 , -0.39807817 , 0.97576509 , 0.66624567 , -0.1638865 , 1.38271609 , 0.56420638 , 1.1162091 , 0.10588563 , 1.19957033 , -0.77393907 , 0.57298091 , 0.86170779 , 0.5594398 , 0.76878768 , 0.63322121 , 1.54781647 , 0.79756252 , 1.14699407 , 0.31932998 , 0.85866898 , 0.96087106 , 0.82147876 , 0.47303374 , 0.82330526 , 0.40731853 , 0.41868677 , 0.34854016 , 0.78653901 , 0.07794176 , 0.53470114],[ 0.73110253 , 0.38413989 , -0.5065441 , -0.27491097 , 0.24319813 , 0.62701715 , 1.43009941 , -0.79195287 , -0.2848337 , 0.07263011 , 0.13825573 , 0.20446747 , 0.75675395 , 1.34023785 , -0.2060629 , -0.52068519 , -0.18576145 , 0.99756908 , 0.07704409 , 1.26506286 , 1.68912531 , 0.13208586 , 1.42977682 , 0.01018244 , 1.05034629 , 0.88949103 , -1.29624767 , 2.76438181 , 0.63298185 , 1.15727176 , 0.06547858 , 2.42224048 , 0.57588996 , -0.03595845 , 0.43025158 , 0.92012725 , 2.14408279 , 0.55018236 , 1.07297477 , 0.87838958 , 1.52823896 , 0.33183321 , 0.77248562 , 0.83416326 , 0.18775042 , 1.00744424 , 0.88041039 , 0.89813591 , 0.51050258 , 1.89375829 , 1.1284551 , 1.83100545 , 0.25034283 , 0.39309254 , 0.66126581],[ 0.19401747 , 0.9594682 , 1.45129143 , 0.77843466 , 0.31505041 , 0.1466312 , 0.59750282 , -0.00510151 , 1.01687376 , 0.94914598 , 0.73740328 , 0.39233207 , -0.20700769 , -0.31217465 , -0.65708114 , 1.10242999 , 0.86284516 , 0.42828583 , 0.19238057 , 0.32573547 , 0.48598737 , -0.54909005 , -0.53924629 , 0.9001979 , 0.59317095 , -0.13871104 , -0.21677157 , 0.34801828 , 1.65968014 , 1.09867465 , 0.18630539 , -0.19322074 , 0.91338721 , 0.96742443 , 0.04994539 , 0.58766901 , 0.05943595 , 0.76962638 , -0.32053517 , 1.53968096 , -0.1249417 , 1.50049889 , 1.06990517 , -0.44788418 , 0.10701463 , 0.49871063 , 0.15789832 , -0.56557809 , -1.20752432 , -0.01745505 , 0.21549643 , -0.04672726 , 0.28523785 , 0.92866967 , 0.6166202 , ],[ 1.40140609 , -0.29241671 , 1.75144873 , 0.32980392 , -0.4514579 , 2.27197379 , 2.07925574 , -0.48811289 , -0.09851123 , 0.87539952 , -0.07486713 , -0.31074828 , -0.79776585 , 1.91622992 , 0.83786865 , 0.10770503 , 0.48375355 , -0.34888637 , 0.62134913 , -0.96052258 , 0.40709837 , 1.76354192 , 0.26012724 , 0.86950914 , -0.40778049 , 0.5308779 , 0.56768197 , 0.11946238 , 1.07143473 , 1.30238096 , 0.04128279 , 0.31536198 , 0.76447105 , -0.7608623 , -1.41900419 , 0.11659685 , 1.40418304 , -0.03482625 , 1.21895081 , -0.93148043 , 0.72159052 , -1.02043989 , 2.19714415 , 1.57431783 , 0.33686259 , -0.15556868 , -0.95594525 , 0.35967186 , 1.6672954 , 1.57025109 , -0.14541744 , 0.53262 , 0.93498274 , 0.26062097 , -1.03216791],[ 0.66133617 , 0.1426506 , 0.61971429 , 0.74804488 , 0.33056727 , 0.88539202 , 1.48295694 , 0.90427438 , 0.6543505 , 0.87128634 , 0.01242153 , 0.91774886 , 0.61356509 , 0.68304151 , 0.64640937 , 0.04470534 , 0.47994074 , 0.88702717 , 0.2335433 , 0.67262706 , 0.33836825 , 0.44649859 , 0.24111805 , 0.23267076 , -0.0071746 , 0.26663966 , 0.81729968 , -0.19474419 , 0.29490457 , 0.45700911 , 0.28465951 , 0.32844395 , 0.41681636 , 0.2994771 , 0.26008181 , 0.50510331 , 1.2615398 , 0.10970531 , 0.74992626 , 0.41761673 , -0.13232727 , 0.93949448 , 0.36930698 , -0.20344057 , 0.16348445 , 0.14062929 , 0.3993861 , 0.58402188 , 0.43380294 , 0.62422931 , 0.95110087 , 0.54984171 , 0.95907233 , 0.03409972 , 0.42119484],[ 0.57588475 , 0.24114135 , 0.38869756 , 0.3701421 , -0.67392256 , 1.10049914 , 0.6685297 , 1.02378831 , -0.1192846 , 0.00547043 , 0.6536436 , -0.07483805 , 0.63149511 , 0.23441976 , 1.07541728 , 1.70515188 , 0.66812162 , 1.2959003 , -0.51692157 , 0.03322477 , 0.42860791 , 0.14814371 , 0.90610789 , 0.18440235 , 0.71861819 , 0.20329944 , 0.1007563 , 0.58021309 , 0.04070388 , -0.03173042 , -0.31508433 , 1.43340596 , 1.1285532 , 0.23848625 , -0.50894779 , 0.31024622 , 1.17399513 , 0.47869022 , 1.1787223 , 1.12625454 , -1.26055281 , 0.04028691 , 1.07039474 , 1.70425753 , 0.67069336 , 0.09374035 , 0.57803147 , 0.67411874 , 0.78789429 , 1.24053582 , 0.28440391 , 0.76511984 , 0.30746833 , 0.1012214 , 1.14488553],[ 0.25094202 , 0.69963059 , 1.88545357 , 0.20983861 , 0.23030625 , 2.54844122 , 0.46957547 , -0.56878672 , -0.64621673 , 1.18858996 , 0.87428484 , -0.11402136 , 0.46126652 , 0.57158038 , 0.74347034 , 0.63117098 , 1.53099238 , 0.80079729 , -0.53575775 , -1.17950065 , -0.0131442 , 0.44462326 , 0.58079843 , 1.22274403 , 0.77729 , 0.58605704 , -0.0355512 , 0.34243101 , -0.06061468 , -0.6504466 , 0.4171048 , 0.22072346 , 1.7766135 , 0.79956815 , 0.55078339 , -0.17865842 , 2.38943679 , 0.30770729 , 1.27626929 , -0.10452894 , 1.78922135 , 0.06411623 , 0.1211205 , 1.56586891 , 0.81840718 , 0.72251603 , 0.41495413 , 0.16015855 , 0.41712484 , -0.16610174 , 0.27162481 , 0.62372031 , 0.88764997 , 0.73999123 , 0.36304719],[ 0.4523478 , 0.24895352 , 0.18098759 , 0.09469001 , 0.53413906 , 0.17097772 , 0.17309411 , 0.84875006 , -0.14408881 , 0.47699303 , 0.68754088 , 0.53621914 , 1.01843634 , 0.3652951 , 0.47063222 , 0.5070798 , 0.89809976 , -0.0520788 , 0.74433064 , 0.47239989 , -0.19597823 , 0.95586979 , 0.24522261 , 0.02014939 , 0.08364345 , 0.4708077 , 1.30208646 , -0.31991883 , -0.23818102 , -0.22198449 , 0.88796579 , 0.55347815 , 0.35453325 , -0.38576293 , 1.22489113 , 0.96922911 , 0.32590466 , 0.66934769 , 0.32231275 , 0.31747896 , 0.48595078 , 1.18810622 , 0.07854607 , -0.05334716 , 0.70687444 , 0.66921676 , 0.54456804 , 0.92580654 , 1.26704956 , 0.23887029 , 0.22639627 , 0.48122062 , 0.3504399 , 0.45700098 , 0.22452934],[ 0.97501912 , 0.46542286 , 1.70063507 , -0.17256016 , 3.13238227 , 1.1902391 , -0.25089474 , -0.66264518 , -0.41019943 , 0.381637 , -0.2978674 , 1.18972901 , 1.33291888 , 2.87993589 , 3.75534871 , -0.75410805 , 0.72732962 , -0.24350196 , 0.26363763 , 0.96630901 , -0.26499426 , -2.2798314 , 1.92185916 , 0.83579156 , 0.96493977 , -0.08397152 , -0.45598249 , 0.7097075 , 0.51578126 , 0.59264685 , 0.62868617 , 0.02460599 , 0.56008425 , 0.05297714 , 0.79528672 , -0.19707349 , -0.0601899 , 0.79800185 , 0.21860888 , 0.328861 , -1.26940476 , -0.19963017 , 2.04426562 , 0.58202567 , 2.52756706 , 1.34459444 , 0.78899475 , 0.65906266 , 1.8038155 , 1.0649086 , 0.21164857 , 1.10205204 , 0.95318652 , 0.51480487 , 2.10499565],[-0.6512916 , 0.48727321 , 0.66568515 , -0.39846744 , 1.23711799 , -0.47989444 , -1.20137281 , -0.78646397 , -1.29390154 , 1.4690798 , -0.57456306 , 0.85894971 , 0.85628419 , -0.3876416 , -1.30426344 , -0.97517112 , 1.19428248 , -1.36149277 , 2.24110064 , -1.36407324 , 1.70318834 , -0.84126534 , 0.73704157 , 2.28772299 , -1.7010573 , 0.55609847 , 0.96066532 , -0.42827472 , -0.17828238 , 1.71415312 , 0.93923869 , 0.41820497 , 1.95516274 , 0.9340248 , -0.16679317 , 0.37512903 , 0.55838158 , 0.98599235 , -0.10440344 , 2.36329775 , 3.10856723 , 0.2459681 , 1.20587859 , -0.56800032 , 0.28749795 , 0.23203848 , 0.48550812 , -0.78586466 , -0.65994864 , 0.60946129 , 1.00349541 , 0.76506661 , 0.24183306 , 0.92874722 , 0.5858827 , ],[ 0.30525918 , 0.86495645 , 1.46544625 , 0.13358069 , -0.65275597 , 0.47281992 , 0.79373865 , 0.85806723 , 0.21174223 , 1.23455176 , 0.60034358 , 0.46619034 , 0.70590247 , 0.37873355 , 0.1618612 , -0.40396304 , 0.71595789 , 1.5271691 , -0.28254494 , 0.70778338 , 0.39804789 , 0.48118834 , 0.90742075 , 0.90901006 , 0.93575258 , 0.6161376 , 0.02466965 , 0.99603787 , -0.57831142 , 0.47292533 , 0.58679822 , 0.77706893 , 0.06351633 , -0.14144225 , 0.44986797 , -0.12246774 , 1.22413231 , 0.66173306 , 1.41100712 , 0.2672235 , 0.82026584 , 0.06424201 , 0.65932373 , 0.79894515 , 0.08406484 , 0.27295251 , 0.13133626 , 0.73390188 , 0.94780391 , 0.81194895 , 1.16943998 , 1.0179196 , 0.24810393 , 0.77199378 , 0.75586286],[ 0.81109472 , -0.19737384 , 0.68115439 , 1.4501705 , 1.66034059 , -0.11899047 , -0.53604313 , 1.62165431 , -0.8425522 , 0.36819874 , 1.13233754 , 0.91030789 , 0.46629555 , 1.11812966 , 0.24090297 , 0.98099103 , 0.80587738 , 0.63507813 , 0.40052009 , 1.62062719 , 0.05720254 , -0.36003359 , 1.37250395 , 0.31725017 , 0.13608861 , 0.4747042 , -0.44349436 , -0.1990471 , -1.68327973 , 0.81528722 , 0.31132709 , -0.2187486 , -0.40904847 , 0.37027597 , -0.0399396 , -0.04230213 , -0.61984038 , 1.27820193 , -0.38107667 , 0.46474947 , 0.5710376 , 1.04029581 , -0.12224055 , 1.45386132 , 1.35417455 , 0.76067084 , 0.94981838 , -0.67267179 , -0.62556721 , 0.45077108 , -0.04447324 , 0.0724043 , 0.5291394 , 0.51601782 , 0.43650279],[ 0.31837119 , 0.40564536 , 0.06872601 , 0.57685932 , 0.07125287 , 0.12560888 , 0.34753627 , 0.4259909 , 0.5991862 , 0.25783888 , 0.82273698 , 0.98531089 , 0.64607457 , -0.01699068 , 0.77346251 , 0.64688654 , 0.43177376 , 0.56199866 , 0.42534685 , 0.36020316 , -0.06676877 , 1.00327092 , -0.04310535 , 0.91283086 , 0.07124799 , 0.67826936 , 0.55560928 , 0.92571232 , 0.8311577 , 0.92947701 , 0.22469725 , 0.33906402 , 0.64317844 , 0.87399125 , 0.46990499 , 0.0713795 , 0.43663689 , 0.82737424 , 0.53019077 , 0.03290358 , 0.58509478 , 0.74917068 , 0.34921241 , 0.90256926 , 0.26053034 , 0.26662329 , 0.42419208 , 0.27834842 , 0.11590385 , 0.13516216 , 0.25427828 , 0.80381638 , 0.07062871 , 0.69483282 , 0.96556265],[-0.70515647 , 1.9630074 , 0.45951957 , -0.13205211 , 0.9304299 , -0.80720011 , 0.89777195 , 1.16409683 , -0.56592599 , 0.27991605 , -0.30082294 , 0.41884218 , 0.06920264 , 1.38383601 , 0.39920285 , 0.22317467 , 1.3034319 , 0.52176854 , 0.16683247 , 0.72452576 , -1.15797473 , 1.51915079 , -0.25398433 , 0.76225091 , -0.80055004 , 0.37290391 , 0.79435728 , 1.20034642 , 2.4486645 , 0.31721944 , 1.33452973 , -0.25802979 , 1.25934263 , -0.61501173 , 2.36518081 , 1.71059935 , -1.84970078 , 0.99883833 , 0.91595931 , -0.42038655 , -1.31493264 , -0.02219995 , 1.81652661 , 1.52123938 , 1.04120388 , 1.48610837 , 0.83795842 , -0.24938887 , 0.11855199 , -0.23479435 , -0.82958174 , -0.17071629 , 0.99006188 , 0.26283213 , -0.67832212],[ 1.26430203 , 1.2959088 , 2.20142978 , 0.54067156 , 0.97885311 , -1.11733261 , 1.98269237 , 0.96623664 , -0.07393896 , 1.54586328 , -0.03743108 , -0.01256494 , 0.94877108 , 0.467925 , 0.14374596 , 1.23960049 , 0.68358933 , 1.19056118 , 0.68672345 , 0.93819781 , -0.89705275 , 0.82849373 , 1.19840632 , 0.32849513 , -0.0270548 , 1.00684701 , 1.07421954 , 0.52240774 , -0.53653263 , 0.80125824 , 0.92677959 , 0.58154586 , 0.06755823 , 1.64704411 , 1.40055986 , 1.13592749 , 0.46459629 , 1.00890373 , 0.46524702 , -0.09469513 , 0.75662145 , 0.94443194 , 0.95801079 , 0.89550713 , -1.58896672 , 0.66145775 , 0.35078742 , 0.27615567 , -0.1746909 , -1.11635921 , -0.05438556 , 0.11628134 , 0.92023395 , 0.75865623 , 0.71294598],[ 0.03457809 , 0.90422639 , 0.30398749 , 0.08007755 , 0.158682 , 0.40208115 , 0.9194187 , 1.08152005 , 0.38081399 , 0.97034702 , 0.15058378 , 0.7666787 , 0.86172123 , 0.35233443 , 0.39927579 , 1.10169367 , 0.98269664 , 0.42853894 , 0.16491043 , 0.27041021 , 0.34594138 , 0.64118713 , 0.98908752 , 0.6535646 , 0.54743009 , 0.22446452 , 0.17482475 , 0.45773975 , 1.00186216 , -0.025078 , 0.26001769 , 0.40590908 , 0.73027058 , 0.49605746 , 0.64943263 , 0.06080129 , 0.75657326 , 0.92951907 , 1.09515276 , 0.5577973 , 0.59639093 , 0.04801001 , 0.19150964 , 0.22449915 , 0.47659957 , 0.35266703 , 0.30526818 , 0.41469576 , 0.81948514 , 0.59281883 , 1.00609576 , 0.23326442 , 0.99919907 , 0.21160059 , 0.19524724],[ 1.01133703 , -0.55953671 , 0.9661084 , 1.05437276 , 0.5797454 , -0.98220163 , 1.28772076 , 2.19885157 , 1.06448803 , 0.95815256 , 0.16357481 , 0.49179237 , -0.18962708 , 0.21181103 , 0.67574738 , -0.56090978 , 0.64566146 , -0.14141351 , -0.09101642 , 1.53714633 , 2.44952792 , 0.018523 , -0.2151606 , 1.22147419 , -0.33619155 , -0.44383992 , 0.80493001 , 1.43827578 , 1.05213961 , 2.78592427 , 0.46831952 , 0.39345742 , 0.70743963 , 1.6496134 , 0.02121663 , 0.01064666 , 1.92577018 , 0.08002491 , -0.21086784 , 0.81505493 , -0.85795806 , 0.41958879 , 0.7396617 , 0.73089034 , 0.92916157 , 0.3874492 , 0.26802636 , -0.32459948 , -0.45433171 , 0.8722856 , 0.90413353 , 0.58177432 , 0.46269262 , 0.08049254 , 0.52202011],[ 0.04123027 , 0.77118673 , -0.01128561 , 0.53489573 , 0.75111288 , 0.25688574 , -0.37434539 , 1.09975006 , 1.44326296 , -0.10195382 , 1.32015944 , 0.80088413 , -0.21036471 , 0.72587301 , 0.61194069 , 0.73470859 , -0.15278913 , 0.89176795 , 0.90877245 , -0.95066272 , -0.77817504 , 1.4682617 , 0.81917228 , 0.66494895 , 0.09412401 , 0.67609861 , -0.39120243 , 1.20760776 , 0.65990917 , -0.16091712 , 1.10855605 , 0.55390511 , 1.30110707 , 0.24634112 , 0.89955759 , -0.41475403 , 0.00954882 , 1.04754023 , 0.47889862 , 0.73070855 , 0.77765374 , 1.20233382 , 1.86473027 , 0.91905902 , 1.03307128 , 0.85050004 , 0.43584864 , 1.11527873 , 1.06619024 , 1.10109185 , 0.90150593 , 0.20172179 , 0.42230145 , 0.99628983 , 2.14719438],[ 4.41144748e-01 , 1.99091928e-01 , -4.98146892e-01 , 2.94410422e-01 , 5.91322875e-01 , 2.26091643e+00 , -1.37053497e+00 , 6.25856698e-01 , 8.31734433e-01 , -5.44635891e-01 , 2.88426880e-01 , 1.21734995e+00 , 1.42126309e+00 , 4.47930455e-01 , 1.55631422e+00 , 2.06226992e+00 , 2.12631139e-01 , 6.82022824e-02 , 1.37068602e+00 , 1.27364844e+00 , 8.80339984e-01 , 1.89414433e-02 , 1.48588346e+00 , 1.10845683e+00 , 2.83625561e-01 , 8.83026471e-01 , 8.70523346e-01 , 7.10751660e-01 , 4.59489833e-01 , 6.13240808e-01 , 4.59851886e-01 , 1.41929795e+00 , 6.63260354e-01 , 3.03485861e+00 , 1.69600333e+00 , 2.75574498e-03 , 4.88926467e-01 , -5.75364934e-01 , 1.92561498e+00 , 2.35754923e-01 , 1.44842721e+00 , 1.02129679e+00 , 2.11227414e+00 , 1.52389262e+00 , 1.60776820e-01 , 1.24594851e+00 , 2.03276732e+00 , 8.15911841e-01 , 1.55944788e+00 , 3.42155907e-02 , 2.82129775e-01 , 6.79208612e-01 , 6.91878951e-01 , 3.55815266e-01 , 1.92474191e+00],[ 0.48350464 , -0.0427939 , 1.02236631 , 0.42237196 , 0.14724642 , 0.77623713 , 0.6891925 , 0.65914789 , -0.04517386 , 0.96836686 , 0.79364917 , 0.88563945 , 0.66381797 , 0.74968915 , 0.43214528 , 0.60829969 , 0.61251589 , 0.1905166 , 0.17938498 , 0.85456984 , 0.7001395 , 0.00454351 , 0.14497393 , 0.51242891 , 0.38933723 , 0.74059511 , 0.58023377 , 0.52094447 , 0.11041462 , 0.97825866 , 0.44111818 , 0.46130328 , 0.81012777 , 0.55928199 , 0.69565689 , 0.37734573 , 0.56697305 , 0.6683366 , 0.77243913 , 0.19476664 , 0.33944382 , 0.94904795 , 0.00107932 , 0.97285836 , 0.15651773 , 0.67472725 , 0.08321204 , 0.60917792 , 0.12160091 , 0.20193144 , 0.70715055 , 0.0010925 , 0.71536681 , 0.07157088 , 0.33855856],[-0.03457238 , 0.3862812 , 0.17092008 , 0.37098553 , 0.51655756 , 0.10101346 , 0.63569408 , 0.88737389 , 0.35199248 , 0.5441101 , 0.28784988 , 0.55852373 , 0.53134788 , 1.08205802 , 0.50191451 , 0.67520383 , 0.55300916 , 0.78784142 , 0.85708649 , 0.66508569 , 0.75299418 , 1.12377951 , 0.53624798 , 1.19289014 , 0.79472138 , -0.12574322 , 0.24329669 , 0.90638655 , 0.21051791 , 0.94754733 , 0.49829416 , 0.70870573 , 0.68283442 , 0.21746138 , 0.81961732 , 0.46501812 , 0.3782899 , 0.41011272 , 0.37222026 , 0.79449208 , 0.26080966 , 0.73042711 , 0.27301925 , 0.41760094 , 0.7046711 , 0.27967228 , 0.50747362 , 0.00708808 , 0.22401946 , 0.56064337 , 0.66735161 , 0.66430074 , 0.16455121 , 0.49894625 , 0.75866811],[ 0.53311612 , -1.17540436 , 1.22935486 , 0.73757002 , 0.80065998 , 0.66589918 , -1.04763161 , 0.79550881 , 0.26638511 , 0.69809627 , 0.99037572 , 1.01161323 , 0.06050098 , 0.52445453 , -0.005361 , 2.55542338 , 1.02193487 , 0.0165484 , 0.7906114 , 0.28953993 , 2.05405067 , 0.2345014 , 0.71139228 , 0.55764033 , -0.36034985 , -0.17425468 , 1.09624376 , -1.006629 , 1.27836611 , 1.3875414 , 1.09232591 , -0.59625942 , 1.07716821 , 0.30547714 , 0.40203177 , -0.86024105 , 0.91098836 , 0.44261234 , -0.43517432 , 1.66365036 , 0.5242461 , 0.5009368 , 0.11574698 , 1.08993787 , 0.5578587 , 0.90402287 , 0.61627907 , 0.50674776 , 0.83205753 , -0.22108028 , 1.42178151 , 2.09182255 , 0.11158815 , 0.87124252 , 0.51029589],[ 0.81468727 , 0.75510312 , 0.4629179 , 0.44629771 , 0.82574541 , 0.81469989 , 1.08860729 , 0.70796921 , 1.11313434 , 1.12187928 , 0.64013458 , 0.89749737 , 0.82329514 , 0.36937738 , 0.74297042 , 0.35773171 , 0.6494804 , 0.48009076 , 0.81755829 , 0.67564268 , -0.22231131 , -0.04429721 , 0.29598876 , 0.71267611 , 0.33027523 , 0.87771909 , 0.94426011 , -0.29093243 , 0.44915319 , 0.10111267 , 0.34572438 , 0.21644865 , 0.89886116 , -0.15390017 , 0.64986957 , 1.57613549 , 0.3542913 , 0.48556184 , 0.82232293 , 0.16066817 , 0.91986257 , 0.58390573 , 0.76956498 , 0.43254451 , 1.17430362 , 0.97623458 , 0.17189641 , 0.64473108 , 0.05282524 , 0.45309297 , 0.9440346 , 1.30976733 , 0.83281386 , 0.84277341 , 0.13939364],[ 0.04859096 , -0.99642234 , 1.06954945 , 1.03815709 , 0.55036376 , 0.31872929 , 2.02899487 , -0.93736574 , 1.4090449 , 0.23455883 , 1.1435884 , 0.70559663 , 1.05899009 , -2.31553366 , 1.05224516 , 1.03820885 , 0.95879596 , 0.6575691 , -0.48222125 , 0.43554099 , 0.60007619 , 1.55908954 , -0.06710944 , -0.09844638 , 0.36321587 , 0.58606675 , -0.43596977 , 2.16721435 , 0.55260034 , -0.50797866 , 0.48732104 , 0.01366557 , 0.42310221 , -2.08643519 , 2.24388984 , 0.95981639 , 1.66619875 , 0.76687037 , -0.68481524 , 0.93466015 , 2.40428212 , 1.44372015 , 2.45520392 , 2.1855261 , 2.33915053 , 0.39441291 , 0.09325284 , 0.33777612 , 2.6406692 , 0.38490298 , -0.02951488 , 0.14332798 , 0.69502081 , 0.13559916 , 1.1233848 , ],[ 0.78714654 , 0.12747979 , 0.84004243 , -0.4394942 , 0.50462207 , 1.18231905 , 0.0899958 , 1.26256001 , 0.9033079 , 0.93411921 , 0.1241819 , 0.24982588 , 0.81267144 , -0.09296862 , 0.48068095 , 0.50135243 , 0.36463616 , 0.08607895 , 0.59252157 , 0.46102331 , -0.35481396 , 1.41442067 , 0.73754124 , 0.60697551 , 0.00878064 , 0.76880983 , 1.16087381 , 0.52563685 , 0.2129704 , 0.8861036 , 0.75602579 , 0.662115 , 0.47210484 , 0.6279814 , 0.48675967 , 0.35199676 , 0.75215762 , 0.71226882 , 0.74371983 , 0.6569225 , 0.96444144 , 0.08920318 , 0.18261603 , -0.11976781 , 0.28394048 , 0.6835294 , 0.25034344 , 0.29765875 , 0.24346226 , 1.11446337 , 0.91375751 , 0.05218937 , 0.55976167 , 0.69299775 , 1.19210855],[ 0.70347946 , 0.48761202 , 0.75001563 , 0.15101002 , 0.65612937 , 1.14593402 , 0.38084542 , 0.8253486 , -0.28965941 , 0.63599524 , 0.31216237 , 0.64406072 , 1.00145754 , 0.1171768 , 0.65859195 , 1.72950222 , 0.96130404 , 0.04231273 , -0.09748662 , 0.60230865 , 1.06349757 , 1.10747618 , 0.49980805 , 0.74515788 , 0.84934106 , 0.47071083 , 1.0575708 , 0.80204164 , 0.50098705 , 0.43779818 , 0.86002641 , 0.71528029 , 0.56836945 , 0.20766393 , 0.45052125 , -0.15953342 , 0.84990944 , 0.68040473 , 0.90249515 , 0.79330751 , 0.50990905 , -0.44152413 , -0.23110425 , 0.72901777 , 0.30726586 , 0.12422939 , 0.79597463 , 0.18809655 , 0.40605671 , 0.20180948 , 0.07999964 , 0.57734046 , 0.9213225 , 0.04976526 , 0.52971697],[ 0.52266333 , -0.32013028 , 0.81174487 , -0.17153716 , 0.40382298 , 0.52385323 , -0.86414698 , 0.24179686 , 1.15083481 , 0.4878625 , 0.52303458 , -1.2253799 , 0.76078159 , -1.23051235 , -1.38428169 , 2.0240341 , 0.8206591 , 0.20597244 , 0.03547642 , -0.11791576 , -0.66644924 , 0.27018581 , -0.93842793 , 0.79733245 , 0.8317572 , -0.23236096 , -0.81612073 , 1.87490038 , 1.39099698 , -0.528126 , 0.81311413 , -0.19995136 , 0.08399321 , 1.70673293 , -0.42559512 , -0.56138418 , -2.80234374 , 0.62975339 , 0.14320366 , 0.57188237 , -0.27450577 , 0.12281734 , 0.37041275 , 0.56925329 , 0.31752868 , -0.23893171 , -0.39334613 , -0.91721044 , -0.38408638 , 0.22675141 , 0.44179825 , 0.8405608 , 0.42037679 , 0.98048676 , 0.87514712],[ 0.23684661 , 0.05128357 , 0.19754691 , 0.34726798 , 0.04745209 , -0.39382183 , 0.7158884 , 1.3722366 , 1.60269801 , 0.93320286 , 0.30815682 , 0.54027865 , -0.35510517 , 0.37598015 , 2.0948457 , 0.54430841 , 0.75798168 , 0.68528014 , -0.2136936 , 0.67572606 , 0.80864619 , 1.64670932 , 1.39537939 , 0.66269721 , 0.19457035 , 0.22768877 , 0.79044863 , 1.43502272 , -0.50291685 , 0.20761335 , 0.62224774 , 0.47658202 , 2.60500057 , 0.58731895 , 0.28318538 , 1.53045485 , -0.60230423 , 0.37904497 , 0.99518897 , 1.15784383 , 0.65278066 , -0.76246544 , -1.5345917 , -0.85156348 , -0.1970513 , 0.44315929 , 0.11162637 , 1.50602002 , 0.63341791 , 0.15500839 , -0.05824007 , 0.59502538 , 0.62755933 , 0.13206359 , 0.07465146],[ 0.39435711 , 0.8830566 , 2.11806257 , 1.93234585 , 1.09303088 , 0.4903627 , -1.22493421 , 0.84434206 , -0.4142828 , 1.39223833 , 1.05447361 , 0.3066641 , 1.38177763 , 1.2462367 , -0.66385149 , -1.12488478 , 1.04498305 , 1.43400759 , -0.24276395 , 0.69313972 , 1.19826856 , 0.20884197 , 1.79475456 , 0.84963198 , 1.32471592 , 0.52438436 , 0.75874029 , 0.55447012 , -0.11021829 , -0.32216085 , 0.95301017 , 0.09961632 , 0.93035608 , 0.17931792 , 0.92454784 , 0.09918287 , 0.79615533 , 0.97800138 , 1.27226022 , 0.72438582 , 0.41625982 , 0.5732341 , 1.73884382 , -0.12119124 , -0.25678993 , -0.12891128 , 0.57398607 , -0.70741598 , 0.0209655 , 0.53171282 , 1.361121 , 0.57438114 , 0.54902989 , 0.83496804 , 1.82149328],[-0.44510573 , 0.39992675 , 0.64973164 , 0.09966006 , 0.46870571 , 0.13302587 , 0.49684181 , 0.68134147 , 0.42985319 , 0.32578576 , 0.24329759 , -0.03682512 , 0.98420547 , 1.49043348 , 0.62533 , 1.41138193 , 0.69660373 , 0.2962722 , 0.45800289 , 0.87051244 , 0.71168834 , 1.61739938 , 0.7183323 , 1.35924418 , 0.80231619 , 0.25914316 , 1.16694984 , 0.5270044 , 0.22232895 , 0.77780915 , 1.23746552 , 0.69373007 , 0.13474629 , 0.22891488 , 1.60919092 , 1.23171029 , 0.98457765 , 0.42670749 , 0.12917257 , 1.96409704 , 1.61876983 , 0.1402602 , 0.94371815 , 1.13089461 , 0.39065081 , 0.93542786 , 0.61732638 , -0.32525249 , -0.52415339 , 1.0866129 , 0.7620296 , 0.17943304 , 0.76746907 , 0.65291251 , 0.64403248],[-0.28575786 , 0.20830715 , 0.19757975 , 0.23665884 , 0.59626862 , 0.61548506 , 1.55902057 , 0.25064344 , 0.29719887 , 0.51710628 , 0.55226845 , 0.13352842 , 0.58352182 , 0.81603018 , 0.57426789 , 1.19155374 , 0.72814131 , 0.42614222 , 0.85414942 , 0.43681241 , 0.98056921 , 0.52532369 , -0.10953245 , 1.22865474 , 0.43883856 , 0.49997382 , 0.81679759 , 0.56603511 , -0.73657177 , -0.22499298 , 0.58008222 , 0.47857451 , 1.05991186 , 0.88266309 , -0.56837975 , -0.3056557 , 0.25060584 , 0.3806898 , 0.9010888 , 0.438756 , -0.14652954 , 0.50081743 , -0.04958761 , 0.99326634 , 0.71121919 , 0.1385732 , 0.29872809 , 0.23213859 , 0.9823065 , 0.74090005 , 0.7151647 , 0.70963788 , 0.07445385 , 0.49525869 , 1.0170342 , ],[ 0.07950351 , -0.31125067 , 0.04086641 , 0.66287346 , -0.11845181 , 0.08688313 , 0.86718629 , 0.95717406 , 1.44351628 , 0.89090277 , 0.60389171 , -0.03848673 , 0.28429284 , 1.35887305 , 0.37766252 , 0.41761438 , 0.63513326 , 0.36397212 , 0.97046692 , 0.76409361 , 0.62805642 , 0.95701238 , 1.00699854 , 1.02928845 , 0.38583054 , 1.06847674 , -0.00234375 , -0.7205838 , 0.90301316 , -0.38963606 , 1.18062491 , 0.32909937 , -0.35300451 , 0.3467457 , 1.09916832 , -0.20214331 , 1.30579431 , 0.77016418 , 0.69325959 , 0.6505235 , 1.42593712 , 1.1599456 , 0.83189943 , 0.18727114 , 0.47474673 , 0.32011655 , 0.83144114 , 0.23924128 , 0.5654212 , 0.83839213 , 0.26593661 , 0.60317094 , 0.12798621 , 0.01623739 , 0.57385969],[ 0.02134688 , -0.29623003 , 0.92452815 , 0.26948794 , -0.11517197 , 0.30403318 , 0.77891042 , 0.54927914 , 0.00443919 , 0.40267923 , 0.3670043 , 0.41466884 , 0.94155425 , 1.45400888 , 0.71627501 , 0.71434692 , 0.71999789 , 0.86111529 , 0.36348392 , 0.43592204 , 0.47297034 , 0.47512335 , 0.94920129 , 0.55375006 , 0.84256026 , 0.12751765 , 0.65742841 , 0.52082379 , 1.13208801 , 0.02849249 , 0.46574692 , 0.80360135 , 0.37621968 , 0.41807338 , 1.09371984 , 0.37216526 , 0.78554021 , 0.7561898 , 0.29007273 , 0.47472754 , 0.73755155 , 0.53484139 , 0.43533256 , 0.36614668 , 0.8659837 , 0.02234628 , 0.18500367 , 0.73158217 , -0.2111947 , 0.42893599 , 0.45919099 , 0.76398134 , 0.54291791 , 0.07785483 , 0.23612162],[ 0.50577695 , -0.03445198 , 0.42631401 , 0.56196499 , 0.47427953 , 0.14810319 , 1.00112708 , 0.71662611 , 0.78241161 , 1.09916963 , 0.15285126 , 0.01652198 , 0.38350923 , 0.83626168 , 0.89042291 , 1.00631258 , 0.93077509 , 0.86505807 , 0.2282472 , -0.02825635 , 1.01436766 , 0.64155915 , 0.39311226 , 0.49165732 , 0.84956918 , 1.07918608 , 0.44879086 , 0.65613235 , 0.65044834 , 0.44014483 , 0.80722146 , 0.86627954 , 0.69559352 , 0.28551014 , -0.07035587 , 0.53750745 , 0.75565697 , 1.15616071 , 0.55038971 , 0.12365892 , 0.42676926 , 0.278696 , 0.61921429 , 0.02571733 , 0.15128202 , 0.31778512 , 0.07882539 , 0.20824997 , 0.69900945 , 0.49285176 , 0.09868682 , 0.30128112 , 0.18078542 , 0.82248512 , 0.93303226],[ 0.46591303 , 0.94891521 , 1.82944061 , 0.06833499 , 0.61814285 , 0.04084193 , 0.41798218 , 0.21025722 , -0.46498316 , 1.42292256 , -0.41179562 , 1.23965237 , -0.25718032 , 0.60977062 , 1.42490548 , 0.31836197 , 1.08971803 , 0.62271241 , 0.50285689 , -0.33902891 , 0.76928072 , 1.53611338 , -0.02669029 , 1.21048182 , 0.56943508 , 1.15468734 , -0.25451835 , 1.26332697 , -0.89805646 , 0.72385016 , 0.72718408 , 0.23790704 , 0.77669832 , 0.9145447 , 0.96646118 , 0.10998939 , 0.10492149 , 0.83895203 , 0.46685997 , 0.1431642 , 1.74902129 , -0.07345814 , -0.9393601 , 0.82319581 , 0.49327425 , -0.06462574 , -0.11302913 , 0.87492647 , 1.27360006 , 0.5444389 , 0.08260021 , 0.17267428 , 0.22995237 , 0.07044587 , 0.95330106],[ 0.07251638 , 0.90082814 , 0.27483219 , 1.10750767 , 0.63598921 , 0.76496313 , 0.96734089 , 0.17242372 , 0.21872578 , 0.85100272 , 0.16575062 , -0.01316084 , 1.18333519 , 1.0585721 , 0.36613083 , -0.31055907 , 0.57939563 , 0.38987785 , 0.62815382 , 0.94372539 , -0.03591475 , 1.15438833 , 0.81729424 , 0.91503878 , 1.11841395 , 0.35915234 , 0.78960118 , 0.26103517 , -0.2824724 , 0.69115371 , 0.45455568 , 0.19609284 , 0.83529038 , 0.93014521 , 1.01433599 , 1.46008832 , 0.70383142 , 0.80878877 , 1.14239815 , 0.11947548 , -0.25510352 , 0.70920371 , 0.02374445 , 0.52456054 , 0.45723298 , 0.16146911 , 0.40520334 , 0.90612652 , 0.23754933 , 0.45301284 , 0.86163802 , 0.39623561 , 0.05347136 , 0.92730578 , 0.35630602]])
        #self.pesosCapaSalida = np.array([-5.87570349 , 1.3574826 , -2.18508513 , -4.73435871 , 3.49621611 , 4.69603699 , 1.20587643 , -2.99884228 , 4.19798571 , 1.33825479 , 6.86991582 , -6.39900717 , 2.20034292 , 3.40846739 , -0.34008275 , -5.23432874 , 4.09283892 , -0.87831597 , 3.93284824 , -4.04770259 , -5.39638196 , -0.02114689 , -0.71029902 , 4.12006774 , 1.65608182 , 5.62509982 , -1.61542662 , -1.65846144 , 4.88992045 , -3.95918498 , 4.43704173 , -2.76393064 , -1.74488592 , -2.31393368 , -0.83837142 , -0.90610288 , 2.7945696 , 1.47893731])
        self.biasCapaOculta = 1
        self.biasCapaSalida = 1
        self.generarMatrizDeConfusion(1)
        self.calcularPrecision(1)
        print("error global: {}".format(self.error))
        print("matriz de confusion")
        print(self.matrizDeConfusion[0])
        print(self.matrizDeConfusion[1])
        print("precision")
        print(self.precision)
        return self.y_resultado,self.salidaDeseada
        
        
    def ProbarModeloAlgoritmosGeneticos(self):
        self.pesosCapaOculta = np.array([[-0.50918272,-2.2843677,2.79971185,3.15774816,3.34280869,1.59687457,-0.23817164,-3.09905431,-0.95777786,2.07574941,6.87284883,-6.53130398,0.61988439,-2.42108879,-0.22736728,0.61066881,-2.98928822,-4.02226833,1.67452814,-1.90258446,-0.10268473,1.23133978,0.21707525,1.10805426,1.34544721,-1.88863887,-1.45265033,-3.50105366,2.08741487,5.45232582,1.65493986,3.03468942,-0.5989309,-1.30692705,-0.30933594,-4.67163451,3.31505654,-1.42177789,-1.21451403,1.4382302,-4.44762511,0.83878984,7.10350949,2.32952921,-2.99912876,-2.18739418,-4.5886126,2.03724069,-1.62513734,-2.98596198,4.30385001,3.82790403,1.40849649,0.26463258,1.26186143],[4.35569432,0.4284183,1.26560786,-0.01741445,0.00902578,3.69133063,-1.45779112,-4.19848091,2.38660517,3.91865818,1.92339229,-2.39405249,0.34637378,5.91261262,0.40575905,-0.65112073,3.4537609,5.25543986,1.21243256,1.90751768,-0.08300334,3.75203613,-3.42938456,8.33401904,-5.51504824,-0.4229728,-5.74153127,-0.14203639,-0.50612724,-3.55703736,1.92581829,-4.04971025,3.94349927,0.3680134,-2.70859687,2.30084409,4.33782305,0.06779027,-1.11243971,-4.29280255,-2.2072666,1.65476403,-0.32457864,-1.04898505,-1.99112683,-3.14545418,3.19155446,-2.39028257,-0.15107881,2.96035087,2.94694954,-1.76624805,1.57164501,-2.2781078,-3.00609687],[-0.06615468,-3.00132366,8.69294875,9.07918248,4.59562941,0.63349121,-4.86734618,0.87668555,-4.05926243,4.07383669,0.93802452,-4.98059334,-1.67116477,5.25897201,-1.67039279,-1.16960701,2.41879218,4.4723254,-2.10513472,-0.04822578,3.64867289,-2.22251989,9.95162757,-3.97636668,0.81563724,-1.08935694,-0.41006922,-3.18896729,0.48976201,2.28042784,1.23271563,4.67118986,-1.86297698,-0.03026901,4.95716482,2.11440748,-3.46317974,1.98577679,-2.51775701,3.9042671,-2.8461619,-7.79133602,-5.37270667,-3.72009243,5.40648364,0.75486862,-0.75722568,2.07651219,-0.76212896,-0.44389774,-4.18301791,-0.18881633,-7.20229881,-2.30256533,0.04412553],[-5.10510214,-5.7347728,-4.55998126,-0.53833829,-3.13830442,-4.36361616,-2.7604505,-0.0313692,0.27680884,1.94446899,3.43875806,0.56992521,-1.84737831,0.64799491,-3.23075299,3.66104664,2.89343046,-0.0684818,0.61102498,0.1387002,-1.16406563,-3.66767316,-1.17079844,-3.69559726,-2.24213782,0.70907697,1.76635223,1.75066546,5.31080563,3.21473312,3.24289857,1.68963564,2.50117007,-0.57232795,-6.60790879,0.17282676,-1.80056154,1.22495857,-1.62674549,2.31350417,0.98391861,-5.62440853,1.00151629,-1.00429439,2.4351364,0.81869761,-0.76058166,3.32048709,-3.24491511,-4.68084349,-0.12846312,3.52501583,3.85001269,2.86062679,5.55973467],[-7.21463952,-1.2837846,-0.11970395,-0.52987129,-1.28632876,-7.91256283,1.26933942,-0.73469817,2.50409624,-1.95110534,-0.60974057,-0.58968001,-2.38070426,-4.74845029,2.7868778,2.16716353,2.21345478,0.21001493,-4.25644931,9.32045161,2.26370111,-0.30009796,-1.46186654,4.2089,0.55802212,4.27302915,0.82684663,-1.68310448,-1.75669209,-0.35389578,-1.47190028,0.03916831,3.09686663,0.04994121,-0.96704336,1.48338351,-2.60156451,-0.27890261,-0.91765333,1.8143739,-0.27886713,0.73392632,-3.37573117,1.52386595,-1.14052226,0.48162177,3.46324886,0.84801357,1.2873528,-3.09743285,0.14643613,0.12376872,-1.07079072,-0.79270491,-0.97174638],[4.30343461,-4.50489506,-7.14033296,2.39612943,2.26745273,1.47353816,-6.67031409,0.38257919,0.96382815,-4.2710341,0.96534239,2.37874825,-5.5159339,2.88130788,-0.34185,0.38872343,-2.78350656,-1.93598245,-0.77019447,1.56878309,0.0392898,-2.77087136,1.37585231,7.05865737,-4.11669647,-5.24089508,1.13250534,2.88612601,-1.00895369,-0.58291958,-3.65331645,-2.90829991,-2.50698139,-0.68952912,0.04191441,1.45765603,-1.83073906,-0.63814019,-3.94184074,-2.94912318,-2.06682987,0.53066359,-1.00860807,-3.58315453,-5.25995991,9.30582302,1.98280246,4.83692473,-0.26423574,2.59666272,-5.24329544,-1.67867352,-0.62754052,1.73892745,-4.50097498],[-1.65205022,2.69985749,-0.20277079,-3.87135649,-4.65081885,-9.17922108,3.84062637,-2.87692266,-0.48993031,3.09789301,3.93624801,3.25703888,0.26639552,2.57888143,5.1889453,0.90329265,-3.20257184,1.32905595,7.21518883,-1.26667322,0.2294633,1.05857165,5.80612577,0.42081733,-1.23960783,-1.83849923,-0.33174526,-0.10915136,-2.10583787,2.38997909,-0.02087265,3.07499146,-2.30169213,1.66375852,5.31785098,5.82056096,-0.35905773,0.31996973,6.07506469,-0.19224847,-1.29025034,-3.21996411,0.92756877,1.86358179,-4.76870526,6.74698306,4.17549955,0.12609616,2.37361523,-0.20426993,-0.18607404,1.13562165,-6.04929895,0.37198342,6.39490968],[6.27485934,-1.074453,-1.42713714,0.86818412,-4.1410524,2.38390004,3.45184189,1.89039496,1.17404968,-3.63433312,2.45504517,8.13219754,-0.09299558,0.32212224,-1.36939116,-2.3571303,0.28167471,-3.47588376,-2.85857595,3.85473133,-1.18452516,3.81109583,2.22798499,-0.52575171,-3.70181268,1.01150139,2.92613312,-2.55137155,-4.35140786,-1.14410634,-0.10669806,1.50073767,-2.49551904,-1.03901549,4.9609835,-3.60146472,3.00812651,1.30052384,-0.13792461,0.92740842,2.40814591,0.85172041,1.15190329,-6.34237297,-1.72614613,-1.13628493,2.79673972,-4.56144271,9.03103851,-0.48683422,-0.35414124,-1.69156952,-1.05879357,-0.10595804,2.66068896],[5.56448148,0.09691911,2.19534338,5.30131671,7.26759179,1.70255995,-0.96522333,2.43602841,-2.89383289,2.64714939,7.40354514,-5.30769829,-0.51725386,0.086989,1.44266617,2.10956942,-4.4790201,2.31654182,3.42556649,-1.63076228,-2.84469988,4.67007575,0.48942963,-0.96539858,-3.80748456,2.25771936,-3.40552968,-1.37195961,-0.27547317,1.5389912,-2.122523,3.51028311,-5.17368244,-0.85058229,-2.07079261,4.069802,-2.33623661,0.87262196,-4.69699344,2.50997816,3.27564231,2.48226906,-8.15519657,-3.39922223,2.20129127,0.85130269,-4.7988875,1.62813215,2.22575248,-4.44334523,-1.47531882,3.0854516,0.62537597,-0.73214421,-2.86281903],[-1.80403725,-6.30577449,1.75695959,1.27654668,0.13336643,1.29082435,2.02939073,-1.54422171,-0.2636105,0.26467725,3.74537854,-2.09483541,-5.67845645,-2.21007565,-0.33311215,3.86461342,-4.45961304,2.81782029,-6.52540995,4.630613,-3.08666792,1.32490548,-1.11693627,2.84890889,-3.18995885,5.50359138,-1.32595837,0.35755957,0.58589066,-1.90241781,0.14219475,0.18800992,2.24623415,0.69839596,-0.9967559,-0.42824171,-1.86645426,1.24410026,2.06823441,-0.85323078,-0.6678248,-0.21308643,2.95999866,2.2427774,0.4610966,1.07636989,-6.92679302,3.09513526,2.01924714,2.39895852,-1.97962399,8.16784556,5.34431938,-6.50478269,0.16558305],[-0.3394184,-2.66972884,1.30787242,1.80412604,5.73641526,-0.32130506,-2.60893273,1.32261546,-1.29381226,-0.66917499,1.27783391,2.27038469,3.45430272,2.89155021,2.60738087,-2.9917802,4.1504329,3.27570403,0.41830166,-0.70713959,0.5618994,-3.80984541,0.36086055,-3.0940723,-0.22477612,-7.95389563,-1.60996773,-1.95131081,3.08249925,1.67749019,4.37955443,-1.64728643,-0.34135012,3.87866117,3.6747738,2.05954731,-1.23713973,-1.21053097,0.92621727,-4.35582162,-3.83119664,-2.09015229,-1.03865782,3.48300681,-1.85169013,-0.58745954,-4.61781894,1.76459401,1.13917308,1.42496149,0.41571007,-2.40922841,-1.0043287,-1.28232812,0.20853796],[0.64636162,-2.05876974,1.26756031,-2.51933188,-6.38137619,2.54789771,0.70822941,-1.11567194,0.57526122,-0.3507091,3.62501275,1.01704046,1.70887286,1.85282513,3.9736072,2.93010046,-3.28397207,-2.06639244,-6.54874813,1.4386673,-1.23078377,-3.36935657,-2.01690037,1.50196208,-2.85657298,-6.65100018,0.54877184,-0.75011737,4.66570765,-3.37781367,-1.78309321,1.1689792,-5.15328236,1.69686803,4.25131792,1.92016429,0.09220477,-3.17479077,-4.78882767,-2.60632104,1.92855456,0.59078904,-1.55174691,-2.54953306,7.47836577,-1.03440638,-1.2554766,-1.97121066,-3.23937634,-4.19964903,2.00237086,-0.77858123,-2.21961115,2.12928007,1.61118427],[4.64356373,3.66793051,-1.4625734,1.85675349,0.36468666,-3.21174249,3.16819974,-2.67062973,0.98627829,1.65240461,1.14385112,2.25526873,3.77115506,-2.72237606,-1.83011433,-2.02317991,-0.39899644,0.79665898,-1.3692368,0.85016276,-5.93418713,0.12860023,3.53752044,-6.15202762,0.29983488,-1.38827482,-0.57033589,-2.6121959,3.20465453,-1.64506405,0.5568253,-0.0072255,-7.02265707,3.53883063,-2.3651524,6.16298653,-2.40099926,0.85754924,-2.52765841,1.93762507,0.07856613,2.32110476,1.22649352,2.23908938,3.3761712,1.33534002,0.49023238,3.92279728,-2.50019345,1.23074726,1.00897174,-2.44376352,-1.58948354,-1.52907352,4.72070963],[-0.0955725,4.41687574,1.44926214,3.08526479,-0.53188982,-0.59158357,-2.23569596,1.12124103,-1.25194061,-2.57435923,-0.38406555,1.21977986,-0.01145313,-1.28840636,4.77893183,-1.20751584,-0.82714757,-2.89549227,-3.81298331,-1.08153532,-0.33465561,-1.85824797,0.16510136,0.38897548,-4.05792935,0.84761515,-0.22286065,-1.43793587,-2.74004082,-2.08900606,1.8689089,-3.67387906,-5.93242217,-0.62103267,7.73411456,-5.51170178,-2.34766675,2.32166962,-2.361268,-2.88438457,1.28889654,-2.81675042,5.3780953,-0.89042954,2.12231538,-3.47917748,-2.56953539,-0.35119161,-2.34080486,3.29651757,0.42640973,1.1329309,2.49456799,-2.47199155,3.22045107],[-1.3021829,0.38118938,2.83378911,-3.12270736,0.0469275,-0.02542857,5.31342626,1.37096387,0.16583656,2.22575619,0.38299,5.62799342,2.19027829,0.69768693,1.1790105,4.15205097,-3.66605354,-1.91837883,-4.99715511,0.71593562,1.30829704,-0.01213546,-2.44858232,-4.00683926,-4.19206144,-0.1216387,-0.04786071,1.65751992,-4.44709257,0.58324501,0.97516387,-0.32853572,-0.21768286,0.31863802,2.24821625,4.34438467,0.30650716,4.44455272,-2.4867666,1.33861085,-6.52216895,4.78273704,-2.43346596,0.7906704,0.85157158,2.59997948,0.61894355,0.71514646,-2.71575834,-0.63800389,4.45372008,1.08743398,-2.26112703,2.37795143,-0.37194152],[-3.03087516e+00,-1.83944087e+00,-3.72573679e-01,6.08119823e+00,5.03246191e+00,5.25331611e+00,3.02310465e+00,-2.30977109e+00,1.93375331e+00,-4.75723393e-01,3.31802268e+00,4.16001845e+00,2.61298493e+00,-3.42880006e+00,2.73429647e-01,1.30164781e+00,1.40446736e+00,-2.15131792e-01,6.37046955e+00,1.59084787e+00,2.44336085e+00,-6.03253011e+00,5.64788401e-03,2.69617613e+00,-3.92097123e+00,4.79443138e+00,4.58192734e+00,1.46772153e+00,3.13538083e+00,-3.35799393e+00,1.85060677e+00,-2.53621163e+00,-7.99159319e-01,-1.51964740e+00,-1.61573348e+00,-6.69001302e+00,-2.28088957e+00,-5.33522550e+00,-5.29363471e+00,2.91871861e+00,6.12274563e+00,6.86634330e+00,3.59868956e-01,-1.06452922e+00,3.27490101e+00,-2.13397615e+00,-1.63682821e+00,-8.98975726e-01,4.16377584e+00,4.44089134e+00,-2.55948992e+00,-1.70854205e+00,6.14848524e-01,-2.07941390e+00,1.09472645e+00],[1.55744996,-1.65864096,-2.08528387,-1.50158231,-0.71258566,-2.33435412,3.69432138,-1.15994397,-1.01951969,-0.12072404,-1.94948293,0.28004795,1.0234626,5.03667399,-0.27200464,2.28570292,7.62217062,1.54877158,0.56218944,2.33514877,1.75417142,-2.54282329,-1.96514936,-5.42650575,6.4530925,4.20323796,-0.15504487,6.05413235,-0.04898817,1.12563147,-1.36835906,3.44248002,0.97250069,-5.0367091,3.30155075,-1.23891801,-0.6174174,1.81382368,2.05348075,7.9589297,-3.83119415,5.7531,-0.38560483,0.28348698,1.15518919,-0.95165245,-8.57445507,0.07089916,-0.67955056,7.69472053,3.06142954,8.38805875,3.39436181,-1.23899583,0.59272948],[-1.46546729,0.9950538,-6.53272418,-2.62195897,-1.50946312,2.16916657,-4.5572516,0.16293416,-1.75430052,-4.80725301,0.18235023,-3.42796354,-4.32247822,-1.14753584,0.39323048,4.90690346,-1.7117885,4.24135451,-2.32432211,1.00997865,4.92444177,0.36232064,-4.25724698,-2.83424311,-5.18780889,0.12097375,-2.84954032,-4.99931357,-1.00966124,-1.32017386,-3.37372596,0.71014652,0.12271294,5.60736413,3.95750164,-0.92862088,-2.43109734,-4.48474754,1.05524985,-0.0916654,1.99887968,-1.98714289,-0.75383654,0.92491653,-2.71373575,7.41176009,-4.46947173,3.12010608,3.42807468,-3.24472798,-3.45027248,3.09728683,7.09062789,0.21587962,0.44014575],[-1.07937629,-2.22098809,-2.59964842,3.80353394,6.308656,-1.42229491,0.97024403,0.29919355,-2.73052499,-0.62591052,-0.64014099,7.1855852,-9.14263646,-0.06913567,3.52001074,0.23314976,3.0829643,3.72870788,-1.34326458,1.02648203,0.9566857,-4.11034486,2.04145987,-3.81180598,1.50766575,4.96688881,-0.36462081,-1.69498657,-4.59741419,-0.30035897,1.65636494,6.806582,-1.50705971,4.60946011,7.03745064,-2.30773443,-2.31205973,-0.03574934,0.78845533,-1.83899554,-0.84234311,2.85445629,-0.28660367,7.73028842,6.85159769,0.40446551,-1.67005359,4.49835426,-2.91703972,1.03685049,5.62177379,5.92940252,-1.45245643,1.00304015,2.67205736]])
        print(len(self.pesosCapaOculta[0]))
        self.pesosCapaSalida = np.array([ 1.42836811,2.35888883,-0.15781838,0.83819872,0.75879613,-4.59399388,-1.49398467,0.37475927,1.57348491,1.99288901,1.28317179,2.91856509,-2.45347023,0.88089455,1.44437785,0.44776998,-0.61725887,-4.18934193,2.27176863])
        self.biasCapaOculta = 1.7268800064325531
        self.biasCapaSalida = 2.9022300003110884
        self.generarMatrizDeConfusion(1)
        self.calcularPrecision(1)
        print("error global: {}".format(self.error))
        print("matriz de confusion")
        print(self.matrizDeConfusion[0])
        print(self.matrizDeConfusion[1])
        print("precision")
        print(self.precision)
        return self.y_resultado,self.salidaDeseada
        
if __name__ == '__main__':
    e = RedNeuronal()
    e.backpropagation1CapaOculta(e.datos.Datos, e.datos.Resultado,10)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #error2 = e.backpropagation1CapaOculta(e.datos.Datos, e.datos.Resultado,19)
    #error3 = e.backpropagation1CapaOculta(e.datos.Datos, e.datos.Resultado,30)
    # Creamos la figura
    '''plt.figure()
    
    # Dibujamos  en negro discontinuo con etiqueta y1
    plt.plot([0],[80], 'k--', linewidth = 2, label = '9')
    
    # Mantenemos la misma figura parta la siguiente grÃ¡fica
    plt.hold(True)
    
    # Esta vez dibujamos - y en rojo co etiqueta y2
    plt.plot([105],[80],'-', linewidth = 2, label = '19')
    
    # Mantenemos la misma figura parta la siguiente grÃ¡fica
    plt.hold(True)
    
    # Esta vez dibujamos - y en rojo co etiqueta y2
    plt.plot([0],[200],'_', linewidth = 2, label = '29')
    
    # AÃ±adimos la leyenda
    plt.legend(loc = 3)
    
    # AÃ±adimos las etiquetas poniermo en Latex "mu" sÃ­mbolo de micras
    plt.xlabel("Iteración", fontsize = 24, color = (1,0,0))
    plt.ylabel("ECM", fontsize = 24, color = 'blue')
    
    # AÃ±adimos texto
    plt.text(x = 1, y = 0.0, s = u'T = 0.05', fontsize = 24)
    
    # AÃ±adimos la rejilla
    plt.grid(True)
    plt.grid(color = '0.5', linestyle = '--', linewidth = 1)
    
    # AÃ±adimos los ejes
    plt.axis('tight')
    
    # AÃ±adimos el tÃ­tulo 
    plt.title('Comparación',fontsize = 28, color = '0.75', verticalalignment = 'baseline', horizontalalignment = 'center')
    
    # Guardamos
    plt.savefig('plotCompleta.png')
        
    # Mostramos en pantalla
    plt.show()'''
    
    
    
