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

    def __init__(self):
        self.factorEntrenamiento = 0.1;
        self.errorPermitido = 0.1;
        self.datos = Datos()
        self.datos.datosConBinarios()
        self.entradaDeseada = self.datos.Datos
        self.salidaDeseada = self.datos.Resultado
        #pesosCapaOculta,pesosCapaSalida = self.backpropagation(self.factorEntrenamiento, self.errorPermitido, self.entradaDeseada, self.salidaDeseada,19,10)
        #pesosCapaOculta,pesosCapaSalida = self.backpropagation1CapaOculta(self.factorEntrenamiento, self.errorPermitido, self.entradaDeseada, self.salidaDeseada,19)
        #self.ProbarModelo()

    def sigmoide(self,x):
        return 1 / (1 + math.exp(-x))
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
                print("matriz de confusion")
                print(self.matrizDeConfusion[0])
                print(self.matrizDeConfusion[1])
                print("precision")
                print(self.precision)
                
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
    
    def backpropagation1CapaOculta(self,entrenamiento,errorPermitido,entradaDeseada,salidaDeseada,numeroNeuronasCapaOculta):
        
        self.pesosCapaOculta = pesosCapaOculta = np.random.rand(numeroNeuronasCapaOculta,len(entradaDeseada[0])) #np.array([[0.3568718 ,  0.6821255],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969],[0.0093142 ,  0.2266969]])#pesos de la capa oculta, deben ser aleatorios
        self.pesosCapaSalida = pesosCapaSalida = np.random.rand(numeroNeuronasCapaOculta)  #np.array([0.69900,0.50459,0.5859,0.4789])# pesos de la capa de salida
        
        elementos = np.random.permutation(math.ceil(len(entradaDeseada)*0.8)) # lista desordenada para el entrenamiento
        #elementos = [0,1,2,3] # orden temporal mientras las pruebas
        #print (elementos)
        mejorError=99999
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
                salidaObtenida,salidaCapaOculta,entradaNetaCapaOculta,entradaNetaCapaSalida = self.RedNeuronal1CapaOculta(patronIn)
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
                
                self.pesosCapaOculta = pesosCapaOculta
                self.pesosCapaSalida = pesosCapaSalida
                
                #Calculamos el error
                self.generarMatrizDeConfusion(1)
                self.calcularPrecision(1)
                    
                print("error global: {}".format(self.error))
                print("matriz de confusion")
                print(self.matrizDeConfusion[0])
                print(self.matrizDeConfusion[1])
                print("precision")
                print(self.precision)
                
                if(self.error<mejorError):
                    mejorError = self.error
                    F = open("matriz.txt","w")
                    F.write("pesos capa oculta 1")
                    for m in range(len(pesosCapaOculta)):
                            F.write(str(pesosCapaOculta[m]))
                    F.write("pesos capa salida")
                    F.write(str(pesosCapaSalida))
                    F.write("mejor error")
                    F.write(str(mejorError))
                    F.close()
                if(self.error <errorPermitido):
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
    
    # funcion que genera la matriz de confusion
    # porcentajePrueba, el porcentaje de datos del dataset que se van a usar para las pruebas
    # numeroCapas =  1 para cuando el numero de capas ocultas es una, 2 si el numero de capas ocultas son dos
    def generarMatrizDeConfusion(self,porcentajePrueba,numeroCapas=1):
        self.matrizDeConfusion = [[0,0],[0,0]]
        elementos = np.random.permutation(math.ceil(len(self.entradaDeseada)*porcentajePrueba)) # lista desordenada para el test
        self.error=0
        for j in elementos:
            entrada = self.entradaDeseada[j]
            salidaEsperada = self.salidaDeseada[j]
            if numeroCapas == 1:
                salida,a,b,c = self.RedNeuronal1CapaOculta(entrada)
            else:
                salida,a,b,c,d,e = self.RedNeuronal(entrada)
            self.error += 0.5*(salida - salidaEsperada)**2
            self.matrizDeConfusion[round(salida)][salidaEsperada] += 1
            #print("salida {} salidadeseada {}".format(salida,salidaEsperada))
            
    def calcularPrecision(self,porcentajePrueba):
        total = math.ceil(len(self.entradaDeseada)*porcentajePrueba)
        self.precision = (self.matrizDeConfusion[0][0] + self.matrizDeConfusion[1][1])/total
        
    def ProbarModelo(self,pesosCapaOculta,pesosCapaSalida,biasCapaOculta,biasCapaSalida):
        #self.pesosCapaOculta = np.array([[0.62972763,-0.33723407,-1.08255794,0.02724721,0.57767761,-0.2487474,-0.69581832,1.45393275,0.67232272,0.17904853,-0.59724445,1.9636341,0.22386637,-1.18376154,0.90769228,1.53064995,0.52666167,0.397916,0.04875601,2.8269017,0.88177378,2.13350652,0.4562378,1.13073135,0.76204806,-0.2328692,1.61166143,-1.12962299,1.98955297,1.09249102,1.363435,0.53469893,1.01644712,2.73835149,1.77346195,-0.46331771,3.12786825,-0.15702667,2.72486875,-0.27585965,2.74799486,2.18629686,1.95426053,1.10296117,1.20602334,1.64577516,0.06937628,0.86685781,-1.02739663,0.61893413,2.4117933,0.7923125,0.85836492,0.48032376,2.20176826],[0.11002702,0.26355545,2.7755299,-0.11942971,1.40115458,0.1694512,-0.9970919,1.50814064,-1.08172959,2.2629862,0.71247446,-0.61198922,1.49034432,1.59147045,0.64901848,-0.47829422,0.74017721,2.14120731,0.25200953,0.45950122,1.72653862,1.52732597,0.0271832,0.93208941,2.40786394,-0.75393802,0.69587332,-0.31382507,-0.29799765,1.54206456,0.34039248,0.83011701,1.25519273,-0.43174128,1.66407018,1.25052775,0.40838819,0.32182125,1.84022993,-0.34464222,-0.09565023,1.05782243,0.51934702,1.1146469,-0.60725702,0.36412315,0.03909931,0.98619219,0.11432104,0.23907546,0.97502534,1.53553664,0.95120314,0.84394429,1.49509471],[-0.20007113,0.47518662,1.65679547,1.86611951,2.2328,-0.18619276,1.72817849,2.24173486,1.34957083,1.14849177,1.70313182,0.81186416,-1.16348676,3.14585685,1.0527425,-0.13138222,1.95382632,0.09298333,0.58411299,1.53184131,-1.34936091,0.84732175,2.36862402,1.07873451,-0.84114243,-0.02529284,0.68675924,-0.20288779,-0.23302494,2.72355292,0.08246072,0.73021548,-1.59688761,1.65937712,0.05160105,-0.21427971,0.56172815,0.67193657,0.11947396,0.67053609,0.51292926,2.79298937,-2.01411233,1.58356626,2.90491901,-0.09337754,0.55808316,0.7717971,0.33345068,0.81119372,0.48847137,1.71884915,0.87986419,0.2240618,0.56092157],[-0.48335391,0.89720804,1.33456048,1.59543292,1.24616661,0.3687664,1.08398288,1.15346702,2.32213568,1.39053511,0.80772711,0.10440265,0.75139024,0.88314889,-0.06541093,-0.54869444,0.61899383,1.04347926,-0.01135654,1.32774952,-0.32829638,1.68952593,-0.12195938,0.52088412,0.82956322,0.75525991,0.37684538,2.01749167,0.27113515,-1.59925246,0.85367277,0.51174563,2.6083158,0.18796788,-0.3367419,0.75131395,0.14682224,0.99699337,0.92184029,0.416978,-0.319526,0.56549156,1.6764084,0.36076527,0.10178246,-1.01709166,0.81574258,0.95292761,-0.62973477,0.39508574,0.95069075,1.32018319,0.32136882,0.46798864,1.29895503],[-0.72321154,0.77648174,2.70996005,0.41298562,-0.91573388,0.8062967,-0.69281056,-0.49396912,1.0409624,0.68995345,2.22683458,0.76108751,-0.01483403,1.31529747,-0.52164118,-0.4266642,0.75665064,0.09848374,1.16403503,-0.21652293,0.03217823,0.46727784,2.37191519,-0.10611002,-0.38458396,0.64572401,1.99112519,1.98839092,1.66210131,0.19786008,0.8348327,-1.38238952,0.60409842,-0.33980274,0.08599113,-1.48951183,0.77519512,0.70897432,0.38919703,0.85034977,0.75699055,-0.62457075,1.52882529,0.53712649,-0.69874511,-0.82765434,-0.5554365,1.45224308,1.08354984,1.05446045,0.61162248,0.74382059,0.91476151,0.82155781,0.38930058],[1.59081291,1.76074136,1.87087983,2.14642674,1.05799474,-0.85977135,1.94542949,1.6466357,1.77714612,0.66735662,1.54437663,0.99792642,1.46806566,-1.99213895,0.5723292,0.65313283,1.47062164,1.17161291,-0.30992064,0.61362338,-0.52596987,-0.36345721,1.64042908,1.01692951,-1.0006965,1.04121579,-0.00479114,2.64756791,-0.54829995,1.18612114,0.34269364,-1.04345464,1.47593906,-0.15815237,2.12080919,0.75981346,-0.1304503,-0.18487782,0.18723494,2.10879255,1.91631716,1.8163198,2.60312088,1.33988792,1.38665325,-0.01031081,0.28449583,-0.34996379,-0.18753698,-0.5150921,-0.19250171,0.79424164,0.70900762,0.8262412,0.93143438],[1.33956849,0.59497364,0.77886039,1.08081842,-0.77063863,-0.70688278,1.55235245,2.17086833,-0.18999941,1.44469166,0.28232654,-0.15239269,1.95360944,1.83249767,2.03101095,-0.3678401,1.64169679,-0.36348736,0.41394412,-1.04313542,-1.78743932,3.1924236,0.3264953,1.04233678,0.70727947,2.3326002,-0.18196411,1.14222918,1.16862835,0.1824333,1.39862425,0.43807343,0.62274741,0.3296913,-0.13873598,1.21273701,-0.92601648,0.16498764,0.10011755,2.32393981,-0.37646368,1.57786828,-0.95976609,-0.42067329,1.99389989,1.6253411,0.38983225,0.8365269,0.7359938,1.6482715,0.50980488,0.72619334,0.3538259,0.59127896,0.10707699],[0.80380272,2.02653561,-0.39726214,0.0446585,-1.12423855,0.85499735,1.56518337,0.10166213,1.50575345,-0.57107415,-0.44884482,0.61885402,2.23386227,0.32156441,-0.09341366,1.09413817,0.04590635,0.32538907,3.00267257,1.31262555,0.69691283,0.33691542,-0.32844524,0.9952698,1.48304903,0.55914917,1.07319727,1.33401799,-0.92409961,-1.37425054,1.44599835,0.77269967,0.1212827,-0.65104894,1.16848863,0.9875591,-1.9779509,0.5364244,1.72031785,1.17043161,2.24480145,1.2277942,2.71539999,2.289536,-0.77144885,2.42770833,1.60592057,1.12153161,0.57246076,-0.40491969,-0.00993519,0.28511704,0.51178899,0.6225254,-1.81519876],[0.63420613,-0.75080912,0.87066859,0.49635251,2.46815068,0.4098518,-1.62402862,1.30940206,-0.4221099,0.9859362,0.56607067,1.0387536,0.47348342,1.00224105,0.26048816,1.79947175,0.51940717,-0.25171927,0.54124555,2.41302782,1.96471269,-0.65584922,0.19996935,0.49277859,0.1100048,1.18771136,1.29327344,1.27251162,0.20575265,0.36889731,0.79439199,0.77141003,0.31221922,0.27921062,0.4380318,-2.10514603,0.607217,0.74616901,-0.02556221,1.71832931,0.98290328,0.56889853,-1.6656339,-0.00358854,1.36096782,0.42444006,0.1457114,0.6225656,-0.18626712,0.05271653,1.07551351,0.19253434,0.98118868,0.55701347,0.73758893],[-0.8278623,-0.17668815,-0.30701783,-0.03121358,-0.00359238,-0.06876695,0.95864154,0.69326578,1.03037488,0.15933364,0.57807784,0.91669418,0.70461556,0.64658529,0.00757632,3.30948762,-0.35005523,-0.92560166,0.15157444,1.62221497,1.84445172,0.52808757,2.57718274,0.89328396,0.42191722,0.35114258,1.03179241,2.02210344,-0.38679019,-0.52429388,0.9840805,1.78926434,1.49782942,1.51741235,0.94087476,1.89984007,0.6980188,0.9052104,1.2666005,1.8712523,0.14293573,0.56583509,0.89091332,1.96911818,-0.8463918,1.15614359,0.81528236,1.5880498,0.47277021,1.96237915,-0.86981329,-0.43034789,0.17615137,0.96678589,2.10136857],[1.34075033,-1.33715072,2.01126173,-0.00688274,0.28764881,-0.260743,2.39428275,2.07941422,-0.82979388,1.36346142,0.14625602,0.66237472,-0.34195018,1.73421789,-0.03933327,2.97833986,0.5883614,1.09629242,-1.3349231,1.71682815,-1.25157951,-1.54438058,-2.07684302,-0.52821224,1.16391747,1.04630803,-1.30932793,-0.47440596,2.68425567,0.38765448,0.31184629,1.23043691,-1.00160313,1.1712577,-0.10147886,2.18270879,0.45188533,0.44804453,0.3549314,0.49843628,-0.42009463,1.62670877,2.25656939,0.11811388,-0.14826327,0.15755337,-0.09590552,0.20970114,0.25396569,0.94650227,-0.27907858,0.16462798,0.53129268,0.97465731,0.78578396],[0.34592969,-0.20834099,0.96213641,0.93707031,-0.82333321,1.73751854,-0.15613746,0.27633407,3.8995425,0.40753841,1.38399932,0.99180078,0.04372506,0.64377366,0.31214858,-0.4185949,0.22968352,1.10689319,2.08406017,-0.09283607,2.54979857,2.20884545,-2.88686114,0.77290879,0.10976391,-1.42050928,0.67752166,3.15993606,0.65996168,0.76768141,0.06558643,0.92246347,-0.53298774,-0.14074604,0.62228049,-0.28559769,0.6722718,0.5707168,0.52500428,-0.46886363,0.4996984,0.65392159,2.02165963,1.16242968,3.68932869,-1.14551561,1.03562664,-2.65527214,-0.7713427,0.65527874,0.24559192,1.00651215,0.64905846,0.77108564,-0.34236371],[0.90297104,0.14143566,-0.29471783,1.68256678,0.7745197,1.74211845,-1.02882274,-0.46133599,-0.4769613,0.97422258,1.24328567,2.45545919,0.06267022,0.44251875,-0.0927202,-0.28209238,0.66825053,1.31619756,0.54202294,0.52537316,1.05709371,1.27170969,-1.72396232,1.29391847,0.68158832,0.70290869,-0.06869774,0.98844077,1.55972875,0.6210778,1.16188375,0.95915245,0.90269033,-0.29299538,0.08961142,-0.85651818,0.97053842,0.16231838,0.64271002,1.4190606,0.17438914,0.75430852,0.02935225,1.20147882,-0.15445821,0.22096343,0.57439529,-0.27815316,1.628692,0.82372855,-0.79541166,-0.09423679,0.91868981,0.10152649,0.71476458],[0.26772135,0.07042521,0.71813155,0.75683755,2.51830125,0.06091144,0.33721563,-0.36373065,0.98501453,0.52682579,1.14607215,1.87169123,1.86389015,0.96466463,-0.81276581,1.68155082,1.01291614,0.15788093,0.76004869,0.62848447,1.32536258,0.69460097,1.60067763,0.52168061,0.51918252,0.92392892,0.93840613,-0.06987614,-1.61115514,0.14287956,0.89023939,0.41235886,0.39786715,-0.18844928,2.06737305,0.39697418,0.70490279,0.43875149,-0.07421018,0.31251646,1.25433618,0.29044152,0.98506388,1.29431399,-0.36138301,-0.32950295,-0.37600338,0.49348481,0.11762595,-0.72447073,0.50262683,1.24687684,0.36073037,0.16967807,-0.11173837],[0.29251456,0.16717115,1.9919551,-1.08654902,0.21835299,1.73450039,0.34270423,-0.51782635,-1.37781377,1.43169096,-0.37252217,-0.40412067,0.12340524,-0.93280731,0.37346704,0.67230667,0.09203754,1.70390114,0.06196267,-0.90160776,-0.70031906,-0.63130626,2.17391107,0.53581672,1.53093382,-0.47412399,0.16555712,-1.83637857,1.56093048,-2.2384612,-0.51790635,-0.58897475,1.90002188,1.52589066,-1.1263222,1.00481078,-1.3874534,-0.41399405,1.42143185,1.28079228,0.47247451,0.67883513,0.84380946,0.97471161,0.84116826,-0.37550498,0.58939029,-0.00373572,0.08294536,0.33982994,2.65923209,0.50619328,0.72160415,0.08751195,0.30486405],[0.70434071,0.02625384,0.68340944,1.2841442,-0.98237799,-0.21521718,2.05292736,1.98984587,0.92574012,1.58270743,0.89178287,0.42220534,-0.5537176,1.65922071,0.35488821,0.78835043,0.92140881,-0.11036304,0.97370588,-0.02728739,0.38108425,-0.04401823,3.21655496,0.53577621,1.14504601,-0.04891532,2.62788569,1.58227265,0.70902083,-0.58982236,0.37530446,1.95448912,1.62399816,0.39816483,0.70186266,0.67263052,1.65901543,1.23267692,0.05873289,-0.25418087,0.20833793,-1.12575572,0.02953825,0.41245629,0.74874594,1.52068664,0.57231343,0.43158492,-0.02625321,0.76822803,0.51994109,1.62026104,0.49947618,0.49476606,-0.9360265,],[-0.23588812,-1.21876831,0.74401728,1.27654266,0.42119259,1.21689786,1.27627796,-1.16118015,-0.31326697,0.21020445,1.39964821,0.57405135,0.34782133,0.30190917,0.91741465,1.55575885,0.95531134,0.1138018,1.4610987,0.23180273,-1.90680742,3.11570147,-0.74488606,0.50267045,-0.7736957,0.72579407,0.56193328,0.36510859,-0.35555339,0.24664969,0.60967649,-0.85912145,0.89747686,0.63332119,-0.25664819,0.27439024,0.01163045,1.35243326,1.69464164,-1.74133494,-0.28508312,-1.25599039,-0.9746563,-0.468629,1.49331905,-0.4080355,0.38276094,1.65759231,1.08952836,1.6138047,0.17545872,0.31062929,0.56647155,0.00490186,-1.43724625],[0.44147631,3.88408863,-0.14553955,0.19787708,-0.98100461,-1.83632926,0.43096895,2.58356049,1.26404819,-0.30465665,-0.18236218,1.47302555,0.0649027,2.09706202,-3.52195437,-1.36971494,-0.69037603,1.60604968,-0.45232728,0.59427226,-2.0696163,0.50161793,-0.41164916,0.71301821,0.65875973,-0.02389045,-0.91802027,1.41757486,-2.18555514,5.95682588,1.22791847,1.22105835,-0.1427802,-1.12176142,-2.32219324,-1.51344766,-1.77199761,-0.10004421,1.76208001,0.53421353,-1.68853604,2.26837991,1.81632578,0.06576051,-0.72706655,1.98062788,-0.38241743,0.79673414,-4.83553076,0.16133476,-2.18583462,-2.07204554,0.64613666,0.60312798,0.68905415],[0.44583652,1.37697813,-0.71582944,1.21551862,0.53122446,0.431759,-1.08516982,-0.19133894,2.27269092,0.68161667,0.63909129,1.14897438,0.83613265,0.32343275,-0.58565989,1.01566185,1.20154615,0.76695403,1.62248673,-0.09695148,0.87479709,-1.23004118,0.14437555,2.40277865,0.8669818,0.14578078,-0.67093789,0.12926904,-0.41819837,1.89524541,1.91210376,1.05818059,2.64889132,0.7012283,1.18175579,0.15615029,-0.03303959,1.40863952,0.81138973,-0.81482309,-0.1466386,-0.3958516,1.96702951,0.27313907,0.03400407,1.69913033,1.41359028,1.45474843,0.93245488,1.05435722,-0.86334827,-0.06298241,0.23542136,0.95759291,0.30197674]])
        #self.pesosCapaSalida = np.array([-6.94335683,5.45412349,5.78930716,3.92492698,5.19828945,5.74864534,-6.73635607,-6.30403354,4.31949674,-5.75124391,5.62215911,6.37520338,-5.3092763,3.71834992,5.20652386,-5.83688495,5.70589252,-4.15367808,-5.8127265 ])
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
        
        
    
RedNeuronal()
