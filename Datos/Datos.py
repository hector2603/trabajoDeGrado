'''
Created on 3/10/2017

@author: hector
'''
from _ast import Try
class Datos(object):
    Datos = []
    Resultado = []
    Archivo = 0 # archivo de los datos
    
    def __init__(self):
        try:
            self.Archivo = open("../datos_retinopatia.arff", "r")
        except Exception as inst:
            self.Archivo = open("datos_retinopatia.arff", "r")
    
    #funcion que asigna los arreglos de datos y resultados usando la funcion de microaneurismas con binarios 
    def datosConBinarios(self):
        datosProcesados = []
        resultado = []
        for (key,linea) in enumerate(self.Archivo.readlines()[24:]):
            lineaSeparada = self.separarLinea(linea)
            self.normalizarExudados(lineaSeparada)
            lineaSeparada=self.normalizarAneurismasBinarios(lineaSeparada)
            #print key
            datosProcesados.append(lineaSeparada[:-1])
            resultado.append(lineaSeparada[-1])
            #print len(lineaSeparada)
        self.Datos = datosProcesados
        self.Resultado = resultado
       
    #funcion que asigna los arreglos de datos y resultados usando la funcion de microaneurismas normalizados 
    def datosConEnterosNormalizados(self):
        datosProcesados = []
        resultado = []
        for (key,linea) in enumerate(self.Archivo.readlines()[24:]):
            lineaSeparada = self.separarLinea(linea)
            #self.normalizarExudados(lineaSeparada)
            #lineaSeparada=self.normalizarAneurismas(lineaSeparada)
            #print key
            datosEnteros = [float(x) for x in lineaSeparada[:-1]]
            datosProcesados.append(datosEnteros)
            resultado.append(int(lineaSeparada[-1]))
            #print len(lineaSeparada)
        self.Datos = datosProcesados
        self.Resultado = resultado
        
    def separarLinea(self,linea):
        return linea[:-1].split(",")
    
    #funcion que normaliza los datos de los exudados y la distancia del disco optico a la macula y los convvierte en float, los demas datos los convierte en int
    def normalizarExudados(self,linea):
        for (key,dato) in enumerate(linea):
            if 8 <= key <= 17:
                linea[key]=float(dato)/100
            else:
                linea[key]=int(dato)
            if linea[key]==0 and key != 19:
                linea[key]=-1
    #funcion que convierte a binario el numero de microaneurismas 
    def normalizarAneurismasBinarios(self,linea):
        lista = []
        binarios = [64,32,16,8,4,2,1]
        for (key,dato) in enumerate(linea):
            if 2 <= key <= 7:
                linea[key]=bin(dato)
                for x in range(0,7):
                    if binarios[x]&dato == 0:
                        lista.append(-1)
                    else:
                        lista.append(1)
        lineanueva = linea[0:2] + lista + linea[8:]
        return lineanueva
    
    #funcion que normaliza los microaneurismas
    def normalizarAneurismas(self,linea):
        for (key,dato) in enumerate(linea):
            if 2 <= key <= 7:
                if linea[key] !=-1 :
                    linea[key] = dato
        return linea 
    
    # convierte una entrada de 19 a una entrada de 55 datos 
    def Convertir19A55(self,entrada):
        self.normalizarExudados(entrada)
        entrada = self.normalizarAneurismasBinarios(entrada)
        return entrada
        
            
        


if __name__ == '__main__':
    d = Datos()
    da = [1.0, 1.0, 1.0, 2.0, 3.0, 4, 5, 6, 7, 7, 8, 9, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    print(d.Convertir19A55(da))
    
    
    