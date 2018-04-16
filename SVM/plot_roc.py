

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
from Datos.Datos import Datos
from backpropagation.Prueba import RedNeuronal



class Plot_roc:
    
    def __init__(self):
        self.datos = Datos()
        self.datos.datosConEnterosNormalizados()
        self.entradaDeseada = self.datos.Datos
        self.salidaDeseada = self.datos.Resultado

    def plot_roc_completa(self):
        X = self.datos.Datos
        y = self.datos.Resultado
        
        # cargar el modelo de SVM
        loaded_model_SVM = pickle.load(open("../SVM/SVC with linear kernelPruebaDatosRetinopatia.sav", 'rb'))
        Y_resultado = loaded_model_SVM.decision_function(X)
        fprSVM, tprSVM, _ = roc_curve(y, Y_resultado)
        roc_aucSVM = auc(fprSVM, tprSVM)
        
        #Cargar el modelo de Red neuronal entrenada por backpropagation
        backpropagation = RedNeuronal()
        Y_resultado,y=backpropagation.ProbarModeloBackpropagation()
        fprBP, tprBP, _ = roc_curve(y, Y_resultado)
        roc_aucBP = auc(fprBP, tprBP) 
        
        #Cargar el modelo de red neuronal entrenada por algoritmos genéticos
        algoritmoGenetico = RedNeuronal()
        Y_resultado,y=algoritmoGenetico.ProbarModeloAlgoritmosGeneticos()
        fprAG, tprAG, _ = roc_curve(y, Y_resultado)
        roc_aucAG = auc(fprAG, tprAG)             
        
        plt.figure()
        lw = 2
        plt.plot(fprAG, tprAG, color='darkorange',
                 lw=lw, label='Algoritmos Genéticos (área = %0.2f)' % roc_aucAG)#plot roc Algoritmos genéticos
        
        plt.plot(fprBP, tprBP,
         label='Backpropagation (área = {0:0.2f})'
               ''.format(roc_aucBP),
         color='deeppink', linestyle=':', linewidth=4)#plot roc Backpropagation

        plt.plot(fprSVM, tprSVM,
         label='SVM (área = {0:0.2f})'
               ''.format(roc_aucSVM),
         color='navy', linestyle='-.', linewidth=2)
        
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Ratio Falsos Positivos')
        plt.ylabel('Ratio Verdaderos Positivos')
        plt.title("Comparación")
        plt.legend(loc="lower right")
        plt.show()
        

if __name__ == '__main__':
    p = Plot_roc()
    p.plot_roc_completa()
