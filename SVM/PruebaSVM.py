
#Prueba2
from sklearn import svm, datasets
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from Datos.Datos import Datos
from sklearn.datasets import make_blobs
import numpy as np
import math
import pickle

class SVM(object):
    
    def __init__(self):

        
        self.datos = Datos()
        self.datos.datosConEnterosNormalizados()
        self.entradaDeseada = self.datos.Datos
        self.salidaDeseada = self.datos.Resultado
        X = self.datos.Datos
        y = self.datos.Resultado
        X,X_test,y,y_test =  train_test_split(X, y, test_size=.8,
                                                    random_state=0)
        C = 1.0  # SVM regularization parameter
        models = (svm.SVC(kernel='linear', C=C),
                  svm.LinearSVC(C=C),
                  svm.SVC(kernel='rbf', gamma=0.7, C=C),
                  svm.SVC(kernel='poly', degree=4, C=C))
        titles = ('SVC with linear kernel',
                  'LinearSVC (linear kernel)',
                  'SVC with RBF kernel',
                  'SVC with polynomial (degree 3) kernel')
        
        for key in (0,1,2,3):
            print(titles[key])
            model = models[key].fit(X, y)
            score = model.score(X_test,y_test)
            Y_resultado = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, Y_resultado)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(" {} ---->  precisión: {}".format(titles[key], score))
            plt.legend(loc="lower right")
            plt.show()
            
            score = model.score(X,y)
            print(score)
            self.generarMatrizDeConfusion(model)
            #filename = 'PruebaDatosRetinopatia.sav'+titles[key]
            #pickle.dump(models[key], open(filename, 'wb'))
    
    def generarMatrizDeConfusion(self,model ,porcentajePrueba = 0):
        self.matrizDeConfusion = [[0,0],[0,0]]
        if porcentajePrueba == 0:
            elementos = len(self.entradaDeseada)
        else:
            elementos = np.random.permutation(math.ceil(len(self.entradaDeseada)*porcentajePrueba)) # lista desordenada para el test
        self.error=0
        for j in range(elementos):
            entrada = self.entradaDeseada[j]
            salidaEsperada = self.salidaDeseada[j]
            salida = model.predict([entrada])[0];
            self.error += 0.5*(salida - salidaEsperada)**2
            self.matrizDeConfusion[round(salida)][salidaEsperada] += 1
            #print("salida {} salidadeseada {}".format(salida,salidaEsperada))
            
        print("matriz de confusion")
        print(self.matrizDeConfusion[0])
        print(self.matrizDeConfusion[1])
        
        self.precision =  (self.matrizDeConfusion[0][0] + self.matrizDeConfusion[1][1] )/elementos
        #print(self.precision)
            
        
        
    
    
    #loaded_model = pickle.load(open(filename, 'rb'))
    #print(loaded_model.predict(X))
if __name__ == '__main__':
    SVM()
