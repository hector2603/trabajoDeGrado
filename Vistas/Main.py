# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from Vistas.ProbarModelo import Ui_ProbarModelo

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 530)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 631, 181))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 430, 621, 71))
        self.label_2.setObjectName("label_2")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(230, 150, 164, 241))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.horizontalLayoutWidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        
        self.ProbarModelos = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.ProbarModelos.setMinimumSize(QtCore.QSize(162, 23))
        self.ProbarModelos.setMaximumSize(QtCore.QSize(162, 23))
        self.ProbarModelos.setObjectName("ProbarModelos")
        self.ProbarModelos.clicked.connect(self.openProbarModelo)
        
        self.verticalLayout.addWidget(self.ProbarModelos)
        '''self.GenerarModelos = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.GenerarModelos.setObjectName("GenerarModelos")
        self.verticalLayout.addWidget(self.GenerarModelos)
        self.VisualizarDatos = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.VisualizarDatos.setObjectName("VisualizarDatos")
        self.verticalLayout.addWidget(self.VisualizarDatos)'''
        self.label.raise_()
        self.label_2.raise_()
        self.horizontalLayoutWidget.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Trabaj de Grado"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; color:#ff0000;\">Categorización de retinopatía diabética <br/>utilizando algoritmos de inteligencia artificial</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">Héctor Fabio Ocampo Arbeláez - 1355858</span></p><p align=\"center\"><span style=\" font-size:16pt;\">Universidad del valle</span></p></body></html>"))
        self.ProbarModelos.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Aquí se podrá encontrar los modelos utilizados para las pruebas en el proyecto, generar la gráfica ROC y utilizar los modelos para generar una categorización a partir de datos en concreto. </span></p></body></html>"))
        self.ProbarModelos.setText(_translate("MainWindow", "Probar Modelos Guardados"))
        #self.GenerarModelos.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Aquí se podrá generar nuevos entrenamientos de las técnicas de redes neuronales con backpropagation y algoritmos evolutivos. </span></p></body></html>"))
        #self.GenerarModelos.setText(_translate("MainWindow", "Generar modelos"))
        #self.VisualizarDatos.setText(_translate("MainWindow", "Visualizar los daos"))
    
    def openProbarModelo(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_ProbarModelo()
        self.ui.setupUi(self.window)
        #MainWindow.hide()
        self.window.show()        


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

