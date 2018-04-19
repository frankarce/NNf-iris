from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

import matplotlib.pyplot as pl
matplotlib.use('TkAgg')
import numpy as np
#a es el conjunto de datos con 4 atributos refejados en 4 columnas, cada una de ellas reprecenta la medida
#en centimetros de el petalo y el cepalo del el largo y el ancho y la ultima columna contiene la clase donde
#0 es iris cetosa, el 1 es iris virginica y el 2 es iris versicolor.
a=[
[5.1,3.5,1.4,0.2,0],
[4.9,3.0,1.4,0.2,0],
[4.7,3.2,1.3,0.2,0],
[4.6,3.1,1.5,0.2,0],
[5.0,3.6,1.4,0.2,0],
[5.4,3.9,1.7,0.4,0],
[4.6,3.4,1.4,0.3,0],
[5.0,3.4,1.5,0.2,0],
[4.4,2.9,1.4,0.2,0],
[4.9,3.1,1.5,0.1,0],
[5.4,3.7,1.5,0.2,0],
[4.8,3.4,1.6,0.2,0],
[4.8,3.0,1.4,0.1,0],
[4.3,3.0,1.1,0.1,0],
[5.8,4.0,1.2,0.2,0],
[5.7,4.4,1.5,0.4,0],
[5.4,3.9,1.3,0.4,0],
[5.1,3.5,1.4,0.3,0],
[5.7,3.8,1.7,0.3,0],
[5.1,3.8,1.5,0.3,0],
[5.4,3.4,1.7,0.2,0],
[5.1,3.7,1.5,0.4,0],
[4.6,3.6,1.0,0.2,0],
[5.1,3.3,1.7,0.5,0],
[4.8,3.4,1.9,0.2,0],
[5.0,3.0,1.6,0.2,0],
[5.0,3.4,1.6,0.4,0],
[5.2,3.5,1.5,0.2,0],
[5.2,3.4,1.4,0.2,0],
[4.7,3.2,1.6,0.2,0],
[4.8,3.1,1.6,0.2,0],
[5.4,3.4,1.5,0.4,0],
[5.2,4.1,1.5,0.1,0],
[5.5,4.2,1.4,0.2,0],
[4.9,3.1,1.5,0.1,0],
[5.0,3.2,1.2,0.2,0],
[5.5,3.5,1.3,0.2,0],
[4.9,3.1,1.5,0.1,0],
[4.4,3.0,1.3,0.2,0],
[5.1,3.4,1.5,0.2,0],
[5.0,3.5,1.3,0.3,0],
[4.5,2.3,1.3,0.3,0],
[4.4,3.2,1.3,0.2,0],
[5.0,3.5,1.6,0.6,0],
[5.1,3.8,1.9,0.4,0],
[4.8,3.0,1.4,0.3,0],
[5.1,3.8,1.6,0.2,0],
[4.6,3.2,1.4,0.2,0],
[5.3,3.7,1.5,0.2,0],
[5.0,3.3,1.4,0.2,0],
[7.0,3.2,4.7,1.4,1],
[6.4,3.2,4.5,1.5,1],
[6.9,3.1,4.9,1.5,1],
[5.5,2.3,4.0,1.3,1],
[6.5,2.8,4.6,1.5,1],
[5.7,2.8,4.5,1.3,1],
[6.3,3.3,4.7,1.6,1],
[4.9,2.4,3.3,1.0,1],
[6.6,2.9,4.6,1.3,1],
[5.2,2.7,3.9,1.4,1],
[5.0,2.0,3.5,1.0,1],
[5.9,3.0,4.2,1.5,1],
[6.0,2.2,4.0,1.0,1],
[6.1,2.9,4.7,1.4,1],
[5.6,2.9,3.6,1.3,1],
[6.7,3.1,4.4,1.4,1],
[5.6,3.0,4.5,1.5,1],
[5.8,2.7,4.1,1.0,1],
[6.2,2.2,4.5,1.5,1],
[5.6,2.5,3.9,1.1,1],
[5.9,3.2,4.8,1.8,1],
[6.1,2.8,4.0,1.3,1],
[6.3,2.5,4.9,1.5,1],
[6.1,2.8,4.7,1.2,1],
[6.4,2.9,4.3,1.3,1],
[6.6,3.0,4.4,1.4,1],
[6.8,2.8,4.8,1.4,1],
[6.7,3.0,5.0,1.7,1],
[6.0,2.9,4.5,1.5,1],
[5.7,2.6,3.5,1.0,1],
[5.5,2.4,3.8,1.1,1],
[5.5,2.4,3.7,1.0,1],
[5.8,2.7,3.9,1.2,1],
[6.0,2.7,5.1,1.6,1],
[5.4,3.0,4.5,1.5,1],
[6.0,3.4,4.5,1.6,1],
[6.7,3.1,4.7,1.5,1],
[6.3,2.3,4.4,1.3,1],
[5.6,3.0,4.1,1.3,1],
[5.5,2.5,4.0,1.3,1],
[5.5,2.6,4.4,1.2,1],
[6.1,3.0,4.6,1.4,1],
[5.8,2.6,4.0,1.2,1],
[5.0,2.3,3.3,1.0,1],
[5.6,2.7,4.2,1.3,1],
[5.7,3.0,4.2,1.2,1],
[5.7,2.9,4.2,1.3,1],
[6.2,2.9,4.3,1.3,1],
[5.1,2.5,3.0,1.1,1],
[5.7,2.8,4.1,1.3,1],
[6.3,3.3,6.0,2.5,2],
[5.8,2.7,5.1,1.9,2],
[7.1,3.0,5.9,2.1,2],
[6.3,2.9,5.6,1.8,2],
[6.5,3.0,5.8,2.2,2],
[7.6,3.0,6.6,2.1,2],
[4.9,2.5,4.5,1.7,2],
[7.3,2.9,6.3,1.8,2],
[6.7,2.5,5.8,1.8,2],
[7.2,3.6,6.1,2.5,2],
[6.5,3.2,5.1,2.0,2],
[6.4,2.7,5.3,1.9,2],
[6.8,3.0,5.5,2.1,2],
[5.7,2.5,5.0,2.0,2],
[5.8,2.8,5.1,2.4,2],
[6.4,3.2,5.3,2.3,2],
[6.5,3.0,5.5,1.8,2],
[7.7,3.8,6.7,2.2,2],
[7.7,2.6,6.9,2.3,2],
[6.0,2.2,5.0,1.5,2],
[6.9,3.2,5.7,2.3,2],
[5.6,2.8,4.9,2.0,2],
[7.7,2.8,6.7,2.0,2],
[6.3,2.7,4.9,1.8,2],
[6.7,3.3,5.7,2.1,2],
[7.2,3.2,6.0,1.8,2],
[6.2,2.8,4.8,1.8,2],
[6.1,3.0,4.9,1.8,2],
[6.4,2.8,5.6,2.1,2],
[7.2,3.0,5.8,1.6,2],
[7.4,2.8,6.1,1.9,2],
[7.9,3.8,6.4,2.0,2],
[6.4,2.8,5.6,2.2,2],
[6.3,2.8,5.1,1.5,2],
[6.1,2.6,5.6,1.4,2],
[7.7,3.0,6.1,2.3,2],
[6.3,3.4,5.6,2.4,2],
[6.4,3.1,5.5,1.8,2],
[6.0,3.0,4.8,1.8,2],
[6.9,3.1,5.4,2.1,2],
[6.7,3.1,5.6,2.4,2],
[6.9,3.1,5.1,2.3,2],
[5.8,2.7,5.1,1.9,2],
[6.8,3.2,5.9,2.3,2],
[6.7,3.3,5.7,2.5,2],
[6.7,3.0,5.2,2.3,2],
[6.3,2.5,5.0,1.9,2],
[6.5,3.0,5.2,2.0,2],
[6.2,3.4,5.4,2.3,2],
[5.9,3.0,5.1,1.8,2]
]

#primero configuramos el conjunto de datos a utilizar donde se prepara a "alldata" para contener
#las dimensiones de la entrada 4,1 y el numero de clases en nb_classes que en este caso son 3
alldata = ClassificationDataSet(4,1,  nb_classes=3)
#con este ciclo insertamos a alldata todo lo que contiene "a" donde en alldata.addSample la
#a[n][:4] contiene los atributos y a[n][4] contiene la clase
for n in range(len(a)):
        alldata.addSample(a[n][:4], a[n][4])
# aqui dividimos los datos en proporcion a lo que decidamos usar para entrenar y para probar, en este caso
# se usa el 80% para entrenar que se almacenara en trndata y el resto en partdata
trndata, partdata = alldata.splitWithProportion( 0.80 )
#ademas dividiremos partdata al 50% parttada contiene 30 elementos por lo que tstdata y validata tendra 15
#elementos cada 1
tstdata, validata = partdata.splitWithProportion( 0.20 )
#para la clasificacion en redes neuronales es recomendable codificar las clases con una neurona de salida
#por clase, esto sucede porque se usa la funcion de activacion SoftMaxLayer que lo requiere de esa manera 1 neurona por salida

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
validata._convertToOneOfMany()
print trndata.indim, trndata.outdim, tstdata.indim, tstdata.outdim

#configuracion de la red neuronal para este caso es una red neuronal difusa o FNN con 4 entradas
#3 neuronas ocultas y 3 neuronas de salida.
net = buildNetwork(4,5,3, outclass=SoftmaxLayer)
#y tenemos que configurar el algoritmo de entrenamiento backpropagation
trainer = BackpropTrainer(net,dataset=trndata,momentum=0.1,verbose=True,weightdecay=0.01)
#ahora para entrenar lo que inicializamos utilizamos la siguiente linea
trnerr,valerr = trainer.trainUntilConvergence(dataset=trndata,maxEpochs=100)
pl.interactive(False)
pl.plot(trnerr,'b',valerr,'r')
pl.show()
out= net.activateOnDataset(tstdata)
out=np.argmax(out,axis=1)
output = np.array([net.activate(x) for x, _ in validata])
output = output.argmax(axis=1)
eror=percentError(output,validata['class'])
print 100-eror
print output
print validata['class']
print type (validata)