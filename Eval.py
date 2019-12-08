import numpy as np
import keras
from Dataset import DataSetBacteria
from Models import *
from keras.optimizers import Adam

def LoadEvalDataset():
    ev = DataSetBacteria()
    ev.load('dataset/eval_qtd_classes(28).npz',True)
    x_ev, y_ev = ev.getDataSet()
    x_ev = np.float32(x_ev)
    x_ev = x_ev/255
    y_ev = keras.utils.to_categorical(y_ev)
    
    n_classes = len(ev.getClasses())

    return x_ev,y_ev,n_classes

def evaluate(model,x,y): 

    ev = model.evaluate(x=x,y=y)
    print('loss:\t',ev[0])
    print('acc:\t',ev[1])


if __name__ == "__main__":
    
    x,y,n = LoadEvalDataset()

    model,model_label = LeNet(330,330,3,n)

    opt = Adam(lr=0.0001,decay=0.1e-6)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    try:
        model.load_weights(filepath='{} peso.hdf5'.format(model_label))
        print("Loaded")
    except Exception as ex:    
        print(ex)

    evaluate(model,x,y)

