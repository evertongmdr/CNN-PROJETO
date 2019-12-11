import keras
import numpy as np
from keras.models import Model
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Input,Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from Dataset import DataSetBacteria
from matplotlib import pyplot as plt
from keras.callbacks import TensorBoard
from Eval import evaluate,LoadEvalDataset
import time,os

from Models import *
import time

w = 330
h = 330
c = 3
batch_size = 4
epochs = 100

tini = time.time()

tini_train = time.time()
train = DataSetBacteria()
train.load('dataset/train_qtd_classes(28).npz',True)
x_train,y_train = train.getDataSet()
x_train = np.float32(x_train)
x_train = x_train/255

print(time.time() - tini_train)

tini_test = time.time()
test = DataSetBacteria()
test.load('dataset/test_qtd_classes(28).npz',True)
x_test, y_test = test.getDataSet()
x_test = np.float32(x_test)
x_test = x_test/255
print(time.time() - tini_test)
print(time.time() - tini)


n_classes = len(train.getClasses())

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# modelo
model,model_label = rocket(w,h,c,n_classes)

opt = Adam(lr=0.0001,decay=0.1e-6)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

try:
    model.load_weights(filepath='{} peso.hdf5'.format(model_label))
    print("Loaded")
except Exception as ex:    
    print(ex)

dir_pesos = 'pesos/'
if not os.path.exists(dir_pesos):
    os.mkdir(dir_pesos)

tensorboard = TensorBoard(log_dir="tensorborad//{}".format(time.time()))
checkpoint = ModelCheckpoint(filepath=dir_pesos+'/peso_'+model_label +'.{epoch:02d}-{val_loss:.3f}.hdf5',monitor='val_acc',verbose=1,mode = 'max')

model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test),callbacks=[checkpoint,tensorboard])

x_ev,y_ev,_ = LoadEvalDataset()

evaluate(model,x_ev,y_ev)