import sys
from keras.models import Model
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Input,Dropout,Activation

from Libs.InceptionV4Modules import *

w,h,c,qtd = 330,330,3, 28

def info(model,label):
    print("------------------------------------------------------------------------------")
    print(label)
    print("------------------------------------------------------------------------------")
    model.summary(); 

def vgg16_c(w,h,c,n_classes):
    inputs = Input(shape=[h,w,c],name='Input1')

    x = Conv2D(64,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv1')(inputs)
    x = Conv2D(64,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv2')(x)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling1')(x)

    x = Conv2D(64,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv3')(x)
    x = Conv2D(64,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv4')(x)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling2')(x)

    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv5')(x)
    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv6')(x)
    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv7')(x)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling3')(x)

    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv8')(x)
    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv9')(x)
    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv10')(x)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling4')(x)


    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv11')(x)
    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv12')(x)
    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv13')(x)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling5')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(128,activation='tanh',name='Dense1')(x)
    x = Dropout(0.25,name='Dropout1')(x)

    x = Dense(256,activation='tanh',name='Dense2')(x)
    x = Dropout(0.25,name='Dropout2')(x)

    x = Dense(512,activation='tanh',name='Dense3')(x)
    x = Dropout(0.25,name='Dropout3')(x)

    x = Dense(n_classes,activation='softmax',name='Output')(x)

    model = Model(inputs=inputs,outputs=x)

    label = "vgg16_c"
    info(model,label)
    return(model,label)


def rocket(w,h,c,n_classes):
    inputs = Input(shape=[h,w,c],name='Input1')

    x = Conv2D(32,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv1')(inputs)
    x = Conv2D(32,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv2')(x)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling1')(x)

    x = Conv2D(64,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv3')(x)
    x = Conv2D(64,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv4')(x)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling2')(x)

    x = Conv2D(64,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv5')(x)
    x = Conv2D(64,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv6')(x)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling3')(x)

    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv8')(x)
    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv9')(x)

    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling4')(x)

    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv11')(x)
    x = Conv2D(128,[3,3],strides=[1,1],padding='same',activation='relu',name='Conv12')(x)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling5')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(128,activation='tanh',name='Dense1')(x)
    x = Dropout(0.25,name='Dropout1')(x)

    x = Dense(256,activation='tanh',name='Dense2')(x)
    x = Dropout(0.25,name='Dropout2')(x)

    x = Dense(512,activation='tanh',name='Dense3')(x)
    x = Dropout(0.25,name='Dropout3')(x)

    x = Dense(n_classes,activation='softmax',name='Output')(x)

    model = Model(inputs=inputs,outputs=x)

    label = "rocket"
    info(model,label)
    return(model,label)

def inceptionv4(w,h,c,n_classes):
    inputs = Input(shape=[h,w,c],name='Input')

    x = Steam(inputs,'steam_')

    for i in range(0, 4):
        x = Inception_A(x, 'a_{}_'.format(i))

    x = reduction(x,'ra_',224,192,256,384)

    for i in range(0, 7):
        x = Inception_B(x, 'b_{}_'.format(i))

    x = reduction(x,'rb_',224,192,256,384)

    for i in range(0, 3):
        x = Inception_C(x, 'c_ins_c_{}_'.format(i))

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='avg')(x)

    x = Flatten(name = 'Flatten_Layer')(x)

    x = Dropout(0.2)(x)
    x = Dense(n_classes,name="c_Dense_output")(x)
    outputs = Activation('softmax',name="c_output")(x)

    model = Model(inputs=[inputs],outputs=outputs)


    label = "inceptionv4"
    info(model,label)
    return(model,label)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        m = sys.argv[1]
        model = None
        if m == "LeNet":
            model,label = LeNet(w,h,c,qtd)
        elif m == "vgg16_c":
            model,label = vgg16_c(w,h,c,qtd)
        elif m == "rocket":
            model,label = rocket(w,h,c,qtd)
        elif m == "inceptionv4":
            model,label = inceptionv4(w,h,c,qtd)
        else:
            print("Não encontrado")

    else:
        print("python Models.py [modelo]\n")
        print("LeNet")
        print("vgg16_c")
