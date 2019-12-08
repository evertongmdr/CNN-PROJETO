import sys
from keras.models import Model
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Input,Dropout

w,h,c,qtd = 330,330,3, 28

def info(model,label):
    print("------------------------------------------------------------------------------")
    print(label)
    print("------------------------------------------------------------------------------")
    model.summary(); 

def LeNet(w,h,c,n_classes):
    inputs = Input(shape=[h,w,c],name='Input1')
    x = Conv2D(20,[5,5],strides=[1,1],padding='same',activation='tanh',name='Conv1')(inputs)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling1')(x)

    x = Conv2D(20,[5,5],strides=[1,1],padding='same',activation='tanh',name='Conv2')(x)
    x = MaxPool2D(pool_size=[2,2],strides = [2,2],name='pooling2')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(500,activation='tanh',name='Dense1')(x)
    
    x = Dense(n_classes,activation='softmax',name='Output')(x)

    model = Model(inputs=inputs,outputs=x)

    label = "LeNet"
    info(model,label)
    return(model,label)


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

if __name__ == "__main__":
    if len(sys.argv) == 2:
        m = sys.argv[1]
        model = None
        if m == "LeNet":
            model,label = LeNet(w,h,c,qtd)
        elif m == "vgg16_c":
            model,label = vgg16_c(w,h,c,qtd)
        else:
            print("Não encontrado")

    else:
        print("python Models.py [modelo]\n")
        print("LeNet")
        print("vgg16_c")
