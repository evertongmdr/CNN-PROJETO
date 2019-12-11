from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Concatenate

def Steam(in_lr,pre):
    """
    :param in_lr: camada de entrada
    :param pre: prefixo que sera adicionado ao nome (pre + 'conv1')
    :return: output
    """
    x = Conv2D(32, (3, 3), strides= 2, padding='valid', activation='relu', name=pre + 'conv1_st2')(in_lr)
    x = Conv2D(32, (3, 3), padding='valid', activation='relu', name=pre + 'conv2')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name=pre + 'conv3')(x)

    t1 = Conv2D(96, (3, 3), strides= 2, padding='valid', activation='relu', name = pre + 't1_conv_st2')(x)
    t2 = MaxPooling2D((3, 3), strides= (2,2), padding='valid', name = pre + 't2_pool_st2')(x)
    x = Concatenate(name=pre + 'c1')([t1,t2])

    t1 = Conv2D(64, (1, 1), padding='same', activation='relu', name=pre + 't1_2_conv1')(x)
    t1 = Conv2D(96, (3, 3), padding='valid', activation='relu', name=pre + 't1_2_conv2')(t1)
    t2 = Conv2D(64, (1, 1), padding='same', activation='relu', name=pre + 't2_2_conv1')(x)
    t2 = Conv2D(64, (7, 1), padding='same', activation='relu', name=pre + 't2_2_conv2')(t2)
    t2 = Conv2D(64, (1, 7), padding='same', activation='relu', name=pre + 't2_2_conv3')(t2)
    t2 = Conv2D(96, (3, 3), padding='valid',activation='relu', name=pre + 't2_2_conv4')(t2)
    x = Concatenate(name=pre + 'c2')([t1, t2])

    t1 = Conv2D(192, (1, 1), padding='valid', activation='relu', name=pre + 't1_3_conv1')(x)
    t1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name=pre + 't1_3_pool_st2')(t1)

    t2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name=pre + 't2_3_pool_st2')(x)
    x = Concatenate(name=pre + 'c3')([t1, t2])

    output = x
    return output

def reduction(in_lr,pre,l,k,m,n):
    """
    :param in_lr:
    :param pre:
    :param l: Numero de Unidades para t3 conv 2
    :param k: Numero de Unidades para t3 conv 1
    :param m: Numero de Unidades para t3 conv 3
    :param n: Numero de Unidades para t2
    :return:
    """
    t1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name=pre + 't1_pool')(in_lr)

    t2 = Conv2D(n, (3, 3), padding='valid', strides=2, activation='relu', name=pre + 't2_conv1')(in_lr)

    t3 = Conv2D(k, (1, 1), padding='same', activation='relu', name=pre + 't3_conv1')(in_lr)
    t3 = Conv2D(l, (3, 3), padding='same', activation='relu', name=pre + 't3_conv2')(t3)
    t3 = Conv2D(m, (3, 3), padding='valid', strides=2, activation='relu', name=pre + 't3_conv3')(t3)

    out = Concatenate(name=pre)([t1,t2,t3])
    return out





def Inception_A(in_lr,pre):

    t1 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', name=pre + 't1_avg')(in_lr)
    t1 = Conv2D(96, (1, 1), padding='same', activation='relu', name=pre + 't1_conv1')(t1)

    t2 = Conv2D(96, (1, 1), padding='same', activation='relu', name=pre + 't2_conv1')(in_lr)

    t3 = Conv2D(64, (1, 1), padding='same', activation='relu', name=pre + 't3_conv1')(in_lr)
    t3 = Conv2D(96, (3, 3), padding='same', activation='relu', name=pre + 't3_conv2')(t3)

    t4 = Conv2D(64, (1, 1), padding='same', activation='relu', name=pre + 't4_conv1')(in_lr)
    t4 = Conv2D(96, (3, 3), padding='same', activation='relu', name=pre + 't4_conv2')(t4)
    t4 = Conv2D(96, (3, 3), padding='same', activation='relu', name=pre + 't4_conv3')(t4)

    out = Concatenate(name=pre)([t1,t2,t3,t4])
    return out


def Inception_B(in_lr,pre):

    t1 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', name=pre + 't1_avg')(in_lr)
    t1 = Conv2D(128, (1, 1), padding='same', activation='relu', name=pre + 't1_conv1')(t1)

    t2 = Conv2D(384, (1, 1), padding='same', activation='relu', name=pre + 't2_conv1')(in_lr)

    t3 = Conv2D(192, (1, 1), padding='same', activation='relu', name=pre + 't3_conv1')(in_lr)
    t3 = Conv2D(224, (1, 7), padding='same', activation='relu', name=pre + 't3_conv2')(t3)
    t3 = Conv2D(256, (1, 7), padding='same', activation='relu', name=pre + 't3_conv3')(t3)

    t4 = Conv2D(192, (1, 1), padding='same', activation='relu', name=pre + 't4_conv1')(in_lr)
    t4 = Conv2D(192, (1, 7), padding='same', activation='relu', name=pre + 't4_conv2')(t4)
    t4 = Conv2D(224, (7, 1), padding='same', activation='relu', name=pre + 't4_conv3')(t4)
    t4 = Conv2D(224, (1, 7), padding='same', activation='relu', name=pre + 't4_conv4')(t4)
    t4 = Conv2D(256, (7, 1), padding='same', activation='relu', name=pre + 't4_conv5')(t4)

    out = Concatenate(name=pre)([t1,t2,t3,t4])
    return out


def Inception_C(in_lr,pre):

    t1 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', name=pre + 't1_avg')(in_lr)
    t1 = Conv2D(256, (1, 1), padding='same', activation='relu', name=pre + 't1_conv1')(t1)

    t2 = Conv2D(256, (1, 1), padding='same', activation='relu', name=pre + 't2_conv1')(in_lr)

    t3 = Conv2D(384, (1, 1), padding='same', activation='relu', name=pre + 't3_conv1')(in_lr)
    t3s1 = Conv2D(256, (1, 3), padding='same', activation='relu', name=pre + 't3s1_conv')(t3)
    t3s2 = Conv2D(256, (3, 1), padding='same', activation='relu', name=pre + 't3s2_conv')(t3)

    t4 = Conv2D(384, (1, 1), padding='same', activation='relu', name=pre + 't4_conv1')(in_lr)
    t4 = Conv2D(448, (1, 3), padding='same', activation='relu', name=pre + 't4_conv2')(t4)
    t4 = Conv2D(512, (3, 1), padding='same', activation='relu', name=pre + 't4_conv3')(t4)
    t4s1 = Conv2D(256, (3, 1), padding='same', activation='relu', name=pre + 't4s1_conv')(t4)
    t4s2 = Conv2D(256, (1, 3), padding='same', activation='relu', name=pre + 't4s2_conv')(t4)

    out = Concatenate(name=pre)([t1,t2,t3s1,t3s2,t4s1,t4s2])
    return out