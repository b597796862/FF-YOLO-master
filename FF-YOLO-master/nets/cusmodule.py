import tensorflow.keras.backend
from tensorflow.keras.layers import (Concatenate, Input, Lambda, UpSampling2D,
                          ZeroPadding2D)
from tensorflow.keras.models import Model

from nets.CSPdarknet import (CSPLayer, DarknetConv2D, DarknetConv2D_BN_SiLU,
                               darknet_body,SiLU)
from nets.yolo_training import get_yolo_loss
from tensorflow.keras.layers import Conv2D,Concatenate,GlobalAvgPool2D,GlobalMaxPool2D,Reshape,Input,Softmax,Permute
import tensorflow.keras as k

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def FEM(x):  #这是分组卷积的版本
    channel_num=x.shape[-1]//4
    x1=Conv2D(filters=channel_num,kernel_size=1,strides=1,padding='same',activation=None)(x[:,:,:,0:channel_num])
    x1=SiLU()(x1)
    x1 = Conv2D(filters=channel_num, kernel_size=3, strides=1, padding='same', activation=None)(x1)
    x1 = SiLU()(x1)

    x2=Conv2D(filters=channel_num,kernel_size=1,strides=1,padding='same',activation=None)(x[:,:,:,channel_num:channel_num*2])
    x2=SiLU()(x2)
    x2=Conv2D(filters=channel_num, kernel_size=[1,3], strides=1, padding='same', activation=None)(x2)
    x2 = SiLU()(x2)
    x2=Conv2D(filters=channel_num, kernel_size=[3,1], strides=1, padding='same', activation=None)(x2)
    x2 = SiLU()(x2)
    x2=Conv2D(filters=channel_num, kernel_size=[3,3], strides=1, padding='same', activation=None)(x2)
    x2 = SiLU()(x2)

    x3 = Conv2D(filters=channel_num, kernel_size=1, strides=1, padding='same', activation=None)(x[:,:,:,channel_num*2:channel_num*3])
    x3 = SiLU()(x3)
    x3 = Conv2D(filters=channel_num, kernel_size=[3, 3], strides=1, padding='same', activation=None)(x3)
    x3 = SiLU()(x3)
    x3 = Conv2D(filters=channel_num, kernel_size=[3, 3], strides=1, padding='same',dilation_rate=2, activation=None)(x3)
    x3 = SiLU()(x3)

    x4 = Conv2D(filters=channel_num, kernel_size=1, strides=1, padding='same', activation=None)(x[:,:,:,channel_num*3:])
    x4 = SiLU()(x4)

    out=Concatenate(axis=-1)([x1,x2,x3,x4])
    return out




def CRC(x1,x2):
    c1=x1.shape[-1]
    c2=x2.shape[-1]
    w1=GlobalAvgPool2D()(x1)
    w2=GlobalAvgPool2D()(x2)
    w=Concatenate(axis=-1)([w1,w2])
    w=k.backend.expand_dims(w,axis=(1))
    w = k.backend.expand_dims(w, axis=(1))
    print(w.shape)
    w=Conv2D(filters=c1+c2,kernel_size=1,activation="sigmoid")(w)
    return Concatenate(axis=-1)([x1,x2])*w


def SCAM(feat,w,h):
    ch_num=feat.shape[-1]

    print("w=",w)
    print("h=",h)
    mp=GlobalMaxPool2D()(feat)
    ap=GlobalAvgPool2D()(feat)
    mp = k.backend.expand_dims(mp, axis=-1)
    ap = k.backend.expand_dims(ap, axis=-1)
    v=Conv2D(filters=ch_num,kernel_size=1)(feat)  #b,w,h,ch_num
    v=Reshape(target_shape=[-1,w*h,ch_num]) (v)     #b,w*h,ch_num

    qk=Conv2D(filters=1,kernel_size=1)(feat)   #b,w,h,1
    qk=Reshape(target_shape=[-1,1,w*h]) (qk)   #b,1,w*h
    qk=Softmax()(qk)
    print(v.shape)
    print(qk.shape)
    vqk=k.backend.batch_dot(v,qk,axes=(2,3))
    vqk=k.backend.squeeze(vqk,axis=(1))      #b,ch_num,1,1
    vqk=Permute((3,2,1))(vqk)     #b,1,1,ch_num
    print(vqk.shape)

    amp=Concatenate(axis=-1)([mp,ap])
    amp=Softmax(axis=-1)(amp)            #b,ch_num,2
    print(amp.shape)

    #amp*v
    ampv=k.backend.batch_dot(amp,v,axes=(-2,-1))
    print(ampv.shape)
    ampv=Reshape(target_shape=(-1,w,h,2))(ampv)
    ampv=k.backend.squeeze(ampv,1)
    print(ampv.shape)

    ampv=Conv2D(filters=1,kernel_size=1)(ampv)   #h,w,1
    print(ampv.shape)

    vqk=Conv2D(filters=ch_num,kernel_size=1)(vqk)  #1,1,c
    print(vqk.shape)
    out=ampv*vqk
    print(out.shape)
    return out+feat

