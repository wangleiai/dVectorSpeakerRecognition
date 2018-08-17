from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, MaxoutDense, GRU
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras import Model
from keras.layers.core import Reshape,Masking,Lambda,Permute
from keras.layers import Input,Dense,Flatten
import keras
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import random

import keras.backend as K
import numpy as np
import librosa
import python_speech_features as psf
import os

# 指定GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

filePath = "/home/user2/untar_data/train"
nClass = len(os.listdir(filePath))


def getwavPathAndwavLabel(filePath):
    wavPath = []
    wavLabel = []
    files = os.listdir(filePath)
    lab = 0
    for file in files:
        wav = os.listdir(filePath+"/"+file)
        for j in range(len(wav)):
            fileType = wav[j].split(".")[1]
            if fileType=="wav":
                wavLabel.append(lab)
                wavPath.append(filePath+"/"+file+"/"+wav[j])
        lab += 1
    return wavPath, wavLabel

wavPath = None
wavLabel = None

def getBW(batchSize=2, second=3, sampleRate=16000):
    """
    :param batchSize: 一个批次大小
    :param second: 音频的长度，默认3.5s,单位为sec
    :param sampleRate: 采样率
    :return:特征矩阵  和 标签
    """
    count = 0
    while True:

        '''按照相同的顺序打乱文件'''
        cc = list(zip(wavPath, wavLabel))
        random.shuffle(cc)
        wavPath[:], wavLabel[:] = zip(*cc)
        x = []
        y = []
        count = 0
        for index, wav in enumerate(wavPath):
            if count == batchSize:
                X = x
                Y = y
                # print(np.array(x).shape)
                X = np.array(X)  # (2, 64, 299, 3)
                Y = np.array(Y)
                Y = keras.utils.to_categorical(y, nClass)
                # print()
                x = []
                y = []
                count = 0

                yield [X, Y]
                # print(X.shape)
                # print(Y.shape)

            else:
                signal, srate = librosa.load(wav, sr=sampleRate)
                # 判断是否超过三秒，
                # 超过三秒则截断
                if len(signal) >= 3 * srate:
                    signal = signal[0:int(3 * srate)]
                # 少于三秒则填充0
                else:
                    signal = signal.tolist()
                    for j in range(3 * srate - len(signal)):
                        signal.append(0)
                    signal = np.array(signal)
                # print(len(signal))

                feat = psf.logfbank(signal, samplerate=16000, nfilt=64)
                feat1 = psf.delta(feat, 1)
                feat2 = psf.delta(feat, 2)
                feat = feat.T[:,:,np.newaxis]
                feat1 = feat1.T[:,:,np.newaxis]
                feat2 = feat2.T[:,:,np.newaxis]

                fBank = np.concatenate((feat,feat1,feat2),axis=2)

                x.append(fBank)
                y.append(wavLabel[index])
                count +=1



def triplet_loss(y_true, y_pred):
    y_pred = K.l2_normalize(y_pred,axis=1)
    batch = batchSize
    #print(batch)
    ref1 = y_pred[0:batch,:]
    pos1 = y_pred[batch:batch+batch,:]
    neg1 = y_pred[batch+batch:3*batch,:]
    dis_pos = K.sum(K.square(ref1 - pos1), axis=1, keepdims=True)
    dis_neg = K.sum(K.square(ref1 - neg1), axis=1, keepdims=True)
    dis_pos = K.sqrt(dis_pos)
    dis_neg = K.sqrt(dis_neg)
    a1 = 17
    d1 = dis_pos + K.maximum(0.0, dis_pos - dis_neg + a1)
    return K.mean(d1)



if __name__ =="__main__":

    wavPath, wavLabel = getwavPathAndwavLabel(filePath)
    print("len wavPath: ", len(wavPath))
    batchSize = 32
    # 卷积核个数
    nFilter = 64
    # 池化层的大小
    poolSize = [2, 2]
    # 池化层步长
    strideSize = [2, 2]
    # 卷积核的大小
    kernelSize = [5, 5]
    model = Sequential()
    model.add(Convolution2D(nFilter, (kernelSize[0], kernelSize[1]),
                            padding='same',
                            strides=(strideSize[0], strideSize[1]),
                            input_shape=(64, 299, 3), name="cov1"))
    model.add(MaxPooling2D(pool_size=(poolSize[0], poolSize[1]), strides=(strideSize[0], strideSize[1]), padding="same", name="pool1"))
    # 将输入的维度按照给定模式进行重排
    model.add(Permute((2,1,3),name='permute'))
    # 该包装器可以把一个层应用到输入的每一个时间步上,GRU需要
    model.add(TimeDistributed(Flatten(),name='timedistrib'))

    # 三层GRU
    model.add(GRU(units=1024, return_sequences=True, name="gru1"))
    model.add(GRU(units=1024, return_sequences=True, name="gru2"))
    model.add(GRU(units=1024, return_sequences=True, name="gru3"))

    # temporal average
    # model.add(Lambda(lambda y: K.mean(y, axis=1), name="temporal_average"))
    def temporalAverage(x):
        return K.mean(x, axis=1)
    model.add(Lambda(temporalAverage, name="temporal_average"))
    # affine
    model.add(Dense(units=512, name="dense1"))

    # length normalization
    # model.add(Lambda(lambda y: K.l2_normalize(y, axis=-1), name="ln"))
    def lengthNormalization(x):
        return K.l2_normalize(x, axis=-1)
    model.add(Lambda(lengthNormalization, name="ln"))

    model.add(Dense(units=nClass ,name="dense2"))
    model.add(Activation("softmax"))

    sgd = Adam(lr=0.00001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    model.fit_generator(getBW(batchSize, sampleRate=16000),steps_per_epoch = int(len(wavPath)/batchSize), epochs=100,
                        callbacks=[
                            # 每次训练保存一次模型
                            ModelCheckpoint("gru1.h5", monitor='loss', verbose=1, save_best_only=False, mode='min'),
                            # 当检测指标不变的时候，学习率lr = lr *0.1x
                            keras.callbacks.ReduceLROnPlateau(monitor='train_loss', factor=0.1, patience=10,
                                                              verbose=0, mode='auto', epsilon=0.0001, cooldown=0,
                                                              min_lr=0)
                        ]
                        )
    # model.save("gru2.h5")
    # layerName = "dense1"
    # targetModel = Model(inputs=model.input, outputs=model.get_layer(layerName).output)
    # targetModel.save("gru3.h5")


#
#     signal,sr  = librosa.load("F:\python\sv\svv\\aishell\dev\S0724\BAC009S0724W0121.wav", sr=16000)
#     if len(signal)>3*16000:
#         signal = signal[0:3*16000]
#
#     x = []
#     feat, eng = psf.fbank(signal=signal, samplerate=16000, nfilt=64)
#     # print(feat.shape)
#     feat = psf.delta(feat, 1)
#     feat = psf.delta(feat, 2)
#     x.append(feat.tolist())
#     x.append(feat.tolist())
#     x.append(feat.tolist())
#     x = np.array(x) # (3, 299, 64)
#     print(x.shape)
#     x = np.reshape(x, (batchSize, x.shape[2], x.shape[1], x.shape[0]))
#     pre = model.predict(x)
#     print(pre.shape)
# # print(pre)





