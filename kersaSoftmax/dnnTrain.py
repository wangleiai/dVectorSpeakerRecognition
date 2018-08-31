import python_speech_features as psf
import tensorflow as tf
import keras
import numpy as np
import librosa
import librosa.display
import os
import os.path

filePath = "aishell/train"
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
        x = []
        y = []
        for index, wav in enumerate(wavPath):
            count +=1
            if count-1==batchSize:
                count = 0
                x = np.array(x)
                y = np.array(y)
                # x = toMfcc(x, sampleRate=sampleRate)
                x = toLogFillterBank(x, sampleRate=sampleRate)
                # print(x.shape," ", x.shape[0], " ", x.shape[1], " ", x.shape[2])
                if x.shape!=(batchSize, x.shape[1], x.shape[2]):
                    x = []
                    y = []
                    count = 0
                else:
                    X = x.reshape(batchSize, -1)
                    # print(x.shape)
                    Y = keras.utils.to_categorical(y, nClass)
                    x = []
                    y = []
                    yield [X, Y]

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
                x.append(signal)
                y.append(wavLabel[index])


def toMfcc(x, sampleRate=16000):
    """
    将signal转变为mfcc
    :param x: signal [batchSize, ]
    :param sampleRate: 采样率 默认 16000
    :return: mfcc矩阵
    """
    mfcc = []
    for i in range(len(x)):
        mfcc0 = psf.mfcc(x[i], samplerate=sampleRate)
        mfcc1 = psf.delta(mfcc0, 1)
        mfcc2 = psf.delta(mfcc0, 2)
        mfccAll = np.hstack((mfcc0, mfcc1)) # 参数是tuple
        mfccAll = np.hstack((mfccAll, mfcc2))
        mfcc.append(mfccAll.tolist())
    mfcc = np.array(mfcc)
    return mfcc

def toLogFillterBank(x, sampleRate=16000):
    """
    将信号变为log fillterbank特征
    :param x: signal
    :param sampleRate: 默认16000
    :return: 特征值
    """
    fillterBank = []
    for i in range(len(x)):
        fb = psf.logfbank(x[i], samplerate=sampleRate)
        fillterBank.append(fb.tolist())
    fillterBank = np.array(fillterBank)
    return fillterBank


def toFillterBank(x, sampleRate=16000):
    """
    将信号变为 fillterbank特征和能量值
    :param x: signal
    :param sampleRate: 默认16000
    :return: 特征值和能量值
    """
    fillterBank = []
    energy = []
    for i in range(len(x)):
        fb, ey = psf.fbank(x[i], samplerate=sampleRate)
        energy.append(ey.tolist())
        fillterBank.append(fb.tolist())
    fillterBank = np.array(fillterBank)
    energy = np.array(energy)
    return fillterBank, energy


# if __name__ =="__main__":
#    wavPath, wavLabel = getwavPathAndwavLabel(filePath)
#     a = getBW()


if __name__=="__main__":
    batchSize = 1
    sampleRate = 16000
    wavPath, wavLabel = getwavPathAndwavLabel(filePath)

    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.optimizers import SGD, Adam
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint

    model = Sequential()

    model.add(Dense(256, input_shape=(7774, ), name="dense1"))
    model.add(Activation('relu', name="activation1"))
    model.add(Dropout(0.5, name="drop1"))

    model.add(Dense(256, name="dense2"))
    model.add(Activation('relu', name="activation2"))
    model.add(Dropout(0.5, name="drop2"))

    model.add(Dense(256, name="dense3"))
    model.add(Activation('relu', name="activation3"))

    model.add(Dense(nClass, name="dense4"))
    model.add(Activation('softmax'))

    sgd = Adam(lr=0.000001)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=sgd,  metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,  metrics=['accuracy'])

    model.fit_generator(getBW(batchSize, sampleRate=sampleRate),steps_per_epoch = int(len(wavPath)/batchSize), epochs=100,
                        callbacks=[ModelCheckpoint("dnn5.h5", monitor='val_acc', verbose=1, save_best_only=False, mode='max')]
                        )

    layerName = "dense4"
    targetModel = Model(inputs=model.input, outputs=model.get_layer(layerName).output)
    targetModel.save("dnn6.h5")