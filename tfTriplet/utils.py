import tensorflow as tf
import numpy as np
import python_speech_features as psf

import random
import keras.backend as K
import librosa
import os
import keras


def selectTriplet(filePath, classNum, perClassNum):
    claPath = []
    label = []
    for i in range(classNum):
        # 随机挑选类别
        temLabel = random.randint(0, len(os.listdir(filePath))-1)
        # 在类别里挑选perClassNum个人
        num = 0
        while(True):
            # print(filePath+"/"+os.listdir(filePath)[temLabel])
            k = random.randint(0, len(os.listdir(filePath+"/"+os.listdir(filePath)[temLabel]))-1)
            a, s = librosa.load(filePath+"/"+os.listdir(filePath)[temLabel]+"/"+os.listdir(filePath+"/"+os.listdir(filePath)[temLabel])[k], sr=16000)
            if len(a) <=3*16000:
                continue

            # print(len(os.listdir(filePath+"/"+os.listdir(filePath)[temLabel])))
            label.append(temLabel)
            claPath.append(filePath+"/"+os.listdir(filePath)[temLabel]+"/"+os.listdir(filePath+"/"+os.listdir(filePath)[temLabel])[k])
            num += 1
            if num==perClassNum:
                break
    # print(label)
    # print(claPath)
    cc = list(zip(claPath, label))
    random.shuffle(cc)
    claPath[:], label[:] = zip(*cc)
    # print(claPath[0:2])
    # print(label)
    return claPath, label

def getBF(clasPath, label):
    x = []
    y = []
    for i in range(len(clasPath)):
        signal, srate = librosa.load(clasPath[i], sr=16000)
        if len(signal)<3*16000:
            continue
        # 归一化
        signal = signal/max(abs(signal))
        # 提取特征
        feat = psf.logfbank(signal[0:16000*3], samplerate=16000, nfilt=64)
        # print("feat: ", feat.shape)
        feat1 = psf.delta(feat, 1)
        feat2 = psf.delta(feat, 2)
        feat = feat.T[:, :, np.newaxis]
        feat1 = feat1.T[:, :, np.newaxis]
        feat2 = feat2.T[:, :, np.newaxis]
        fBank = np.concatenate((feat, feat1, feat2), axis=2)
        x.append(fBank)
        y.append(label[i])
    x = np.array(x)
    y = np.array(y)
    return x, y

def getFeatures(filePath, classNum=5, perClassNum=4):

    classPath, label = selectTriplet(filePath, classNum, perClassNum)
    bFeat, bLabel = getBF(clasPath=classPath, label=label)

    return bFeat, bLabel

def tF(filePath="", classNum=5, perClassNum=4):
    classPath = []
    label = []
    classPath.append("F:\python\sv\svv\\aishell\dev\S0724\BAC009S0724W0121.wav")
    classPath.append("F:\python\sv\svv\\aishell\dev\S0724\BAC009S0724W0122.wav")
    classPath.append("F:\python\sv\svv\\aishell\dev\S0725\BAC009S0725W0121.wav")
    label.append(0)
    label.append(0)
    label.append(1)
    bFeat, bLabel = getBF(clasPath=classPath, label=label)
    print(bLabel)
    return bFeat, bLabel

if __name__ == '__main__':
    a, b = selectTriplet("F:\python\sv\svv\\aishell\\test", 10, 2)
    print(a)
    print(b)