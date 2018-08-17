import python_speech_features as psf
import keras
import numpy as np
import librosa
import librosa.display
import os
import os.path
import librosa
from keras.models import load_model
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
sr = 16000
def getEnroll(filePath, enrNum=20,valNum=80):
    """
    返回三个参数，第一个是注册时需要的语音
    第二个是注册人验证的语音
    第三个是陌生人
    :param filePath: 格式：filepath目录下有多个文件，每个文件代表一个人，每个文件里至少有num+1语音,
    :param num: 几个语句当作注册的
    :return: 三个列表
    """

    dirs = os.listdir(filePath)
    nDifferent = len(dirs)
    # print("nDifferent: ", nDifferent)

    enroll = []
    enrollVerity = []

    for i in range(nDifferent):                           # dev/
        '''生成注册,注册人验证,生成陌生人验证'''
        enrollTem = []
        enrollVerityTem = []
        # 注册
        wavFiles = os.listdir(filePath+"/"+dirs[i]) #s0724/
        # print(wavFiles)
        '''注册信息'''
        for k in range(enrNum):
            enrollTem.append((filePath+"/"+dirs[i]+"/"+wavFiles[k], i))
            # print(filePath+"/"+dirs[i]+"/"+wavFiles[i], i)
        '''注册人验证'''
        for k in range(enrNum, valNum+enrNum):
            # print("注册人验证")
            enrollVerityTem.append((filePath + "/" + dirs[i] + "/" + wavFiles[k], i))
            # print(filePath + "/" + dirs[i] + "/" + wavFiles[i], i)

        enroll.append(enrollTem)
        enrollVerity.append(enrollVerityTem)
    return enroll, enrollVerity

def getBL(filePath, batchSize=10):
    feature = []
    label = []
    for f in filePath:
        wavDir = f[0]
        # print(wavDir)
        label.append(f[1])
        signal, srate = librosa.load(wavDir, sr=sr)
        if len(signal) >= 3 * srate:
            signal = signal[0:int(3 * srate)]
        else:
            signal = signal.tolist()
            for j in range(3 * srate - len(signal)):
                signal.append(0)
            signal = np.array(signal)

        logFea = psf.logfbank(signal=signal, samplerate=sr, nfilt=40)
        # print(logFea.shape)
        logFea = np.ceil(logFea)  # 四舍五入
        feature.append(logFea)
    feature = np.array(feature)
    feature = feature.reshape(feature.shape[0], feature.shape[1]*feature.shape[2])
    return feature, label

def getCDS(a, b):
    """
    返回归一化后的余弦距离，得分CDS越接近1越好
    :param a: shape[1,-1]
    :param b: shape[1, -1]
    :return:
    """

    num = float(a.dot(b.T))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cds = num / denom  # 余弦值
    cds = 0.5 + 0.5 * cds  # 归一化
    return cds


def getWavFeat(wavPath):
    signal,rate = librosa.load(wavPath, sr=16000)
    if len(signal) >= 3 * 16000:
        signal = signal[0:int(3 * 16000)]
    # 少于三秒则填充0
    else:
        signal = signal.tolist()
        for j in range(3 * 16000 - len(signal)):
            signal.append(0)
        signal = np.array(signal)
    feat = psf.logfbank(signal, samplerate=16000, nfilt=64)
    feat1 = psf.delta(feat, 1)
    feat2 = psf.delta(feat, 2)
    feat = feat.T[:, :, np.newaxis]
    feat1 = feat1.T[:, :, np.newaxis]
    feat2 = feat2.T[:, :, np.newaxis]
    fBank = np.concatenate((feat, feat1, feat2), axis=2)
    a = []
    a.append(fBank)
    fBank = np.array(a)
    # print(fBank.shape)
    return fBank


def getFeaturesMFCC(weight_path):
    model = load_model(weight_path)
    def meanFeat(wavList):
        features = []
        for path,_ in wavList:
            fbank = getWavFeat(path)
            features.append(model.predict(fbank))
        features = np.array(features)
        mean_feat = np.mean(features,axis=0)
        return mean_feat
    return meanFeat

if __name__=="__main__":
    batchSize = 32
    sampleRate = 16000

    # model = load_model("gru3.h5")
    # model.summary()
    # exit()


    import time

    ''' 注册'''
    enroll, enrollVeriety = getEnroll("aishell/test", 20,100)

    predict = getFeaturesMFCC("gru3.h5")

    start_time = time.time()
    register = [predict(enr) for enr in enroll]
    print(time.time()-start_time)

    start_time = time.time()
    validate = []
    for perPeopleVal in enrollVeriety:
        perSentence = [predict([val])  for val in perPeopleVal ]
        validate.append(perSentence)

    print(time.time()-start_time)


    confusion_matrix=np.zeros((20,20))

    for i,perPeopleFeat in enumerate(validate):
        for perSenFeat in perPeopleFeat:
            predictList = [ getCDS(perSenFeat,reg )for reg in register ]
            predictLabel = np.argmax(predictList)
            confusion_matrix[i][predictLabel]+=1
    np.set_printoptions(precision=2,suppress=True)
    print(confusion_matrix)


    # for reg in register :
    #
    #     pred =[getCDS(reg,val) for val in validate ]
    #     print(pred)
    #         print(round(getCDS(reg,val),1),end=" | ")
    #     print("\n"+"-"*120 )


    # for i in range(len(enroll)):
    #     '''注册'''
    #     feature, label = getBL(enroll[i], batchSize=len(enroll[0]))
    #     print(feature.shape)
    #     pre = model.predict(feature)
    #     # print("注册：",pre)
    #     pre = np.mean(pre, axis=0)
    #     target = np.reshape(pre, (1, 256))
    #
    #     ver = 0
    #     ll = 1000
    #     for kk in range(len(enroll)):
    #         feature, label = getBL(enrollVeriety[kk], batchSize=len(enrollVeriety[kk]))
    #         print(label)
    #         sum = 0
    #         for j in feature:
    #             j = np.reshape(j, (1,-1))
    #             pre = model.predict(j)
    #             # print("预测：",pre)
    #             pre = np.reshape(pre, (1, 256))
    #             scor = getCDS(target, pre)
    #             sum += scor
    #         print("sum:", sum/float(len(feature)))
    #         average = sum/float(len(feature))
    #         if ver < average:
    #             ver = average
    #             ll = label[0]
    #     print("注册人：",i,"  识别人：", ll)




