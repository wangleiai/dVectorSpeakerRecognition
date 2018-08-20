import os
import numpy as np
import librosa
import tensorflow as tf
import keras
from keras.models import load_model
import python_speech_features as psf
import keras.backend as K

"""
    这部分分为四部分:
    第一部分：分割注册人和陌生人
    第二部分：注册人注册，并保留d-vector
    第三部分：注册人验证, 并做记录
    第四部分：陌生人验证, 并做记录
"""

enrollFile = "aishell/test" #做注册人
foreignFile = "aishell/dev" # 做陌生人


# enrollFile = "/home/user2/untar_data/test" #做注册人
# foreignFile = "/home/user2/untar_data/dev" # 做陌生人
#
# # 指定GPU
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# # 指定占用内存大小
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.45  # 占用GPU45%的显存
# K.set_session(tf.Session(config=config))


enrollList = [] # 注册人列表
enrollListLabel = [] # 注册人列表标签

enrollTestList = [] # 注册人测试列表
enrollTestListLabel = [] # 注册人测试列表标签

foreignList = [] # 陌生人列表
foreignListLabel = [] # 陌生人列表标签

nEL =len(os.listdir(enrollFile)) # 注册人有多少人 20
nELT = len(os.listdir(enrollFile)) # 注册人验证多少人 20
nFL = len(os.listdir(foreignFile)) # 陌生人多少人 40

'''第一部分开始'''
def getAllEnroll(enrollFile, enrollNumber=20, enrollTest=20):
    """

    :param enrollFile:  type string 文件目录,第一层目录下，每一个文件都是一个人，文件里是那个人说的话
    :param enrollNumber: type int 注册需要多少条语音
    :param enrollTest:  type int 注册人验证需要多少条语音
    :return:
    """
    print("getAllEnroll: ", len(os.listdir(enrollFile)))
    enrollList = []  # 注册人列表
    enrollListLabel = []  # 注册人列表标签

    enrollTestList = []  # 注册人测试列表
    enrollTestListLabel = []  # 注册人测试列表标签

    for labelIndex, file in enumerate(os.listdir(enrollFile)): #    test/
        # print(index," ", file) # 0   S0764  1   S0765
        p = []
        la = []
        p2 = []
        la2 = []
        for j, path in enumerate(os.listdir(enrollFile+"/"+file)):
            # print(j, " ", path) # 362   BAC009S0915W0495.wav
            if j<enrollNumber:
                p.append(enrollFile+"/"+file+"/"+path)
                la.append(labelIndex)
            elif j>=enrollNumber and j<enrollTest+enrollNumber:
                p2.append(enrollFile+"/"+file+"/"+path)
                la2.append(labelIndex)
        enrollList.append(p)
        enrollListLabel.append(la)
        enrollTestList.append(p2)
        enrollTestListLabel.append(la2)
    return enrollList, enrollListLabel, enrollTestList, enrollTestListLabel
# print(len(enrollList))
# print(len(enrollListLabel))
# print(len(enrollTestList))
# print(len(enrollTestListLabel))
# enrollList, enrollListLabel, enrollTestList, enrollTestListLabel = getAllEnroll(enrollFile)

def getForeign(foreignFile, foreignNum=20):
    """

    :param foreignFile: type string 文件目录,第一层目录下，每一个文件都是一个人，文件里是那个人说的话
    :param foreignNum: type int 一个陌生人需要多少条语句
    :return:
    """
    foreignList = []  # 陌生人列表
    foreignListLabel = []  # 陌生人列表标签
    print("getForeign: ", len(os.listdir(foreignFile)))
    for labelIndex, file in enumerate(os.listdir(foreignFile)):  # dev/
        # print(labelIndex)
        p = []
        la = []
        for j, path in enumerate(os.listdir(foreignFile+"/"+file)):  # S0724/
            if j==foreignNum:
                break
            p.append(foreignFile + "/" + file + "/" + path)
            la.append(labelIndex+1000)
        foreignList.append(p)
        foreignListLabel.append(la)
    return foreignList, foreignListLabel
# print(len(foreignListLabel))
# print(len(foreignList))
# print(foreignListLabel[0])
# print(foreignList[0])
# foreignList, foreignListLabel = getForeign(foreignFile)
'''第一部分结束'''


def getCDS(a, b):
    """
    返回归一化后的余弦距离，得分CDS越接近1越好
    :param a: shape[n,m]
    :param b: shape[k, m]
    :return: shape[n, k]
    """

    num = a.dot(b.T)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cds = num / denom  # 余弦值
    cds = 0.5 + 0.5 * cds  # 归一化
    return cds

def getWavFeat(wavPath):
    """

    :param wavPath: 音频文件地址
    :return: shape (1, 64, 299, 3)
    """
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

'''第二部分开始----注册人注册，并保留d-vector'''
def getEnrollDVector(enrollList, model):
    """

    :param enrollList:
    :param model: 训练的模型
    :return: a narray
    """
    dVector = []
    for i in range(len(enrollList)):
        dv = []
        for j, path in enumerate(enrollList[i]):
            fBank = getWavFeat(path) # shape (1, 64, 299, 3)
            pre = model.predict(fBank) # shape (1, 512)
            dv.append(pre[0])
            # break
        dvec =np.array(dv)#  shape (20, 512)
        aver = np.average(dvec, axis=0) #  shape (1, 512)
        dVector.append(aver) #
    dVector = np.array(dVector)
    return dVector
# dVector = getEnrollDVector(enrollList, model)
# print("dVector.shape", dVector.shape)
# print("dVecdtor: ", dVector[0])
# print("dVector[0].shape", dVector[0].shape)
# print(np.array(dVector).reshape(20, 512).shape)
'''第二部分结束'''

'''第三部分开始------注册人验证, 并做记录'''
def getEnrollTestResult(enrollTestList, dVector, model):
    """

    :param enrollTestList:
    :param model: 模型
    :return:
    """
    # enrollRecord = []  # 记录对了多少个
    '''注册人对自己的结果'''
    enrollRecordAll = []  # 记录每个对自己的的得分
    # for i in range(len(enrollTestList)):
    #     enrollRecord.append(0)
    print("len(enrollTestList):", len(enrollTestList))
    for i in range(len(enrollTestList)):
        print(i)
        dv = []
        for j, path in enumerate(enrollTestList[i]):
            print(path)
            fBank = getWavFeat(path) # shape (1, 64, 299, 3)
            print(fBank.shape)
            pre = model.predict(fBank) # shape (1, 512)
            print(pre.shape)
            score = getCDS(pre, dVector[i])
            print(score)
            dv.append(score)
            # break
        enrollRecordAll.append(dv)
    print("getEnrollTestResult  finished")

    '''注册人对其他注册人结果'''
    enrollToEnroll = []  # 记录每个对已注册的的得分
    for d in range(dVector.shape[0]):  # 把每个注册人的dVector拿出来, dVector.shape[0]表示有多少个注册人
        dv = []
        print("d in dV: ", d)
        for i in range(len(enrollTestList)):
            print(i)
            if i!=d:
                for j, path in enumerate(enrollTestList[i]):  # 把每个人的语句拿出来算分
                    # print(path)
                    fBank = getWavFeat(path)  # shape (1, 64, 299, 3)
                    print(fBank.shape)
                    pre = model.predict(fBank)  # shape (1, 512)
                    print(pre.shape)
                    score = getCDS(pre, dVector[d])
                    print(score)
                    dv.append(score)
        enrollToEnroll.append(dv)  # 把所有陌生人对一个注册人的所有得分加入到foreignRecordAll里，直到所有人都添加完毕
    return enrollRecordAll, enrollToEnroll
# enrollRecordAll = getEnrollTestResult(enrollTestList, dVector, model)
'''第三部分结束'''

'''第四部分开始------------陌生人验证, 并做记录'''
def getForeignTestResult(foreignList, dVector, model):
    """
    得到每个陌生人对已注册人的得分
    :param foreignList: type  list
    :param dVector:
    :param model: 模型

    :return:
    """
    foreignRecordAll = []  # 记录每个对已注册的的得分
    print("len(foreignList):", len(foreignList))
    for d in range(dVector.shape[0]): # 把每个注册人的dVector拿出来, dVector.shape[0]表示有多少个注册人
        dv = []
        print("d: ", d)
        for i in range(len(foreignList)): # len(foreignList)代表有多少个陌生人
            # print(i)
            for j, path in enumerate(foreignList[i]): # 把每个人的语句拿出来算分
                # print(path)
                fBank = getWavFeat(path) # shape (1, 64, 299, 3)
                # print(fBank.shape)
                pre = model.predict(fBank) # shape (1, 512)
                # print(pre.shape)
                score = getCDS(pre, dVector[d])
                # print(score)
                dv.append(score)
        foreignRecordAll.append(dv)  # 把所有陌生人对一个注册人的所有得分加入到foreignRecordAll里，直到所有人都添加完毕
    print("foreignList  finished ")
    return foreignRecordAll

    # for i in range(len(foreignList)): # len(foreignList)代表有多少个陌生人
    #     dv = []
    #     for j, path in enumerate(foreignList[i]): #
    #         fBank = getWavFeat(path)  # shape (1, 64, 299, 3)
    #         pre = model.predict(fBank)  # shape (1, 512)
    #         score = getCDS(pre, dVector[i])
    #         # print(score)
    #         dv.append(score)
    #         # break
    #         foreignRecordAll.append(dv)
    # return foreignRecordAll
# foreignRecordAll =  getForeignTestResult(foreignList, dVector, model)
'''第四部分结束'''


'''把四部分集合在一起'''
def evaluate(modelPath, enrollRecordFilename="enrollScore.csv", foreignRecordFilename="foreignScore.csv"):
    """
    调用这部分作评估
    :param enrollRecordFilename:
    :param foreignRecordFilename:
    :return:
    """
    # 加载模型
    print("加载模型")
    model = load_model(modelPath)

    print("得到列表")
    enrollList, enrollListLabel, enrollTestList, enrollTestListLabel = getAllEnroll(enrollFile)
    foreignList, foreignListLabel = getForeign(foreignFile)
    print(enrollList)
    print(foreignList)

    print("得到dVector")
    dVector = getEnrollDVector(enrollList, model)
    # print("dVector.shape", dVector.shape)  #(20, 512)
    # print("dVecdtor: ", dVector[0])
    # print("dVector[0].shape", dVector[0].shape) #(512,)

    print("得到结果")
    enrollRecordAll,enrollRecordInner = getEnrollTestResult(enrollTestList, dVector, model)
    foreignRecordAll =  getForeignTestResult(foreignList, dVector, model)


    import csv
    print("写入csv1")
    f = open("csvData"+"/"+enrollRecordFilename, "a", newline="")
    c = csv.writer(f)
    for i in range(len(enrollRecordAll)):
        c.writerow(enrollRecordAll[i])
    # c.writerow()
    f.close()

    print("写入csv2")
    f = open("csvData"+"/test"+enrollRecordFilename, "a", newline="")
    c = csv.writer(f)
    for i in range(len(enrollToEnroll)):
        c.writerow(enrollRecordAll[i])
    # c.writerow()
    f.close()

    print("写入csv3")
    f = open("csvData"+"/"+foreignRecordFilename, "a", newline="")
    c = csv.writer(f)
    for i in range(len(foreignRecordAll)):
        c.writerow(foreignRecordAll[i])
    # c.writerow()
    f.close()
    return enrollRecordAll, foreignRecordAll

a, b = evaluate("gru2.h5")


# import csv
# f = open("data.csv", "a", newline='')
# c = csv.writer(f)
# list = [1, 2]
# list1 = [3 , 3]
#
# c.writerow(list)
# c.writerow(list1)
# f.close()

