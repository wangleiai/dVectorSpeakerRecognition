import tensorflow as tf
import numpy as np
import python_speech_features as psf
import librosa
import os
import random
import random
import keras.backend as K
import librosa
import os
import keras


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

filePath = "../data/aishell/train"
wavPath, wavLabel = getwavPathAndwavLabel(filePath)
nClass = len(os.listdir(filePath))

batchSize = 32

# 卷积核个数
nFilter = 64
# 池化层的大小
poolSize = [1, 2, 2, 1]
# 池化层步长
strideSize = [1, 2, 2, 1]
# 卷积核的大小
kernelSize = [5, 5, 3, 64]

xInput = tf.placeholder(dtype=tf.float32, shape=(batchSize, 64, 299, 3))
y = tf.placeholder(dtype=tf.float32, shape=(batchSize, nClass))
# gruInput = tf.placeholder(dtype=tf.float32, shape=(batchSize, None, 1024))
lr = 0.0001

# 卷积层
conv1 = tf.nn.conv2d(xInput,
                     filter=tf.Variable(tf.truncated_normal([kernelSize[0], kernelSize[1], kernelSize[2], kernelSize[3]], stddev=0.5)),
                     strides=strideSize, padding="SAME")
pool1 = tf.nn.max_pool(conv1, ksize=poolSize, strides=strideSize, padding="SAME" )
permute = tf.keras.backend.permute_dimensions(pool1, (0, 2, 1, 3))
permute = tf.reshape(permute, shape=(permute.shape[0], permute.shape[1], permute.shape[2]*permute.shape[3]))
# GRU
gruCell = tf.contrib.rnn.GRUCell(num_units=1024, name="gru1")
initState = gruCell.zero_state(batchSize, dtype=tf.float32)
# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs1, state = tf.nn.dynamic_rnn(gruCell, permute,
                                   initial_state=initState,
                                   dtype=tf.float32)

gruCell = tf.contrib.rnn.GRUCell(num_units=1024, name="gru2")
initState = gruCell.zero_state(batchSize, dtype=tf.float32)
# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs2, state = tf.nn.dynamic_rnn(gruCell, outputs1,
                                   initial_state=initState,
                                   dtype=tf.float32)

gruCell = tf.contrib.rnn.GRUCell(num_units=1024, name="gru3")
initState = gruCell.zero_state(batchSize, dtype=tf.float32)
# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs3, state = tf.nn.dynamic_rnn(gruCell, outputs2,
                                   initial_state=initState,
                                   dtype=tf.float32)

# 求均值
aver = tf.reduce_mean(outputs3, axis=1)

# affine 是一个全连接层
''' 这里是否需要加激活函数'''
fc1 = tf.contrib.layers.fully_connected(inputs=aver, num_outputs=512, activation_fn=tf.nn.relu)

# # L2正则化   ??????????
# ln = tf.contrib.layers.l2_regularizer(0.001)(fc1)

# 输出层 全连接层
fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=nClass, activation_fn=None)

# 损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=fc2))

# 算准确率
correctPrediction = tf.equal(tf.argmax(y,1),tf.argmax(fc2,1))
# 训练
trainStep = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练多少次
    for epocn in range(100):
        count = 0
        for i, j in getBW(batchSize, sampleRate=16000):
            print(i.shape)
            loss_, _ = sess.run([loss, trainStep], feed_dict={
                xInput:i, y:j,
               })

        # sess.run(correctPrediction, feed_dict={})

