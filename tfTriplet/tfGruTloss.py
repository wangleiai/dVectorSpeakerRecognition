import tensorflow as tf
import os
import keras.backend as K
from utils import getFeatures, tF
tf.logging.set_verbosity(tf.logging.INFO)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# 指定占用内存大小
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.43  # 占用GPU45%的显存
K.set_session(tf.Session(config=config))

filePath = "/home/user2/untar_data/train"
devFilePath = "/home/user2/untar_data/dev"
# filePath = "F:\python\sv\svv\\aishell/test"
# devFilePath = "F:\python\sv\svv\data\\dev"

margin = 1 # 边界
sampleRate = 16000
classNum = 50
perClassNum = 4
batchSize = classNum*perClassNum
# batchSize = 3

class convGruNet:
    def __init__(self):
        # 卷积核个数
        self.nFilter = 64
        # 池化层的大小
        self.poolSize = [2, 2]
        # 池化层步长
        self.strideSize = [2, 2]
        # 卷积核的大小
        self.kernelSize = [5, 5]

    def net(self, xInput):
        with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.Conv2D(filters=self.nFilter,  kernel_size=self.kernelSize,
                                     strides=self.strideSize, padding="same",
                                     activation=tf.nn.relu,# kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001)
                                     )(xInput)
        # 重排矩阵
        permute = tf.keras.backend.permute_dimensions(conv1, (0, 2, 1, 3))
        permute = tf.reshape(tf.convert_to_tensor(permute), shape=(batchSize, -1, 2048))
        # # gru
        # gru1 = tf.keras.layers.GRU(units=1024, name="gru1", return_sequences=True)(permute)
        # gru2 = tf.keras.layers.GRU(units=1024, name="gru2", return_sequences=True)(gru1)
        # gru3 = tf.keras.layers.GRU(units=1024, name="gru3", return_sequences=True)(gru2)

        # GRU
        with tf.variable_scope('gru1', reuse=tf.AUTO_REUSE):
            gruCell = tf.contrib.rnn.GRUCell(num_units=1024, name="gru1")
            initState = gruCell.zero_state(batchSize, dtype=tf.float32)
            # 'state' is a tensor of shape [batch_size, cell_state_size]
            outputs1, state = tf.nn.dynamic_rnn(gruCell, permute,
                                               initial_state=initState,
                                               dtype=tf.float32)

        with tf.variable_scope('gru2', reuse=tf.AUTO_REUSE):
            gruCell = tf.contrib.rnn.GRUCell(num_units=1024, name="gru2")
            initState = gruCell.zero_state(batchSize, dtype=tf.float32)
            # 'state' is a tensor of shape [batch_size, cell_state_size]
            outputs2, state = tf.nn.dynamic_rnn(gruCell, outputs1,
                                               initial_state=initState,
                                               dtype=tf.float32)

        with tf.variable_scope('gru3', reuse=tf.AUTO_REUSE):
            gruCell = tf.contrib.rnn.GRUCell(num_units=1024, name="gru3")
            initState = gruCell.zero_state(batchSize, dtype=tf.float32)
            # 'state' is a tensor of shape [batch_size, cell_state_size]
            outputs3, state = tf.nn.dynamic_rnn(gruCell, outputs2,
                                               initial_state=initState,
                                               dtype=tf.float32)

        # 求均值
        aver = tf.reduce_mean(outputs3, axis=1)
        # 全连接层
        fla = tf.layers.Flatten()(aver) # 先展开
        fc1 = tf.layers.Dense(units=512, activation=tf.nn.relu, name="fc1")(fla)

        # ln
        ln = tf.nn.l2_normalize(fc1,name="ln")
        return ln

    def saver(self):
        return tf.train.Saver()

def train(epoch):
    # tf.logging.set_verbosity(tf.logging.INFO)
    trainAnchorData = tf.placeholder(tf.float32, shape=(batchSize, 64, None, 3), name="anchor")
    trainAnchorLabels = tf.placeholder(tf.int32, shape=(batchSize), name="ancLabel")

    model = convGruNet()
    # a = model.net(xInput=xInput)
    ancOut = model.net(trainAnchorData)


    # loss, pos, neg = computeTripletLoss(anchor_feature=ancOut, positive_feature=posOut, negative_feature=negOut, margin=margin)
    # loss = batch_hard_triplet_loss(labels, emb, margin, squared=False)
    loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=trainAnchorLabels, embeddings=ancOut)
    # loss = batch_hard_triplet_loss(labels=trainAnchorLabels, embeddings=ancOut, margin=margin)
    globalStep = tf.Variable(tf.constant(0))
    # lr = tf.train.exponential_decay(
    #     tf.convert_to_tensor(0.0005), # Learning rate
    #     globalStep,
    #     3000,# 1500steps后衰减
    #     0.95 # Decay step
    # )
    lr = tf.placeholder(dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=globalStep)
    saver = model.saver()
    with tf.Session() as sess:
        trainWriter = tf.summary.FileWriter('./logsTriplet/train',
                                              sess.graph)

        devWriter = tf.summary.FileWriter('./logsTriplet/test',
                                             sess.graph)

        tf.summary.scalar('loss', loss)
        # tf.summary.scalar('positives', pos)
        # tf.summary.scalar('negatives', neg)
        tf.summary.scalar('lr', lr)
        merged = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())

        for e in range(epoch):
            ''' 训练 '''
            count = 0
            for i in range(100):
                count += 1
                globalStep.assign(globalStep+1)
                ancBatch, ancBatchLabel = getFeatures(filePath, classNum=classNum, perClassNum=perClassNum)
                # ancBatch, ancBatchLabel = tF(filePath, classNum=classNum, perClassNum=perClassNum)

                feedDict = {
                    trainAnchorData:ancBatch,
                    trainAnchorLabels:ancBatchLabel,
                    lr:0.001
                }
                _, losses, learningrate, summary = sess.run([optimizer, loss, lr, merged], feed_dict=feedDict)
                trainWriter.add_summary(summary, sess.run(globalStep))
                print("trainLoss: ", losses)
            ''' 验证 '''
            count = 0
            for i in range(5):
                count += 1
                ancBatchDev, ancBatchLabelDev = getFeatures(devFilePath, classNum=classNum, perClassNum=perClassNum)
                # ancBatchDev, ancBatchLabelDev = tF(filePath, classNum=classNum, perClassNum=perClassNum)
                feedDict = {
                    trainAnchorData: ancBatchDev,
                    trainAnchorLabels: ancBatchLabelDev,
                    lr: 0.001

                }
                losses, summary = sess.run([loss,  merged], feed_dict=feedDict)
                devWriter.add_summary(summary, sess.run(globalStep))
                print("testLoss: ", losses)


            saver.save(sess, "triplet/modelCheckpoint", global_step=sess.run(globalStep))
        trainWriter.close()
        devWriter.close()

train(1000)
