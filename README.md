# dVectorSpeakerRecognition
基于dVector的说话人识别keras


  
  
数据集：aishell-v http://www.aishelltech.com/kysjcp

kersaSoftmax/dnnTrain.py实现的是https://research.google.com/pubs/archive/41939.pdf ,只是实验了一下，看到有效果，并没有完整的测试

实现思路：

  按照https://research.google.com/pubs/archive/41939.pdf 这篇论文里的想法，先按照分类跑模型，然后保留除输出层之外的层，最后一个隐藏层作为embding。训练时用交叉熵计算损失。我觉得这篇论文把d-vector说的相当清楚，建议细看。我输入的是一个3秒的语句，把这三秒语句得到logfbank特征，维度是(299，26），然后直接展开成（1，299*26），直接扔到网络里训练。这里如果能像论文里一样用帧拓展输入将会更好。
  
kerasSoftmax/gruTrain实现的是https://arxiv.org/pdf/1705.02304.pdf, 实现了里面第二种结构，先卷积，然后接gru，然后接全连接层。我用aishell-1数据集，做出了不错的效果，而且还没有去静音，应该有提升空间。

实现思路：
  论文中介绍res-cnn的输入和cov-gru的输入一样，由于自己没看懂怎么输入就找了个博客（https://blog.csdn.net/lauyeed/article/details/79936632）,  博客很详细的介绍了输入特征是什么样子的。
  
tfTriplet/ ：使用了triplet loss函数，s只是跑通了代码，因为不太需要高的精确率就没有再尝试。

联系方式：1839522753@qq.com
