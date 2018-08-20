# dVectorSpeakerRecognition
基于dVector的说话人识别keras


  
  
数据集：aishell-v http://www.aishelltech.com/kysjcp

dnnTrainKeras实现的是https://research.google.com/pubs/archive/41939.pdf ，并稍加修改，只做参考

实现思路：

  按照https://research.google.com/pubs/archive/41939.pdf 这篇论文里的想法，先按照分类跑模型，然后保留除输出层之外的层。用交叉熵计算损失。我觉得这篇论文把d-vector说的相当清楚，建议细看。我输入的是一个3秒的语句，把这三秒语句得到logfbank特征，维度是(299，26），然后直接展开成（1，299*26），直接扔到网络里训练。这里如果能像论文里一样用帧拓展输入将会更好。
  
gruTrainKeras实现的是https://arxiv.org/pdf/1705.02304.pdf, 并稍加修改,只做参考。

实现思路：
  论文中介绍res-cnn的输入和cov-gru的输入一样，由于自己没看懂怎么输入就找了个博客（https://blog.csdn.net/lauyeed/article/details/79936632）,  我把输入s按照博客中的改了改就可以用了。论文中用的triple loss，我这里还是用的交叉熵loss。建议用triple loss试试，效果应该会更好。还有就是输入也要改为做了帧z拓展之后的输入，不直接把整个三秒仍进去。这个我训练了2，3个epoch就得到了csv文件中的结果了，训练多了反而效果不好。代码训练时用了分类准确率是为了看看多会儿过拟合，这个完全可以不要。

根据实验，dnnTrainKeras和gruTrainkeras代码都有效果


gruTensorflow没有测试过，但是应该可以加入挺多东西的，现在还是个不完善版。
