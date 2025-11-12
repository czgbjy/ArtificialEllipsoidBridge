from ComplementCell import ComplementCell
import numpy as np


class ComplementNetwork:
    """
    取补变换网络
    """

    def __init__(self, inputDims):
        """
        Initialize the network with the given input dimensions.
        """
        self.neuronCount = inputDims
        self.cells_outCollection = [None] * inputDims  # 输出层的神经元
        self.cells_hiddenCollection = [None] * (inputDims + 1)  # 隐藏层比输入层多一个神经元
        self.cells_InputCollection = [None] * inputDims  # 输入神经元
        self.synapticWeight = 1.0  # 突触权重

        # 为神经网络创建神经元，输入层和输出层都是neuronCount
        for i in range(inputDims):
            self.cells_outCollection[i] = ComplementCell(i)
            self.cells_InputCollection[i] = ComplementCell(i)

        # 隐藏层是neuronCount+1
        for i in range(len(self.cells_hiddenCollection)):
            self.cells_hiddenCollection[i] = ComplementCell(i)

        self.Connect_with_fixed_weights(self.synapticWeight)

    def Connect_with_fixed_weights(self, initialWeight):
        """
        Establish connections between layers with fixed weights.
        """
        # 建立输出层和隐藏层的连接
        for outputCell in self.cells_outCollection:
            i = outputCell.getID()
            hiddenNeuronCount = len(self.cells_hiddenCollection)
            # 反转的连接方式，即第一个输出神经元和倒数第二个隐藏层相连接，第二个和倒数第三个相连接
            outputCell.getSynapsesAll()[self.cells_hiddenCollection[hiddenNeuronCount - 2 - i]] = initialWeight

        # 对于每个隐藏层的神经元
        for hiddenComplementCell in self.cells_hiddenCollection:
            i = hiddenComplementCell.getID()  # 隐藏层的神经元
            for j in range(i):
                hiddenComplementCell.getSynapsesAll()[self.cells_InputCollection[j]] = initialWeight

    def forward(self, inputs):
        """
        Process the input through the network to produce output.
        """
        netActivationForHidden = [0.0] * len(self.cells_hiddenCollection)  # 记录隐藏层神经元的净激活值
        # 对于每个隐藏层的神经元
        for hiddenComplementCell in self.cells_hiddenCollection:
            index = hiddenComplementCell.getID()
            netActivationForHidden[index] = hiddenComplementCell.forward(inputs)  # 把所有的净激活值都记录下来

        # 这里需要把最大值提取出来
        maxNetActivationForHidden = max(netActivationForHidden)
        hiddenLayerOutput = [0] * len(self.cells_hiddenCollection)  # 隐藏层的输出，作为反转层的输入
        for i in range(len(hiddenLayerOutput)):
            if netActivationForHidden[i] >= maxNetActivationForHidden:
                hiddenLayerOutput[i] = 1
            else:
                hiddenLayerOutput[i] = 0

        complementCellOutput = [0] * len(self.cells_outCollection)
        for outputCell in self.cells_outCollection:
            outIndex = outputCell.getID()
            complementCellOutput[outIndex] = outputCell.predict(hiddenLayerOutput, 1.0)  # 计算最终的输出，由于其是反转函数，因而其阈值就是1.

        return complementCellOutput