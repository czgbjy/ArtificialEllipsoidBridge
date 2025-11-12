class ComplementCell:
    def __init__(self, ID):
        """
        Initialize a neuron with a given ID.
        """
        self.ID = ID  # 神经元的ID
        self.synapsesAll = {}  # 存储兴奋性突触的字典，这是其在生成网络时，所有的兴奋性连接
        self.activated = False

    def forward(self, input):
        """
        前向传播,输入还是所有的输入
        """
        totalInput = 0.0  # 这个用来标记
        for cell, weight in self.synapsesAll.items():  # 处理兴奋性突触
            index = cell.getID()
            inputValue = input[index]
            totalInput += inputValue * weight  # 神经元的输入都是1，值为权重
        return totalInput  # 返回总输入

    def predict(self, input, activatedThreshold):
        """
        预测
        """
        return 1 if self.forward(input) >= activatedThreshold else 0  # 阈值1

    def getID(self):
        return self.ID

    def setID(self, ID):
        self.ID = ID

    def getSynapsesAll(self):
        return self.synapsesAll

    def setSynapsesAll(self, synapsesAll):
        self.synapsesAll = synapsesAll