
import numpy as np;
from common.layers import *;

class bpNet:
    def __init__ (
        self,
        inputSize = 1,
        hiddenLayersSize = [],
        outputSize = 1,
        weightInitStd = 0.01,
    ):
        self.inputSize = inputSize;
        self.outputSize = outputSize;
        self.hiddenLayersSize = hiddenLayersSize;
        self.weightInitStd = weightInitStd;
        self.params = None;
        self.initParams();
        self.hiddenLayers = None;
        self.initHiddenLayers();
        self.lastLayer = None;
        self.initOutputLayer();

    # 使用神经网络进行预测
    # 此方法没有调用输出层，即没有调用softmax以及损失函数
    def predict (self, x):
        y = x.copy();
        for layer in self.hiddenLayers:
            y = layer.forward(y);
        return y;

    # 根据输入数据以及监督数据计算损失函数
    # 同时也会对神经网络进行一次整体数据流动
    def loss (self, x):
        y = self.predict(x);
        return self.lastLayer.forward(y);

    # 构建输出层
    def initOutputLayer (self):
        self.lastLayer = radiusSquare();
    # 根据初始化的参数顺序构建隐藏层（包含输入层）
    def initHiddenLayers (self):
        self.hiddenLayers = [];
        for index, value in enumerate(self.params):
            weight = value["weight"];
            bias = value["bias"];
            layer = None;
            if (index == len(self.params) - 1):
                layer = affine(weight, bias);
            else:
                layer = affineReLu(weight, bias);
            self.hiddenLayers.append(layer);
    # 顺序初始化各层参数
    def initParams (self):
        self.params = [];
        layerSizeList = [ self.inputSize ] + self.hiddenLayersSize + [ self.outputSize ];
        for index, value in enumerate(layerSizeList):
            if (index > 0):
                prevSize = layerSizeList[index - 1];
                curSize = value;
                param = self.layerParam(prevSize, curSize);
                self.params.append(param);
    # 生成层参数，包括初始权重和初始偏置
    def layerParam (self, inputSize, outputSize):
        param = { };
        # 利用高斯分布初始化权重矩阵，这里乘以了weightInitStd
        param["weight"] = self.weightInitStd * np.random.randn(inputSize, outputSize);
        param["bias"] = np.zeros(outputSize);
        return param;
    