#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np;
from bpNet.bpNet import bpNet;

def getPoints (r, x, y, z, size):
    thet = np.random.rand(size) * np.pi * 2;
    fai = np.random.rand(size) * np.pi;
    x = x + r * np.sin(thet) * np.cos(fai);
    y = y + r * np.sin(thet) * np.sin(fai);
    z = z + r * np.cos(thet);
    result = np.array([x, y, z]);
    result = result.transpose((1, 0));
    return result;
inputs = getPoints(88, 34, 111, 51, 100);

# 新建一个网络，输入为三维坐标，输出为拟合的球体方程（r, x, y, z）
network = bpNet(3, [32, 16], 4);

result = network.predict(inputs);
print(result);
