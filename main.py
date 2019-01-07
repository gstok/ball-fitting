#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np;
from bpNet.bpNet import bpNet;

network = bpNet(7, [32, 16], 1);

input = np.array([1, 2, 3, 4, 5, 6, 7]);
result = network.predict(input);
print(result);