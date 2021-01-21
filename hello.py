# coding:utf-8

import numpy as np
import pandas as pd
import tools


p = 0.02
train = pd.read_csv("./small_train.csv", skiprows = lambda x: x>0 and np.random.rand() > p)
train.to_csv("very_small.csv")
