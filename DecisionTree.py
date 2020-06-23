import pandas as pd
import numpy as np
import math

class Node:
    def __init__(self, col_of_split=None):
        self.col_of_split = col_of_split
        self.children = dict()
        self.target = None

class DecisionTree:
    def __init__(self, data, col_target):
        self.data = data
        self.target = col_target
        self.root = Node()
        self.dataLength = len(data)
        self.entropy = self.targetEntropy()

    def targetEntropy(self):
        entropy = 0
        for key, value in self.data.groupby(self.target).groups.items():
            val = len(value)/self.dataLength
            entropy -= val*math.log2(val)

        return entropy

    def train(self):
        for col in data:
            if col != self.target:
                for key, value in self.data.groupby([col, self.target]).groups.items:
                    pass

data = pd.read_csv('cardata.txt', sep=',', header=None)
# print(data[5])
root = DecisionTree(data, 6)
root.train()