import numpy as np
import pandas as pd
from collections import Counter

class Apriori(object):

    def __init__(self, data, support, confidence):
        self.data = data
        self.support = support
        self.confidence = confidence
        self.itemset = self.get_itemSet()

    def get_itemSet(self):
        itemDict = {}
        unique, count = np.unique(self.data, return_counts=True)
        itemDict = dict(zip(unique, count))
        print(itemDict)
        return itemDict


df = pd.read_csv("data/retail.dat", sep=" ", header=None)
df = df.iloc[:, :-1].values
print(df)
Apriori(df, support=0.2, confidence=0.8).get_itemSet()