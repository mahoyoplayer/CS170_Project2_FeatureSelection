import numpy as np
import math


data = np.loadtxt("data120.txt")


labels, features = np.split(data, [1], axis = 1)
featureCount = len(features)
labels = labels.astype(int)
print(labels)
print(features)

# Features is a set?
def accuracy(data, features, k = 5, bestAccuracy=0.0) -> float:
    def EuclidDist(inst1, inst2):
        total = 0
        for i in features:
            total += (inst1[i] - inst2[i]) ** 2
        return math.sqrt(total)
    correctCount = 0
    instanceCount = 9
    bestLabel, minDistance = None, float("inf")
    if correctCount / instanceCount < bestAccuracy:
        # Alpha beta pruning
        return -1.0

    return correctCount / instanceCount

#print(data.shape)
#print(data[:2])

#print("Hello World!")