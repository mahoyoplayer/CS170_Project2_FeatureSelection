import numpy as np
import math


data = np.loadtxt("data120.txt")
instanceCount, featureCount = data.shape

# Features can be a set or list here.
def accuracy(data, features, k = 5, bestAccuracy=0.0) -> float:
    def EuclidDist(inst1, inst2):
        total = 0
        for i in features:
            total += (inst1[i] - inst2[i]) ** 2
        return math.sqrt(total)

    def getFolds():
        # This does not work properly
        foldSize = len(data) // k
        folds = []
        for i in range(k):
            start = i * foldSize
            end = start + foldSize
            folds.append(data[start:end])
        return folds

    correctCount = 0
    folds = getFolds()
    for i, testFold in enumerate(folds):
        trainFolds = []
        for j in range(k):
            if j == i: continue
            trainFolds.extend(folds[j])

        testLabels, testFeatures = testFold[:, 0], testFold[:, 1:]
        trainLabels, trainFeatures = trainFolds[:, 0], trainFolds[:, 1:]
        for i in range(len(testFeatures)):
            testLabel = testLabels[i]
            testFeature = testFeatures[i]
            bestLabel, minDistance = None, float("inf")
            for j in len(trainFeatures):
                #trainLabel = trainLabels[j]
                train = trainFeatures[i]
                dist = EuclidDist(testFeature, train)
                if dist < minDistance:
                    minDistance = dist
                    bestLabel = trainLabels[j]
            if bestLabel == testLabel:
                correctCount += 1
            else:
                # Pruning code goes here
                pass
                #return -1
    return correctCount / instanceCount

f = [8, 3, 5]
test = accuracy(data, f)
print(test)