import os
from sklearn import svm, metrics
from sklearn.multioutput import MultiOutputRegressor
from Dataset import FingerprintingDataset
from Preprocessing import Normalizer




def svmExperiment():
    # Define important variables
    pathToData = f'{os.getcwd()}/dataset/db'

    # 1. Define dataset, transforms and loader
    trainDataset = FingerprintingDataset(rootDir=pathToData)
    trainNormalizer = Normalizer(rssMin=trainDataset.getRssMin() - 1)
    xTrain = trainNormalizer(trainDataset.data)
    yTrain = trainDataset.labels[:, 2]
    yTrainReg = trainDataset.labels[:, [0,1]]

    testDataset = FingerprintingDataset(rootDir=pathToData, test=True)
    testNormalizer = Normalizer(rssMin=testDataset.getRssMin() - 1)
    xTest = testNormalizer(testDataset.data)
    yTest = testDataset.labels[:, 2]
    yTestReg = testDataset.labels[:, [0,1]]

    clf = svm.SVC(kernel='linear')
    clf.fit(xTrain, yTrain)

    yPredClf = clf.predict(xTest)
    print("Classification Accuracy: ", metrics.accuracy_score(yTest, yPredClf))

    rgr = svm.SVR()
    mor = MultiOutputRegressor(rgr)
    mor.fit(xTrain, yTrainReg)

    yPredRgr = mor.predict(xTest)
    print("Regression Mean Squared Error X: ", metrics.mean_squared_error(yTestReg[:, 0], yPredRgr[:, 0]))
    print("Regression Mean Squared Error Y: ", metrics.mean_squared_error(yTestReg[:, 1], yPredRgr[:, 1]))

    print(yTestReg)
    print(yPredRgr)





