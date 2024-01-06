import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from Dataset import FingerprintingDataset
from Preprocessing import Normalizer
# from Vanilla 








def execPipeline():
    # Define important variables
    pathToData = f'{os.getcwd()}/dataset/db'
    epochs = 20

    # 1. Define dataset, transforms and loader
    trainDataset = FingerprintingDataset(rootDir=pathToData)
    normalizer = Normalizer(rssMin=trainDataset.getRssMin())
    trainDataset.transform = transforms.Compose([normalizer, torch.from_numpy])
    trainDataset.targetTransform = torch.from_numpy
    trainDataloader = DataLoader(
        dataset=trainDataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    inputs, classes = iter(trainDataloader).next()
    print(inputs.shape)
    print(classes.shape)


    # Training procedure
    #for epoch in range(epochs):





def getDataset():
    pathToData = f'{os.getcwd()}/dataset/db'

    rssTrain = np.zeros((0,620))
    rssTest = np.zeros((0,620))

    for root, _, files in os.walk(pathToData):
        # First iteration is through the root, skip it.
        if root == pathToData:
            continue
        
        # Read RSS file and store data
        rssFiles = list(filter(lambda k: 'rss' in k, files))
        for file in rssFiles:
            m = re.match(r'(.{3})(.{2})(.{3})\.csv', file)
            datasetKind, measurementNumber, dataType = m.group(1, 2, 3)

            pathToData = f'{root}/{file}'
            data = np.genfromtxt(pathToData, delimiter=',')
            #print(data.shape)
            #print(f'Concatting {file}')
            if datasetKind == 'trn':
                rssTrain = np.append(rssTrain, data, axis=0)
            else:
                rssTest = np.append(rssTest, data, axis=0)


            # match datasetKind:
            #     case 'trn':
            #         rssTrain = np.append(rssTrain, data, axis=0)
            #     case 'tst':
            #         rssTest = np.append(rssTest, data, axis=0)

    print('Done')
    print(rssTrain.shape)
    print(rssTrain)
    print(rssTest.shape)
    print(rssTest)
    print(len(rssTrain))

    rssMin = rssTrain.min()
    normalizer = Normalizer(rssMin=rssMin)
    normalizedValues = normalizer(rssTrain)
    print(normalizedValues.shape)
    print(normalizedValues.min())
    print(normalizedValues.max())


if __name__ == "__main__":
    AVAILABLE_MODELS = ['Vanilla', 'Perceiver', 'Linformer', 'BERT']

    # Define constants
    selectedModel = AVAILABLE_MODELS[0]

    #getDataset()

    # Run training, evaluation and gather statistics for selected model
    # match selectedModel:
    #     case 'Vanilla':
    execPipeline()

    

