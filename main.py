import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from Dataset import FingerprintingDataset
from Preprocessing import Normalizer
from Vanilla import VanillaTransformer
from BERT import BERT
from Linformer import Linformer
from Simple import Classic
from Utils import train_single_epoch, eval_single_epoch







def execPipeline(model):
    # Define important variables
    pathToData = f'{os.getcwd()}/dataset/db'
    epochs = 5

    # 1. Define dataset, transforms and loader
    trainDataset = FingerprintingDataset(rootDir=pathToData)
    normalizer = Normalizer(rssMin=trainDataset.getRssMin() - 1)
    #trainDataset.transform = transforms.Compose([normalizer, torch.from_numpy])
    trainDataset.transform = torch.from_numpy
    trainDataset.targetTransform = torch.from_numpy
    trainDataloader = DataLoader(
        dataset=trainDataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    testDataset = FingerprintingDataset(rootDir=pathToData, test=True)
    normalizer = Normalizer(rssMin=testDataset.getRssMin() - 1)
    #testDataset.transform = transforms.Compose([normalizer, torch.from_numpy])
    testDataset.transform = torch.from_numpy
    testDataset.targetTransform = torch.from_numpy
    testDataloader = DataLoader(
        dataset=testDataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )


    # 2. Define training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    regressionLossFunction = torch.nn.MSELoss()
    classificationLossFunction = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # Training procedure
    for epoch in range(epochs):
        print(f'EPOCH {epoch}:')

        train_single_epoch(
            dataloader=trainDataloader, 
            model=model,
            regressionLossFn=regressionLossFunction,
            classificationLossFn=classificationLossFunction,
            optimizer=optimizer,
            device=device
        )

        eval_single_epoch(
            dataloader=testDataloader,
            model=model,
            regressionLossFn=regressionLossFunction,
            classificationLossFn=classificationLossFunction,
            device=device
        )








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
    torch.cuda.empty_cache()

    AVAILABLE_MODELS = ['Vanilla', 'Perceiver', 'Linformer', 'BERT']

    # Define constants
    selectedModel = AVAILABLE_MODELS[2]

    # As = np.array([[1,2],[3,4],[5,6]])
    # Bs = np.array([[3,4],[3,4],[7,8]])

    # Cd = np.sqrt(np.sum((As - Bs)**2, axis=1))

    # As = torch.from_numpy(As)
    # Bs = torch.from_numpy(Bs)

    # Cs = torch.sqrt(torch.sum((As - Bs)**2, dim=1))

    # print(Cd)
    # print(Cs)

    # Run training, evaluation and gather statistics for selected model
    if selectedModel == "BERT":
        model = BERT(
            embed_dim=96,
            hidden_size=512,
            src_vocab_size=200,
            seq_length=620
        )
    elif selectedModel == "Perceiver":
        model = Classic()
    elif selectedModel == "Linformer":
        model = Linformer(
            dim=512,
            seq_len=620,
            depth=6
        )
    else:
        model = VanillaTransformer(
            embed_dim=256,
            src_vocab_size=200,
            seq_length=620,
            num_layers=2,
            expansion_factor=4,
            n_heads=8
        )

    execPipeline(model=model)

    

