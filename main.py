import os
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset import FingerprintingDataset
from Preprocessing import Normalizer
from Vanilla import VanillaTransformer
from BERT import BERT
from Linformer import Linformer
from Perceiver import PerceiverIO
from Utils import train_single_epoch, eval_single_epoch


def execPipeline(model, experimentName):
    # Define important variables
    pathToData = f'{os.getcwd()}/dataset/db'
    epochs = 20

    # 1. Define dataset, transforms and loader
    trainDataset = FingerprintingDataset(rootDir=pathToData)
    normalizer = Normalizer(rssMin=trainDataset.getRssMin() - 1)
    trainDataset.transform = transforms.Compose([normalizer, torch.from_numpy])
    #trainDataset.transform = torch.from_numpy
    trainDataset.targetTransform = torch.from_numpy
    trainDataloader = DataLoader(
        dataset=trainDataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    testDataset = FingerprintingDataset(rootDir=pathToData, test=True)
    normalizer = Normalizer(rssMin=testDataset.getRssMin() - 1)
    testDataset.transform = transforms.Compose([normalizer, torch.from_numpy])
    #testDataset.transform = torch.from_numpy
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=[0.9, 0.98], eps=10e-9)

    epochRegLoss = np.empty(shape=(1, epochs))
    epochClassLoss = np.empty(shape=(1, epochs))
    epochAcc = np.empty(shape=(1, epochs))
    epochDistError = np.empty(shape=(1, epochs))
    epochTrainingTime = 0

    # Training procedure
    for epoch in range(epochs):
        print(f'EPOCH {epoch}:')

        start = time.time()
        train_single_epoch(
            dataloader=trainDataloader, 
            model=model,
            regressionLossFn=regressionLossFunction,
            classificationLossFn=classificationLossFunction,
            optimizer=optimizer,
            device=device
        )
        end = time.time()
        epochTrainingTime += (end - start)

        regLoss, classLoss, acc, distError = eval_single_epoch(
            dataloader=testDataloader,
            model=model,
            regressionLossFn=regressionLossFunction,
            classificationLossFn=classificationLossFunction,
            device=device
        )

        epochRegLoss[0, epoch] = regLoss
        epochClassLoss[0, epoch] = classLoss
        epochAcc[0, epoch] = acc
        epochDistError[0, epoch] = distError

    # Save results
    avgEpochTrainingTime = epochTrainingTime / epochs
    print("--- Average epoch training time: %s seconds ---" % (avgEpochTrainingTime))
    np.savetxt(f'{experimentName}.out', (epochRegLoss.flatten(), epochClassLoss.flatten(), epochAcc.flatten(), epochDistError.flatten()))

    x = range(1, epochs + 1)
    fig1, (ax1, ax2)  = plt.subplots(1, 2)
    ax1.plot(x, epochRegLoss.flatten(), c='b')
    ax1.set_title("Regression Loss Curve", fontsize=16)
    ax1.set_xticks(x)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax2.plot(x, epochClassLoss.flatten(), c='r')
    ax2.set_title('Classification Loss Curve', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    fig1.tight_layout()
    plt.savefig(f'{experimentName}_lossCurve.jpeg', dpi=1200)

    fig2, ax3 = plt.subplots()
    ax3.plot(x, epochAcc.flatten()*100)
    ax3.set_title("Classification Accuracy", fontsize=16)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    fig2.tight_layout()
    plt.savefig(f'{experimentName}_accuracy.jpeg', dpi=1200)

    fig3, ax4 = plt.subplots()
    ax4.plot(x, epochDistError.flatten())
    ax4.set_title("Average Euclidean Difference", fontsize=16)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_ylabel('Distance error (m)', fontsize=12)
    fig3.tight_layout()
    plt.savefig(f'{experimentName}_error.jpeg', dpi=1200)

def test():
    arr = np.loadtxt('epoch_results.out')
    print(arr)

    regLoss = arr[0, :]
    classLoss = arr[1, :]
    acc = arr[2, :]
    err = arr[3, :]

    print(regLoss)
    print(regLoss.shape)

    x = range(1, 6)

    fig1, (ax1, ax2)  = plt.subplots(1, 2)
    ax1.plot(x, regLoss, c='b')
    ax1.set_title("Regression Loss Curve", fontsize=16)
    ax1.set_xticks(x)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax2.plot(x, classLoss, c='r')
    ax2.set_title('Classification Loss Curve', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    fig1.tight_layout()
    plt.show()

    fig2, ax3 = plt.subplots()
    ax3.plot(x, acc*100)
    ax3.set_title("Classification Accuracy", fontsize=16)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    fig2.tight_layout()
    plt.show()

    fig3, ax4 = plt.subplots()
    ax4.plot(x, err)
    ax4.set_title("Average Euclidean Difference", fontsize=16)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_ylabel('Distance error (m)', fontsize=12)
    fig3.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING']="1"

    AVAILABLE_MODELS = ['Vanilla', 'Perceiver', 'Linformer', 'BERT']

    # Define constants
    selectedModel = AVAILABLE_MODELS[0]

    # Run training, evaluation and gather statistics for selected model
    if selectedModel == "BERT":
        model = BERT(
            embed_dim=96,
            hidden_size=512,
            src_vocab_size=200,
            seq_length=620
        )
    elif selectedModel == "Perceiver":
        model = PerceiverIO(
            depth=4,
            dim=512,
            logits_dim=3,
            num_latents=256,
            latent_dim=512,
            seq_dropout_prob=0.2
        )
    elif selectedModel == "Linformer":
        model = Linformer(
            dim=512,
            seq_len=620,
            depth=6
        )
    else:
        # Base model includes:
        # N = 6 Layers
        # D_model = 512
        # h = 8 heads
        # No dropout
        model = VanillaTransformer(
            embed_dim=512,
            src_vocab_size=100,
            seq_length=620,
            num_layers=6,
            expansion_factor=4,
            n_heads=8
        )

    execPipeline(model=model, experimentName=selectedModel)

    

