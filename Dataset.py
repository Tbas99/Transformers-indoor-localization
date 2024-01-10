import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FingerprintingDataset(Dataset):
    """RSSI Fingerprinting dataset. By Mendoza-Silva et al."""

    def __init__(self, rootDir, test=False, transform=None, targetTransform=None):
        """
        Args:
            root_dir (string): Database directory with all 
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        # Gather data
        rssData = np.zeros((0,620))
        posData = np.zeros((0,3))
        for root, _, files in os.walk(rootDir):
            # First iteration is through the root, skip it.
            if root == rootDir:
                continue
            
            # Read Define filter for dataset requirements
            if test:
                rssFiles = list(filter(lambda k: 'rss' in k and 'tst' in k, files))
                posFiles = list(filter(lambda k: 'crd' in k and 'tst' in k, files))
            else:
                rssFiles = list(filter(lambda k: 'rss' in k and 'trn' in k, files))
                posFiles = list(filter(lambda k: 'crd' in k and 'trn' in k, files))

            # Read data
            for file in rssFiles:
                pathToData = f'{root}/{file}'
                data = np.genfromtxt(pathToData, delimiter=',')
                rssData = np.append(rssData, data, axis=0)

            # Read labels
            for file in posFiles:
                pathToData = f'{root}/{file}'
                data = np.genfromtxt(pathToData, delimiter=',')

                # Transform labels to binary classification problem where floor 5 corresponds to 1
                # and floor 3 corresponds to 0
                newData = data
                newData[:, 2] = np.where(newData[:, 2] == 5, 1, 0)

                posData = np.append(posData, newData, axis=0)
        
        self.data = rssData
        self.labels = posData
        self.rootDir = rootDir
        self.transform = transform
        self.targetTransform = targetTransform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get sample and label at index
        sample, label = self.data[idx], self.labels[idx]

        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        if self.targetTransform:
            label = self.targetTransform(label)

        labelReg = label[[0,1]]
        labelClass = label[[2]]

        # return sample and labels
        return sample.long(), labelReg.float(), labelClass
    
    def getRssMin(self):
        return self.data.min()