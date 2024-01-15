from tqdm import tqdm
import torch

def train_single_epoch(dataloader, model, regressionLossFn, classificationLossFn, optimizer, device):
    model.train()

    correctFloorPredsTotal = 0
    sizeTotal = 0
    lossRegressionWeight = 0.4

    with tqdm(dataloader, unit="batch") as tepoch:
        for x, yReg, yClass in tepoch:
            tepoch.set_description("Training...")
            x, yReg, yClass = x.to(device), yReg.to(device), yClass.to(device)

            # -- Forward pass
            logits = model(x)
            yHatReg = logits[:, [0,1]]
            yHatClass = logits[:, [2]]

            # -- Backprop
            optimizer.zero_grad()
            lossRegression = regressionLossFn(yHatReg, yReg)
            lossClassification = classificationLossFn(yHatClass, yClass)
            loss = lossRegressionWeight * lossRegression + lossClassification
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # -- Compute metrics - Regression
            pointwiseEucDist = torch.sqrt(torch.sum((yHatReg - yReg)**2, dim=1))
            avgDistanceError = pointwiseEucDist.mean()
            
            # -- Compute metrics - Classification
            floorPreds = torch.round(torch.sigmoid(yHatClass))
            numCorrect = (floorPreds == yClass).sum().item()
            correctFloorPredsTotal += numCorrect
            sizeTotal += len(yClass)
            classificationAccuracy = correctFloorPredsTotal / sizeTotal
            
            # -- Update the progress bar values
            tepoch.set_postfix(
                lossClass=lossClassification.item(),
                accClassification=format(classificationAccuracy, "3.2%"),
                lossReg=lossRegression.item(),
                distError=avgDistanceError.item()
            )

def eval_single_epoch(dataloader, model, regressionLossFn, classificationLossFn, device):
    model.eval()

    correctFloorPredsTotal = 0
    sizeTotal = 0

    lossRegressionEval = 0
    lossClassificationEval = 0
    avgPointwiseEucDist = 0

    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for x, yReg, yClass in tepoch:
                tepoch.set_description("Validating...")
                x, yReg, yClass = x.to(device), yReg.to(device), yClass.to(device)

                # -- Forward pass
                logits = model(x)
                yHatReg = logits[:, [0,1]]
                yHatClass = logits[:, [2]]

                # -- Compute loss
                lossRegression = regressionLossFn(yHatReg, yReg)
                lossClassification = classificationLossFn(yHatClass, yClass)

                # -- Compute metrics - Regression
                pointwiseEucDist = torch.sqrt(torch.sum((yHatReg - yReg)**2, dim=1))
                avgPointwiseEucDist += pointwiseEucDist.detach().cpu().numpy().sum()
                avgDistanceError = pointwiseEucDist.mean()
                
                # -- Compute metrics - Classification
                floorPreds = torch.round(torch.sigmoid(yHatClass))
                numCorrect = (floorPreds == yClass).sum().item()
                correctFloorPredsTotal += numCorrect
                sizeTotal += len(yClass)
                classificationAccuracy = correctFloorPredsTotal / sizeTotal
                
                # -- Update the progress bar values
                tepoch.set_postfix(
                    lossClass=lossClassification.item(),
                    accClassification=format(classificationAccuracy, "3.2%"),
                    lossReg=lossRegression.item(),
                    distError=avgDistanceError.item()
                )

                lossRegressionEval += lossRegression.detach().cpu().item()
                lossClassificationEval += lossClassification.detach().cpu().item()

    # Compute stats
    alpha = len(dataloader)
    lossRegressionEval /= alpha
    lossClassificationEval /= alpha
    avgPointwiseEucDist /= len(dataloader.dataset)

    return lossRegressionEval, lossClassificationEval, classificationAccuracy, avgPointwiseEucDist