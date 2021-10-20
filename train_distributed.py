from config.food_classifier import FoodClassifier
from config import config, create_dataloaders
from sklearn.metrics import classification_report
from torchvision.models import densenet121
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

def main():
    NUM_GPU = torch.cuda.device_count()
    print(f"[INFO] number of GPUs found: {NUM_GPU}...")

    BATCH_SIZE = config.LOCAL_BATCH_SIZE * NUM_GPU
    print(f"[INFO] using a batch size of {BATCH_SIZE}...")

    trainTransform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    testTransform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    (trainDS, trainLoader) = create_dataloaders.get_dataloader(config.TRAIN,
        transforms=trainTransform, bs=BATCH_SIZE)
    (valDS, valLoader) = create_dataloaders.get_dataloader(config.VAL,
        transforms=testTransform, bs=BATCH_SIZE, shuffle=False)
    (testDS, testLoader) = create_dataloaders.get_dataloader(config.TEST,
        transforms=testTransform, bs=BATCH_SIZE, shuffle=False)

    # load the DenseNet121 model
    baseModel = densenet121(pretrained=True)

    for module, param in zip(baseModel.modules(), baseModel.parameters()):
        if isinstance(module, nn.BatchNorm2d):
            param.requires_grad = False

    # define custom model
    model = FoodClassifier(baseModel, len(trainDS.classes))
    model = model.to(config.DEVICE)

    if NUM_GPU > 1:
        model = nn.DataParallel(model)

    lossFunc = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=config.LR * NUM_GPU)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    lrScheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    trainSteps = len(trainDS) // BATCH_SIZE
    valSteps = len(valDS) // BATCH_SIZE

    # initialize a dictionary to store training history
    H = {"train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []}

    print("[INFO] training the network...")
    startTime = time.time()

    for e in tqdm(range(config.EPOCHS)):
        model.train()

        totalTrainLoss = 0
        totalValLoss = 0

        trainCorrect = 0
        valCorrect = 0

        for (x, y) in trainLoader:
            with torch.cuda.amp.autocast(enabled=True):
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                pred = model(x)
                loss = lossFunc(pred, y)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            totalTrainLoss += loss.item()
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            torch.cuda.empty_cache()
        
        lrScheduler.step()

        with torch.no_grad():
            model.eval()

            for (x, y) in valLoader:
                with torch.cuda.amp.autocast(enabled=True):
                    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                    pred = model(x)
                    totalValLoss += lossFunc(pred, y).item()
                
                valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        trainCorrect = trainCorrect / len(trainDS)
        valCorrect = valCorrect / len(valDS)

        # update training history
        H["train_loss"].append(avgTrainLoss)
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss)
        H["val_acc"].append(valCorrect)

        print("[INFO] EPOCH: {}/{}".format(e+1, config.EPOCHS))
        print("Train Loss: {:.6f}, Train Accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val Loss: {:.6f}, Val Accuracy: {:.4f}".format(avgValLoss, valCorrect))

    elapsed = time.time() - startTime
    print("[INFO] total time taken to train the model: {:.1f}m {:.1f}s".format(elapsed//60, elapsed%60))

    # evaluate the network
    print("[INFO] evaluating network...")
    with torch.no_grad():
        model.eval()
        preds = []
        for (x, _) in testLoader:
            x = x.to(config.DEVICE)
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

    # generate a classification report
    print(classification_report(testDS.targets, preds, target_names=testDS.classes))

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)

    torch.save(model.state_dict(), config.MODEL_PATH)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()