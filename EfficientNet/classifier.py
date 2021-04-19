from dataset import *
from model import *
import torch
from torch.utils.data import DataLoader
from utils import *
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import albumentations
from albumentations.pytorch import ToTensorV2
from torchsampler import ImbalancedDatasetSampler
import pandas as pd


def callback_get_label(dataset, idx):
    # callback function used in imbalanced dataset loader.
    i, target = dataset[idx]
    return int(target)


def imbalanced_dataloader(dataset, bsize, save_address):
    train_loader = DataLoader(dataset, bsize, sampler = ImbalancedDatasetSampler(
        dataset, callback_get_label = callback_get_label))
    torch.save(train_loader, save_address)


def data_loader(bsize, dataset = None, balanced=False, imbalanced_dloader_address=None):
    if not balanced:
        dloader = torch.load(imbalanced_dloader_address)
    else:
        dloader = DataLoader(dataset, batch_size = bsize, shuffle = True)
    return dloader

def load_dataloader(path_train_dloader, path_img):
    print(path_train_dloader)
    train_loader = torch.load(path_train_dloader)
    train_loader.dataset.images_path = path_img


def main():
    model_number = '19041'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    #File configuration
    root_path = './'
    path_img = root_path + 'images'
    path_test_reduced = root_path + 'test_reduced.csv'
    path_train_reduced = root_path + 'train_m_reduced.csv'
    path_val_reduced = root_path + 'val_m_reduced.csv'
    path_save_model = root_path + 'model' + model_number
    path_train_dloader = root_path + 'Data/train_loader_reduced.pt'
    path_val_dloader = root_path + 'Data/val_loader_reduced.pt'
    path_results = root_path + "results_" + model_number

    hparams = {
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 0.0001,
        'frozen_layers': 16
    }

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def train_single_epoch(model, train_loader, optimizer, scheduler, criterion):

        model.train()
        accs, losses = [], []

        for x, y in train_loader:
            y = y.float()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.view(-1, 1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            y_ = y.cpu()
            output_ = torch.round(output.cpu())
            accs.append(accuracy_score(y_.detach(), output_.detach()).item())

        return np.mean(losses), np.mean(accs)

    def eval_single_epoch(model, val_loader, criterion):

        accs, losses, aucs = [], [], []
        with torch.no_grad():
            model.eval()

            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

            for x, y in val_loader:
                y = y.float()
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y.view(-1, 1))
                losses.append(loss.item())
                y_ = y.cpu()
                output_ = torch.round(output.cpu())
                accuracy = accuracy_score(y_.detach(), output_.detach()).item()
                accs.append(accuracy)

                try:
                    auc = roc_auc_score(y_.detach(), output_.detach()).item()
                except ValueError:
                    auc = 0
                aucs.append(auc)

        return np.mean(losses), np.mean(accs), np.mean(aucs)

    data_transforms = albumentations.Compose([
        albumentations.Resize(128, 128),
        albumentations.Normalize(mean, std, max_pixel_value = 255.0, always_apply = True),
        ToTensorV2(transpose_mask = True)

    ])

    train_transforms = albumentations.Compose([
        albumentations.Transpose(p = 0.5),
        albumentations.RandomContrast(p = 0.5),
        albumentations.VerticalFlip(p = 0.5),
        albumentations.Flip(p = 0.5),
        albumentations.Cutout(p = 1),
        albumentations.ShiftScaleRotate(p = 0.5),
        albumentations.OneOf([
            albumentations.RandomBrightnessContrast(brightness_limit = 0.3, contrast_limit = 0.3),
            albumentations.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1)
        ], p = 1),
        albumentations.HueSaturationValue(p = 0.5),
        albumentations.Resize(128, 128),
        albumentations.Normalize(mean, std, max_pixel_value = 255.0, always_apply = True),
        ToTensorV2(transpose_mask = True)
    ])

    valid_transforms = albumentations.Compose([
        albumentations.Resize(128, 128),
        albumentations.Normalize(mean, std, max_pixel_value = 255.0, always_apply = True),
        ToTensorV2(transpose_mask = True)
    ])

    train_dataset = MyDataset(path_img, path_train_reduced, train_transforms)
    val_dataset = MyDataset(path_img, path_val_reduced, valid_transforms)

    #imbalanced_dataloader(train_dataset, hparams["batch_size"], 'Data/train_loader_reduced.pt')
    #imbalanced_dataloader(val_dataset, hparams["batch_size"], 'Data/val_loader_reduced.pt')

    train_loader = data_loader(hparams["batch_size"], dataset = None, balanced = False,
                               imbalanced_dloader_address = path_train_dloader)
    val_loader = data_loader(hparams["batch_size"], dataset = None, balanced = False,
                               imbalanced_dloader_address = path_val_dloader)

    train_loader.dataset.images_path = path_img
    val_loader.dataset.images_path = path_img
    #Network definition
    network = MyModel(frozen_layers=hparams["frozen_layers"], trained_features="effnet" ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), hparams["learning_rate"])
    criterion = nn.BCELoss()

    #Patience and LR Reduction
    scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'max', patience = 5, verbose = True, factor = 0.5)
    es_patience = 18
    best_val = 0


    train_loss_history, val_loss_history, train_acc_history, val_acc_history, val_auc_history = [], [], [], [], []

    for epoch in range(hparams["num_epochs"]):

        train_loss, train_acc = train_single_epoch(network, train_loader, optimizer, scheduler, criterion)
        print(f"Train Epoch {epoch} train loss={train_loss:.2f} train acc={train_acc:.2f}")

        val_loss, val_acc, val_auc = eval_single_epoch(network, val_loader, criterion)
        if val_auc >= best_val:
            best_val = val_auc
            patience = es_patience  # Resetting patience since we have new best validation accuracy
            torch.save(network.state_dict(), path_save_model)  # Saving current best model

        else:
            patience -= 1
            if patience == 0:
                print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                break
        print(f"Evaluate Epoch {epoch} eval loss={val_loss:.2f} eval acc={val_acc:.2f}  eval auc={val_auc:.2f}")
        scheduler.step(val_auc)

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        val_auc_history.append(val_auc)

        results = pd.DataFrame(
            data = [train_loss_history, train_acc_history, val_loss_history, val_acc_history, val_auc_history])
        results.to_csv(path_results)

    fig = plt.figure(figsize = (20, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(train_loss_history, label = 'Training Loss')
    ax1.plot(val_loss_history, label = 'Validation Loss')
    ax1.set_title("Losses")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_acc_history, label = 'Training accuracy')
    ax2.plot(val_acc_history, label = 'Validation Accuracy Score')
    ax2.set_title("Accuracies")
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()


if __name__ == "__main__":
    main()
