from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn


class AnyDataset(Dataset):
    def __init__(
        self, path, labels_path=None,
        transform_orig=None, transform_seg=None
    ):
        self.images_path = path
        self.labels_path = labels_path
        self.transform_img = transform_orig
        self.transform_label = transform_seg

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = plt.imread(self.images_path[idx])
        if self.labels_path:
            mask = plt.imread(self.labels_path[idx])
        image = img[:, :int(img.shape[1]/2)] if not self.labels_path else img
        label = img[:, int(img.shape[1]/2):] if not self.labels_path else mask

        if self.transform_img:
            image = self.transform_img(image)

        if self.transform_label:
            label = self.transform_label(label)

        return image, label


def show(img, pred, label, save_path, epoch, batch_idx=0):
    img, pred, label = img.cpu(), pred.cpu(), label.cpu()
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    img, pred, label = img[batch_idx], pred[batch_idx], label[batch_idx]
    _img, _pred = img, pred.detach().permute(1, 2, 0).numpy()
    ax[0].imshow(_img.permute(1, 2, 0))
    ax[1].imshow(_pred)
    ax[2].imshow(label.permute(1, 2, 0))

    if save_path:
        plt.savefig(f"{save_path}{epoch}") 
    plt.show()


def train(model, train_data, val_data, criterion, optimizer,
        epochs=50, batch_size=8, show_every=10,
        save_path=None):
    train_loader = DataLoader(train_data, batch_size)
    valid_loader = DataLoader(val_data, 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loss, val_loss = [], []

    for i in range(1, epochs + 1):
        running_loss = 0

        for img, label in tqdm(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(round(running_loss / len(train_loader), 6))

        model.eval()
        running_loss = 0
        for img, label in tqdm(valid_loader):
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = criterion(pred, label)
            running_loss += loss.item()
        model.train()
        val_loss.append(round(running_loss / len(valid_loader), 6))

        print(
            f"epoch : {i}",
            f"train loss : {train_loss[-1]},",
            f"valid loss : {val_loss[-1]},"
        )
        if not i % show_every or i == 1:
            show(img, pred, label, save_path, i)

    return train_loss, val_loss,