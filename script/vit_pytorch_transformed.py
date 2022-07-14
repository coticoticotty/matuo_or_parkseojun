import torchvision.datasets
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm

from utils import show_graphs

def main():
    image_size = 224
    # mean = (0.485, 0.456, 0.406)
    mean = IMAGENET_DEFAULT_MEAN  # -> (0.485, 0.456, 0.406) 値は一緒だった。
    # std = (0.229, 0.224, 0.225)
    std = IMAGENET_DEFAULT_STD  # -> (0.229, 0.224, 0.225) 値は一緒だった。

    project_folder = Path(__file__).resolve().parent.parent
    data_folder = os.path.join(project_folder, 'data')
    train_image_dir = os.path.join(data_folder, 'train')
    val_image_dir = os.path.join(data_folder, 'val')

    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=(0.5, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=[-15, 15]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(0.5),
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    train_dataset = torchvision.datasets.ImageFolder(root=train_image_dir, transform=data_transform['train'])
    val_dataset = torchvision.datasets.ImageFolder(root=val_image_dir, transform=data_transform['val'])

    batch_size = 32

    train_dataLoader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataLoader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    dataloaders_dict = {'train': train_dataLoader, 'val': val_dataLoader}

    # 学習済みのVision Transferモデルをロード
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # 損失関数はクロスエントロピー
    loss_func = nn.CrossEntropyLoss()

    # 全ての層のパラメータを訓練不可に
    for param in model.parameters():
        param.requires_grad = False

    # 一部の層を入れ替え（デフォルトで訓練可能）
    model.heads[0] = nn.Linear(768, 2)

    # 最適化アルゴリズム
    optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 50
    train_model(model, dataloaders_dict, loss_func, optimizer, num_epochs=num_epochs)

    torch.save(model.state_dict(), 'pytorch_vit_transfered_model.pth')

# ImageFolderに画像を読み込ませるときにすべてRGBに変換する処理
def myloader(filename):
    return Image.open(filename).convert('RGB')

# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, loss_func, optimizer, num_epochs):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # epochのループ
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} / {num_epochs}')
        print('-------------')

        for phase in ['train', 'val']:

            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # optimizerを初期化
                optimizer.zero_grad()

                # forward計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = loss_func(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # 訓練時はバックプロパゲーションを行う
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # イテレーション結果の計算
                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)


            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            # epochごとの値を格納
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            elif phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            print(f'{phase} Loss {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    models_name = 'ViTTransfered'
    show_graphs(num_epochs, train_loss, val_loss, train_acc, val_acc, models_name)

if __name__ == '__main__':
    main()