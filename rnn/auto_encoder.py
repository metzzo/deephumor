# code adapted from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py

import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

from cnn_experiments.base_model import BaseCNNModel
from datamanagement.cartoon_cnn_dataset import CartoonCNNDataset
from rnn import TRAIN_PATH, IMAGES_PATH


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 256, 256)
    return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


img_transform = [
    transforms.RandomCrop(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]


class AutoEncoderModel(BaseCNNModel):
    def get_network_class(self):
        return lambda x: 42

    def load_image(self, img_name):
        img = super().load_image(os.path.join(IMAGES_PATH, os.path.basename(img_name)))
        img = ImageOps.fit(img, (300, 300), method=Image.BILINEAR)
        return img


def main():
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3

    dataloader = DataLoader(CartoonCNNDataset(file_path=TRAIN_PATH, model=AutoEncoderModel(), trafo=img_transform), batch_size=batch_size, shuffle=True)



    model = AutoEncoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader:
            _, img, _ = data
            img = torch.tensor(img).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.item()))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './dc_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './conv_autoencoder.pth')

if __name__ == '__main__':
    main()
