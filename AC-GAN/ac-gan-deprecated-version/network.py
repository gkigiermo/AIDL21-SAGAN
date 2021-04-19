import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, class_dim, latent_dim, image_size):
        """ Initializes Generator Class with latent_dim and class_dim."""
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.class_dim = class_dim

        # self.label_emb = nn.Embedding(self.class_dim, self.latent_dim)

        self.init_size = image_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        return

    def forward(self, _input):
        _input = _input.view(-1, self.latent_dim)
        out = self.l1(_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):

    def __init__(self, class_dim, image_size):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = image_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, class_dim), nn.Softmax(dim=1))

    def forward(self, _input):
        out = self.conv_blocks(_input)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


if __name__ == '__main__':
    latent_dim = 100
    class_dim = 2
    batch_size = 128
    img_size = 64

    cuda = False if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
    # c = Variable(LongTensor(np.random.randint(0, class_dim, batch_size)))
    label = np.random.randint(0, class_dim, batch_size)
    noise = Variable(FloatTensor(batch_size, latent_dim, 1, 1))
    aux_label = Variable(LongTensor(batch_size))

    noise_ = np.random.normal(0, 1, (batch_size, latent_dim))
    class_onehot = np.zeros((batch_size, class_dim))
    class_onehot[np.arange(batch_size), label] = 1
    noise_[np.arange(batch_size), :class_dim] = class_onehot[np.arange(batch_size)]
    noise_ = (torch.from_numpy(noise_))
    noise.data.copy_(noise_.view(batch_size, latent_dim, 1, 1))
    aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

    print(f'z:{noise.shape}')
    print(f'c:{aux_label.shape}')

    G = Generator(class_dim, latent_dim, img_size)
    D = Discriminator(class_dim, img_size)
    o = G(noise)
    print(o.shape)
    x, y = D(o)
    print(x.shape, y.shape)
