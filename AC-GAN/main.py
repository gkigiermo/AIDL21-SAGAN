import argparse
import os
import re
import sys
from math import sqrt, floor

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from piqa import SSIM, HaarPSI, PSNR, MS_SSIM, MS_GMSD, MDSI
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import gc

gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

os.makedirs("images", exist_ok=True)
os.makedirs("output", exist_ok=True)
np.random.seed(43)

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./ISIC-Archive/Data/",
                    help="images parent directory")
parser.add_argument("--load-model", type=bool, default=False, help="resume training/eval from checkpoint")
parser.add_argument("--eval", type=str, default=False, help="trained model sampling")
parser.add_argument("--n-epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--batch-size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n-cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent-dim", type=int, default=600, help="dimensionality of the latent space")
parser.add_argument("--n-classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--img-size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample-interval", type=int, default=200, help="interval between image sampling")
parser.add_argument("--print-interval", type=int, default=5,
                    help="batch interval between printing out the progress bar")
parser.add_argument("--save-model-interval", type=int, default=100, help="epoch interval between model saving")
opt = parser.parse_args()  # replace if inline execution: opt = parser.parse_args(args=[])
print(opt)
# exit()

cuda = True if torch.cuda.is_available() else False
print(f'Cuda: {cuda}')


# Prepare dataset
class ISICDataset(Dataset):
    def __init__(self, labels_path, images_path, transform=None, color=False, is_test=False):
        super().__init__()
        self.labels_df = pd.read_csv(labels_path)
        self.images_path = images_path
        self.transform = transform
        self.color = color
        self.is_test = is_test

    def __len__(self):
        return self.labels_df.shape[0]

    def __getitem__(self, idx):
        image_path = f"{self.images_path}/{self.labels_df.iloc[idx]['dcm_name']}.jpeg"
        target = self.labels_df.iloc[idx]['target']

        try:
            image = transforms.ToPILImage()(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        except:
            image_path = os.path.join(self.images_path, self.labels_df.iloc[idx]['dcm_name'] + ".png")
            image = transforms.ToPILImage()(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

        if self.transform:
            if self.color:
                image = color_transform(image)
            image = self.transform(image)

        if self.is_test:
            return image

        return image, torch.tensor([target], dtype=torch.float32)


# Auxiliar functions
def get_isic_dataloader(path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor()
    ])

    dataset = ISICDataset(os.path.join(path, 'train_data.csv'), os.path.join(path, 'Images'),
                          transform=transform, color=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return data_loader


def color_transform(x):
    x = F.adjust_saturation(x, 0.9)
    x = F.adjust_gamma(x, 1.5)
    x = F.adjust_contrast(x, 1.7)
    return x


def load_generator(G_path):
    prev_state = torch.load(G_path)

    G = Generator()
    G.load_state_dict(prev_state['model'])
    G.eval()
    return G


def load_discriminator(D_path):
    prev_state = torch.load(D_path)

    D = Discriminator()
    D.load_state_dict(prev_state['model'])
    D.eval()
    return D


def save_model(model, optimizer, e, file_path):
    """ Save model checkpoints. """
    if opt.load_model:
        file_path = os.path.join('output', 'ckpt', f'G_{max(numbers)+e:04}.pth')

    state = {'model': model.state_dict(),
             'optim': optimizer.state_dict(),
             }
    torch.save(state, file_path)
    return


def loss_plot(G_losses, D_losses, toPath):
    os.makedirs(toPath, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(toPath, 'loss_plot.png'))


def compute_metrics(real, fakes, toPath):
    p, s, h, ms, md = [], [], [], [], []

    ssim = SSIM().cpu()
    psnr = PSNR()
    haar = HaarPSI()
    msssim = MS_SSIM()
    ms_gmsd = MS_GMSD()
    mdsi = MDSI()

    if real[0].shape[0] >= fakes.shape[0]:
        thres = fakes.shape[0]
    else:
        thres = real[0].shape[0]

    for i in range(0, thres - 1):
        f = fakes[i].unsqueeze(dim=0).detach().cpu()
        r = real[0][i].unsqueeze(dim=0).detach().cpu()
        r_norm = (r - r.min()) / (r.max() - r.min())
        f_norm = (f - f.min()) / (f.max() - f.min())

        p.append(psnr(r_norm, f_norm))
        s.append(ssim(r_norm, f_norm))
        h.append(haar(r_norm, f_norm))
        ms.append(ms_gmsd(r_norm, f_norm))
        md.append(mdsi(r_norm, f_norm))

    original_stdout = sys.stdout
    with open(os.path.join(toPath, 'metrics_report.log'), mode='a+') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(f'PSNR: {sum(p) / (len(p)):4f}, SSIM: {sum(s) / (len(s)):4f}, '
              f'HAAR: {sum(h) / (len(h)):4f}, MSGMSD: {sum(ms) / (len(ms)):4f}, '
              f'MDSI: {sum(md) / (len(md)):4f}')
        sys.stdout = original_stdout  # Reset the standard output to its original value

    return 0


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def sample_class_images(nImgs, ofst):
    fake_samples = os.path.join("output", "fake_samples")
    os.makedirs(fake_samples, exist_ok=True)

    z = Variable(FloatTensor(np.random.normal(0, 1, (nImgs, opt.latent_dim))))
    labels = Variable(LongTensor(np.ones(nImgs))) if opt.eval else Variable(LongTensor(np.random.randint(0, opt.n_classes, nImgs)))
    with torch.no_grad():
        fake_imgs = generator(z, labels)

    for img in range(nImgs):
        save_image(fake_imgs[img].data, f'{fake_samples}/FISIC_{img+ofst+1:07}.png')



def sample_val_image(z, labels, n_row, iter_done):
    """Saves a grid of generated digits ranging from 0 to n_row"""
    fake_imgs = generator(z, labels)
    save_image(fake_imgs.data, "images/%d.png" % iter_done, nrow=n_row, normalize=True)
    return fake_imgs


# Network definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

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
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax(dim=1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = get_isic_dataloader(opt.path, opt.batch_size)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------
loss_G, loss_D = [], []

if opt.load_model:
    regex = re.compile(r'\d+')
    numbers = []
    try:
        for filename in os.listdir('output/ckpt'):
            [numbers.append(int(x)) for x in regex.findall(filename) if int(x) not in numbers]
    except Exception as e:
        raise e
    G_path = os.path.join('output', 'ckpt', f'G_{max(numbers):04}.pth')
    D_path = os.path.join('output', 'ckpt', f'D_{max(numbers):04}.pth')
    # Load an override generator and discrimnator
    generator = load_generator(G_path).cuda()
    discriminator = load_discriminator(D_path).cuda()

# Sample noise_val and generate labels_val
z_val = Variable(FloatTensor(np.random.normal(0, 1, (64, opt.latent_dim))))
labels_val = Variable(LongTensor(np.random.randint(0, opt.n_classes, 64)))

if not opt.eval:
    avg_loss_G = []
    avg_loss_D = []
    curr_iter = 0
    for epoch in range(opt.n_epochs):
        iteration = 1
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths with label smoothing
            valid = Variable(FloatTensor(batch_size, 1).fill_(0.9), requires_grad=False)
            fake_imgs = Variable(FloatTensor(batch_size, 1).fill_(0.1), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels.squeeze())) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake_imgs) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.squeeze().data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward()
            optimizer_D.step()

            loss_G.append(g_loss.item())
            loss_D.append(d_loss.item())

            if iteration % opt.print_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (epoch + 1, opt.n_epochs, i + 1, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
                )
            iteration += 1

            curr_iter = epoch * len(dataloader) + i
            if curr_iter % opt.sample_interval == 0:
                fake_gen = sample_val_image(z=z_val, labels=labels_val, n_row=int(floor(sqrt(64))), iter_done=curr_iter)

        avg_loss_G.append(sum(loss_G) / (curr_iter + 1))
        avg_loss_D.append(sum(loss_D) / (curr_iter + 1))
        compute_metrics(real=next(iter(dataloader)), fakes=fake_gen, toPath=os.path.join("output"))

        if epoch % opt.save_model_interval == 0:
            os.makedirs(os.path.join('output', 'ckpt'), exist_ok=True)
            save_model(generator, optimizer_G, epoch, os.path.join('output', 'ckpt', f'G_{epoch:04}.pth'))
            save_model(discriminator, optimizer_D, epoch, os.path.join('output', 'ckpt', f'D_{epoch:04}.pth'))

    loss_plot(G_losses=avg_loss_G, D_losses=avg_loss_D, toPath=os.path.join("output"))
    save_model(generator, optimizer_G, os.path.join('output', 'ckpt', opt.n_epochs, f'G_{opt.n_epochs:04}.pth'))
    save_model(discriminator, optimizer_D, os.path.join('output', 'ckpt', opt.n_epochs, f'D_{opt.n_epochs:04}.pth'))

sample_class_images(nImgs=300, ofst=0)
# sample_class_images(nImgs=300, ofst=300)
# sample_class_images(nImgs=300, ofst=600)
# sample_class_images(nImgs=300, ofst=900)

