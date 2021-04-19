import os
import numpy as np
import random

import torch.optim as optim

import torch
from torchvision.utils import save_image
from torch.autograd import Variable

from network import Generator, Discriminator
from dataset import get_isic_dataloader
from utils import weights_init_normal, compute_acc, denorm, save_model

train_path = os.path.dirname(os.path.realpath(__file__))


class ACGANTrain:
    def __init__(self, config):
        self.cuda = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda') if self.cuda else torch.device('cpu')
        print(f'Using {self.device}')

        # Define general configuration
        self.config = config

        self.manualSeed = self.config['manual_seed']
        if self.manualSeed is None:
            self.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.manualSeed)
        random.seed(self.manualSeed)
        torch.manual_seed(self.manualSeed)
        if self.cuda:
            torch.cuda.manual_seed_all(self.manualSeed)

        self.run_path = config['run_path']
        self.path = config['data_path']

        self.lr = config['optima']['lr']
        self.b1 = config['optima']['b1']
        self.b2 = config['optima']['b2']

        self.batch_size = config['batch_size']
        self.epochs = config['epochs']

        # self.log_interval = config['log_interval']
        self.print_interval = config['print_interval']
        self.sample_interval = config['sample_interval']
        self.save_model_interval = config['save_model_interval']

        # Configure data loader
        self.dataloader = get_isic_dataloader(self.path, self.batch_size)
        self.steps_per_epoch = int(np.ceil(self.dataloader.dataset.__len__() * 1.0 / self.batch_size))
        print(f'Training images: {self.dataloader.dataset.__len__()}')

        # Get network parameters
        self.image_size = config['model']['image_size']
        self.noise_dim = config['model']['noise_dim']
        self.class_dim = config['model']['class_dim']

        # Initialize generator and discriminator
        self.generator = Generator(self.class_dim, self.noise_dim, self.image_size).to(self.device)
        self.discriminator = Discriminator(self.class_dim, self.image_size).to(self.device)

        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        # Loss functions
        self.adversarial_criterion = torch.nn.BCELoss().to(self.device)  # discriminator criterion
        self.auxiliary_criterion = torch.nn.NLLLoss().to(self.device)  # classifier criterion

        # Optimizers
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Define Tensor types
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor

        # Tensor placeholders
        self.input = self.FloatTensor(self.batch_size, 3, self.image_size, self.image_size)
        self.noise = self.FloatTensor(self.batch_size, self.noise_dim, 1, 1)
        self.eval_noise = self.FloatTensor(self.batch_size, self.noise_dim, 1, 1).normal_(0, 1)
        self.dis_label = self.FloatTensor(self.batch_size)
        self.aux_label = self.LongTensor(self.batch_size)
        self.real_label = 1
        self.fake_label = 0

        # Define variables
        self.input = Variable(self.input)
        self.noise = Variable(self.noise)
        self.eval_noise = Variable(self.eval_noise)
        self.dis_label = Variable(self.dis_label)
        self.aux_label = Variable(self.aux_label)

        # Noise for evaluation
        self.eval_noise_ = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
        self.eval_label = np.random.randint(0, self.class_dim, self.batch_size)
        self.eval_onehot = np.zeros((self.batch_size, self.class_dim))
        self.eval_onehot[np.arange(self.batch_size), self.eval_label] = 1
        self.eval_noise_[np.arange(self.batch_size), :self.class_dim] = self.eval_onehot[np.arange(self.batch_size)]
        self.eval_noise_ = (torch.from_numpy(self.eval_noise_))
        self.eval_noise.data.copy_(self.eval_noise_.view(self.batch_size, self.noise_dim, 1, 1))

    def sample_image(self, epoch, real, n_row, batches_done):
        os.makedirs(os.path.join(self.run_path, "images"), exist_ok=True)
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        save_image(
            real,
            f'{os.path.join(self.run_path, "images")}/real_samples.png'
        )
        # print(f'Label for eval = {self.eval_label}')
        print(f'Sampling an image...')
        fake = self.generator(self.eval_noise)
        save_image(
            fake.data,
            f'{os.path.join(self.run_path, "images")}/fake_samples_epoch_{epoch:03}_batch_{batches_done:06}.png'
        )

    def train(self):
        avg_loss_D = 0.0
        avg_loss_G = 0.0
        avg_loss_A = 0.0

        for epoch in range(1, self.epochs + 1):
            iteration = 1
            for i, (image, label) in enumerate(self.dataloader):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with real
                self.discriminator.zero_grad()
                image, label = image.to(self.device), label.to(self.device)
                batch_size = image.shape[0]

                # Adversarial ground truths
                self.input.data.resize_as_(image).copy_(image)
                self.dis_label.data.resize_(batch_size).fill_(self.real_label)
                self.aux_label.data.resize_(batch_size).copy_(label.squeeze())
                dis_output, aux_output = self.discriminator(self.input)

                dis_errD_real = self.adversarial_criterion(dis_output.squeeze(), self.dis_label)
                aux_errD_real = self.auxiliary_criterion(aux_output, self.aux_label)

                errD_real = dis_errD_real + aux_errD_real
                errD_real.backward()
                D_x = dis_output.data.mean()

                # Compute the current classification accuracy
                accuracy = compute_acc(aux_output, self.aux_label)

                # Train with fake
                self.noise.data.resize_(batch_size, self.noise_dim, 1, 1).normal_(0, 1)
                label = np.random.randint(0, self.class_dim, batch_size)
                noise_ = np.random.normal(0, 1, (batch_size, self.noise_dim))
                class_onehot = np.zeros((batch_size, self.class_dim))
                class_onehot[np.arange(batch_size), label] = 1
                noise_[np.arange(batch_size), :self.class_dim] = class_onehot[np.arange(batch_size)]
                noise_ = (torch.from_numpy(noise_))
                self.noise.data.copy_(noise_.view(batch_size, self.noise_dim, 1, 1))
                self.aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

                fake = self.generator(self.noise)

                self.dis_label.data.fill_(self.fake_label)
                dis_output, aux_output = self.discriminator(fake.detach())

                dis_errD_fake = self.adversarial_criterion(dis_output, self.dis_label.unsqueeze(1))
                aux_errD_fake = self.auxiliary_criterion(aux_output, self.aux_label)

                errD_fake = dis_errD_fake + aux_errD_fake
                errD_fake.backward()
                D_G_z1 = dis_output.data.mean()

                errD = errD_real + errD_fake
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.zero_grad()

                self.dis_label.data.fill_(self.real_label)  # fake labels are real for generator cost
                dis_output, aux_output = self.discriminator(fake)

                dis_errG = self.adversarial_criterion(dis_output, self.dis_label.unsqueeze(1))
                aux_errG = self.auxiliary_criterion(aux_output, self.aux_label)

                errG = dis_errG + aux_errG
                errG.backward()
                D_G_z2 = dis_output.data.mean()

                self.optimizerG.step()

                # compute the average loss
                batches_done = (epoch - 1) * len(self.dataloader) + i
                all_loss_G = avg_loss_G * batches_done
                all_loss_D = avg_loss_D * batches_done
                all_loss_A = avg_loss_A * batches_done
                all_loss_G += errG.item()
                all_loss_D += errD.item()
                all_loss_A += accuracy
                avg_loss_G = all_loss_G / (batches_done + 1)
                avg_loss_D = all_loss_D / (batches_done + 1)
                avg_loss_A = all_loss_A / (batches_done + 1)

                if iteration % self.print_interval == 0:
                    print(
                        f'Epochs: [{epoch}/{self.epochs}] Batch: [{i+1}/{len(self.dataloader)}] || '
                        f'Discriminator loss: {errD.item():.4f} ({avg_loss_D:.4f}) || '
                        f'Generator loss: {errG.item():.4f} ({avg_loss_G:.4f}) || '
                        f'D(x): {D_x:.4f} || D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f} || '
                        f'Acc: {accuracy:.4f} ({avg_loss_A:.4f})'
                    )

                iteration += 1

                if batches_done % self.sample_interval == 0:
                    self.sample_image(epoch=epoch, real=image, n_row=10, batches_done=batches_done)

            if epoch % self.save_model_interval == 0:
                os.makedirs(os.path.join(self.run_path, 'ckpt'), exist_ok=True)
                save_model(self.generator, self.optimizerG, os.path.join(self.run_path, 'ckpt', f'G_{epoch:03}.pth'))
                save_model(self.discriminator, self.optimizerD,
                           os.path.join(self.run_path, 'ckpt', f'D_{epoch:03}.pth'))


if __name__ == '__main__':
    config = {
        'run_path': './runs/run_01',
        'data_path': '/media/mestecha/Samsung_T5/SAGAN/ISIC-Archive/Data/',
        'optima': {
            'lr': 2e-4,
            'b1': 0.5,
            'b2': 0.999
        },
        'n_class': 2,
        'batch_size': 64,
        'epochs': 300,
        'print_interval': 5,
        'log_interval': 5,
        'sample_interval': 100,
        'save_model_interval': 10,
        'model': {
            'image_size': 64,
            'noise_dim': 100,
            'class_dim': 2
        }

    }
    my_model = ACGANTrain(config)
    my_model.train()
