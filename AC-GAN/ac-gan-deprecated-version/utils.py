import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import numpy as np
# torch.manual_seed(0) # Set for our testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)

def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


def denorm(img):
    """ Denormalize input image tensor. (From [0,1] -> [-1,1])

    Args:
        img: input image tensor.
    """

    output = img / 2 + 0.5
    return output.clamp(0, 1)


def save_model(model, optimizer, file_path):
    """ Save model checkpoints. """

    state = {'model': model.state_dict(),
             'optim': optimizer.state_dict(),
             }
    torch.save(state, file_path)
    return


def load_model(model, optimizer, file_path):
    """ Load previous checkpoints. """

    prev_state = torch.load(file_path)

    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])

    return model, optimizer