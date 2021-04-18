import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models import vgg16, resnet50
from efficientnet_pytorch import EfficientNet


class MyModel(nn.Module):

    def __init__(self, frozen_layers=None, h=None, trained_features=None):
        super().__init__()
        self.net_type = trained_features


        if trained_features == "effnet":

            self.model = EfficientNet.from_pretrained("efficientnet-b3", num_classes = 1, include_top = False,
                                                in_channels = 3)

            #Freeze al blocks except for the batchnorm modules and change batchnorm momentum
            for m in self.model.children():
                for param in m.parameters():
                    if isinstance(m, nn.BatchNorm2d):
                        m.momentum = 0.15
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            #Unfreeze convblocks from 18 onwards
            for block in nn.ModuleList(list(self.model.children())[2])[frozen_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

            #Unfreeze batchnorm and change momentum inside the Convblock
            for block in nn.ModuleList(list(self.model.children())[2])[:]:
                for layer in block.children():
                    if isinstance(layer, nn.BatchNorm2d):
                        for param in layer.parameters():
                            param.requires_grad = True
                        layer.momentum = 0.15

            print(self.model)

        self.classifier = nn.Sequential(
            nn.Linear(1536, 1, bias = True),  # b1 1280 b2 1408 b5 2048 b3 1536
            nn.Sigmoid()
        )
        self.batchnorm = nn.BatchNorm2d(1536, eps = 0.001, momentum = 0.15, affine = True,
                                        track_running_stats = True)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p = 0.3)


        pass

    def forward(self, x):
        if self.net_type is None:
            # x = self.features(x)
            # rint('Features Image shape:', x.shape)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = torch.flatten(x, start_dim = 1)

            x = self.fc1(x)
            x = self.fc2(x)


        elif self.net_type == 'effnet':
            x = self.model(x)
            x = self.batchnorm(x)
            x = self.avg_pooling(x)
            x = torch.flatten(x, start_dim = 1)
            x = self.drop(x)
            x = self.classifier(x)
        return x
