import os
import pandas as pd
from PIL import Image
import cv2
from skimage import io
import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision import transforms
from torchvision.utils import save_image


def color_transform(x):
    x = F.adjust_saturation(x, 0.9)
    x = F.adjust_gamma(x, 1.5)
    x = F.adjust_contrast(x, 1.7)
    return x


def corrupted_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return True
    return False


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
            # image = Image.open(image_path)
            # if corrupted_image(image_path):
            #     print(f'Corrupted image: {image_path}')
            image = transforms.ToPILImage()(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        except:
            image_path = os.path.join(self.images_path, self.labels_df.iloc[idx]['dcm_name'] + ".png")
            # image = Image.open(path).convert('RGB')
            # if corrupted_image(image_path):
            #     print(f'Corrupted image: {image_path}')
            image = transforms.ToPILImage()(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

        if self.transform:
            if self.color:
                image = color_transform(image)
            image = self.transform(image)

        if self.is_test:
            return image

        return image, torch.tensor([target], dtype=torch.float32)


def callback_get_label(dataset, idx):
    # Callback function used in imbalanced dataset loader.
    i, target = dataset[idx]
    return int(target)


def get_isic_dataloader(path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = ISICDataset(os.path.join(path, 'tiny_data.csv'), os.path.join(path, 'Images'),
                          transform=transform, color=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # data_loader = DataLoader(dataset, sampler=ImbalancedDatasetSampler(
    #     dataset, callback_get_label=callback_get_label
    # ), batch_size=batch_size)

    return data_loader


if __name__ == '__main__':
    path = '/media/mestecha/Samsung_T5/SAGAN/ISIC-Archive/Data/'
    batch_size = 64

    dl = get_isic_dataloader(path, batch_size)

    img, label = next(iter(dl))
    print(img.shape)
    print(label.shape)
    save_image(img, 'test.png')
