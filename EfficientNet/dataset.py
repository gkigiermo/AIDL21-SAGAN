from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):

    def __init__(self, images_path, description_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.transform = transform
        self.clinical_df = pd.read_csv(description_path, sep = ';')

    def __len__(self):
        return len(self.clinical_df)

    def __getitem__(self, idx):
        id, benign_malign = self.clinical_df.loc[idx, ['dcm_name', 'target']]
        path = os.path.join(self.images_path, id + ".jpeg")
        try:
            sample = Image.open(path)

        except:
            path = os.path.join(self.images_path, id + ".png")  # en caso de que la imagen sea .png uy tenga 4 canales
            sample = Image.open(path).convert('RGB')

        else:
            pass
        if self.transform:
            sample = self.transform(**{"image": np.array(sample)})["image"]  # for albumentations
        return sample, benign_malign

    def get_dcm_name(self):
        dcm = []
        for name in self.clinical_df['dcm_name']:
            dcm.append(name)
        return dcm


def descript_to_csv(description_path,njasonfiles,csv_name):  # converts JSON files to 1 CSV
    for i in range(njasonfiles):  # 10999
        zerostr = ""
        id = "ISIC_" + zerostr.zfill(7 - len(str(i))) + str(i)  # max numero de cifras menos ndigitos

        path = os.path.join(description_path, id)
        df = pd.read_json(path)
        dict = df.loc['clinical', 'meta']
        df = pd.DataFrame.from_dict(dict, orient = "index").T
        df = df.loc[:, ['benign_malignant']]
        df.insert(0, "id", id, True)
        if i == 0:
            df1 = df
        else:
            df1 = pd.concat([df, df1], axis = 0)
    df1 = df1.reset_index(drop = True)
    df1['benign_malignant'] = df1['benign_malignant'].map({'benign': 0, 'malignant': 1})
    df1 = df1.loc[df1['benign_malignant'].isin([0, 1]), :]
    df1.to_csv(csv_name, index = False,
               sep = ';')


def reduce_dataset(path, file,train_name,val_name, test_name):
    df = pd.read_csv(os.path.join(path, file))
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size = 0.1, stratify = y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.3, stratify = y_train)
    x_train.to_csv(train_name, index = False, sep = ';')
    x_val.to_csv(val_name, index = False, sep = ';')
    x_test.to_csv(test_name, index = False, sep = ';')

#reduce_dataset('Data/classifier_reduced_marcos','train_m_reduced.csv', "train_mm_reduced.csv", "val_m_reduced.csv", "test_m_reduced.csv")

