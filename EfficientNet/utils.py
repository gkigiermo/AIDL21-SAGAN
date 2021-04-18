import torch
import os, shutil
from dataset import *
import pandas
from os import listdir
from os.path import isfile, join


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc


def save_model(model, path):
    torch.save(model.state_dict(), path)


def remove_files(path, list): #Remove images that are not in the list
    for filename in os.listdir(path):
        if filename.endswith('.jpeg'):
            filename = filename[:-5]
            extension = '.jpeg'
        elif filename.endswith('.png'):
            filename = filename[:-4]
            extension = '.png'
        if filename not in list:
            full_file_path = os.path.join(path, filename + extension)
            os.remove(full_file_path)

def copy_paste_images(origin, destiny, list): #Copy images that are in the list and paste them into destiny folder
    for name in list:
        try:
            full_file_path = os.path.join(origin, name + '.jpeg')
            shutil.copy(full_file_path, destiny)
        except FileNotFoundError:
            full_file_path = os.path.join(origin, name + '.png')
            shutil.copy(full_file_path, destiny)

def delete_images_from_dataset(train_dataset,val_dataset,test_dataset, data_transforms,img_path,img_reduced_path, dset_path):
##Crea una carpeta con sólo las imágenes que se encuentren en los dataset
    complete_dset = []
    complete_dset = [*train_dataset.get_dcm_name(), *val_dataset.get_dcm_name(), *test_dataset.get_dcm_name()]
    remove_files(img_path,complete_dset)
    dataset = MyDataset(img_path, dset_path)
    dset_dcm = dataset.get_dcm_name()
    copy_paste_images(img_path, img_reduced_path, dset_dcm)


def img_folder_to_csv_list(folder_path, save_csv_path): #lists all the files in a folder and saves it in a csv file
    path = folder_path
    images = [f for f in listdir(path) if isfile(join(path, f))]
    df = pandas.DataFrame(data = {"col1": images})
    df.to_csv(save_csv_path, sep = ';', index = False)
'''
from model import *
model = MyModel()
checkpoint = model.load_state_dict(torch.load('models saved/model09041.pt'))
print(model)
'''
import matplotlib.pyplot as plt
def plot_df_dset(df_path):

    df = pd.read_csv(df_path, sep = ';')
    train_loss_history = df['train loss'].str.replace(',', '.').to_numpy()
    train_loss_history= [float(i) for i in train_loss_history]
    train_acc_history = df['train acc'].str.replace(',', '.').to_numpy()
    train_acc_history = [float(i) for i in train_acc_history]
    val_acc_history = df['val acc'].str.replace(',', '.').to_numpy()
    val_acc_history = [float(i) for i in val_acc_history]
    val_loss_history = df['val loss'].str.replace(',', '.').to_numpy()
    val_loss_history = [float(i) for i in val_loss_history]

    fig = plt.figure(figsize = (20, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)



    ax1.plot(train_loss_history, label = 'Training Loss')
    ax1.plot(val_loss_history, label = 'Validation Loss')
    ax1.set_title("Losses")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_acc_history, label = 'Training accuracy')
    ax2.plot(val_acc_history, label = 'Validation Accuracy')
    ax2.set_title("Accuracies")
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()
plot_df_dset('./results_11041.csv')

