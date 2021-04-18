# EfficientNet model
## Description
This repository includes everything related to the melanoma classification model, from the model itself to the preprocessing 
of the dataset and the training/evaluation loops. The goal of this part of the project is to fine-tune our model architecture and
to provide the tools to establish wether the synthetic imaging approach through GAN was eficient or not.

## Implementation

### Network architecture
To be able to perform the melanoma classification task, we considered some pretrained networks like VGG and ResNet. However, 
we decided to use a state-of-the-art approach by implementing a fine-tuned EfficientNet model, whose accuracy/number of parameters
ratio is much higher and efficient than the other two. This is achieved by scaling the model, using coefficients so the structure
throughout the convolutional layers is the same except it gets modified by the coefficients. 

To implement this architecture we used [lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch) model, which adapts from tensorflow's [original implementation](https://github.com/marketplace).

<img src="https://user-images.githubusercontent.com/37978771/115120986-b6828680-9fb0-11eb-94d7-94cbc80af7ac.png" width="500">

EfficientNet includes batch normalization by the end of every Convolutional Block. This provides the network with a stabilizing effect for the layers’ internal distributions, diminishing the effect of vanishing and exploding gradient.
Usually, Batch Normalization allows to train a network with relatively high learning rates. However, in this case we are fine-tuning our network, which requires a low LR in order not to overfit.

Although the model weights are pretrained and top layers unfrozen, Batch Normalization running mean and variance must keep updating themselves since the input images characteristics may differ from the pretrained ones.

Given all the EfficientNet models available, we have used model b3 for its number of parameters (12M) and depth are good enough to obtain a proper accuracy without taking too long.
Using b3 architecture, the model takes 30 minutes for each epoch. In addition, a deeper network may cause overfitting to the model due to the dataset's size.

### Metrics

To be able to evaluate and measure the effectiveness of our network, we have implemented the following metrics:
	
* Loss: We are training a binary classifier, so we are using Binary Cross Entropy to calculate the loss between the output of our model and the ground truth.
	
* Accuracy: To know the ratio of correct predictions.

* AUC score: Since we have an initially imbalanced dataset, accuracy is not enough to now whether our network is performing properly. 

In order to control whether our network overfits and prevent it from keeping training and spending resources in GCP, we have set a patience threshold so the network stops early in case it is not improving.

* Patience: 15 epochs.

### Parameters

To train our network, we used the parameters specified in figure #. Some parameters were restricted to our working environment, such is the case of the batch size 
and other were tuned by trial and error.


| Variable | Description | Value |
|     :---:    |     :---:      |     :---:     |
| batch size   | Batch size     | 64    |
| num_epochs     | Number of epochs       | 50      |
| learning_rate     | Learning rate       | 0.0001      |
| frozen_layers     | Frozen layers       | 18      |
| patience     | Patience       | 6      |

In order to accomplish our objective, the following parameters from both the model and the training section were tuned:

*	Data augmentation.

*	Fine-tuning

* 	Training the model using synthetic images

#### Data augmentation

Due to the imbalance of the dataset, the model overfits fast. In order to avoid this, we have implemented some augmentations, using the library albumentations.
The following figure show the model implemented without using augmentations, overfitting right at the beginning. 

<img src="https://user-images.githubusercontent.com/37978771/115122062-3bbc6a00-9fb6-11eb-9c05-519a13fa65cf.png" width="900">

Once the augmentation were applied, the model doesn't overfit anymore.

<img src="https://user-images.githubusercontent.com/37978771/115122075-4f67d080-9fb6-11eb-95f8-451a1ac36325.png" width="900">

In addition to the augmentations, all the images were resized to 128x128 so to adapt the network to the images synthetically generated.


### Fine tuning

To achieve a lower loss and thus, a higher accuracy, pretrained weights have been loaded to the network. The following figures present how the model was fine-tuned and improved just by unfreezing layers. Out of all the different configurations, these three were the ones that provided better results.

Apart from freezing/unfreezing layers, other parameters were modified, such as Batch Normalization momentum, which is set to 0.1 by default in the EfficientNet, and we chanched it to 0.15 to attach more weight to the updated running means an variances. 

| Frozen layers | Train loss | Train acc |  Val loss |  Val acc |
|     :---:    |     :---:      |     :---:     |     :---:     |     :---:     |
| 22   | 0.30     | 0.87    |  0.34   |   0.87    | 
| 19     |0.36       | 0.85      | 0.29 | 0.88  |
| 18     | 0.28       | 0.88      | 0.30 | 0.88 |
| 17     | 0.39      | 0.83      | 0.37 | 0.85 |



22 Convblocks frozen.

<img src="https://user-images.githubusercontent.com/37978771/115137343-e750d300-a025-11eb-897e-809e7fcc7e57.png" width="900">

19 Convblocks frozen

<img src="https://user-images.githubusercontent.com/37978771/115137985-9d69ec00-a029-11eb-8e28-dada80fc546c.png" width="900">

18 Convblocks frozen

<img src="https://user-images.githubusercontent.com/37978771/115138051-110bf900-a02a-11eb-8828-2fb9ee8d68da.png" width="900">

17 Convblocks frozen

<img src="https://user-images.githubusercontent.com/37978771/115138090-40bb0100-a02a-11eb-8446-0b4f886f4012.png" width="900">

The difference of epochs between models is due to the network's early stopping. The moment the model stops improving and patience goes to 0, the network
stops training and saves the best scoring model.
The results obtained show that even when the dataset has been reduced considerably, we have achieved a proper accuracy. Within a bigger dataset, we shall consider
changing the model from b3 to b4 or even b5 and fine tuning again our architecture.

After the fine tuning, the model architecture was the following:

| Description | Value |
|     :---:      |     :---:     |
| Batch size     | 64    |
| Number of epochs       | 50      |
| Learning rate       | 0.0001      |
| Frozen layers       | 18      |
| Patience       | 6      |
| Normalization layer       | Batchnormalization      |
| Resize      | 128     |
| Avg. Time per epoch (min)     | 30     |

### Synthetic images

Now that we have already trained our model successfully, we may feed and train it with the synthetically generated images.  There are different approaches to this.
Firstly, we fed our network with a high number of GAN images, resulting in a balanced training. On the other hand, we may mantain the network imbalanced, by using the same
dataset and adding an extra 10% of synthetic images to study how does the model behaves. For this approach it is important to take into account that Imbalance Dataset Sample does not shuffle the dataset, so this was manually done in the CSV file.

We kept the previously set parameters, and tried both approaches on ACGAN and DCNSGAN.

| Type | Dataset | Train loss | Train acc |  Val loss |  Val acc |
|     :---:    |     :---:    |     :---:      |     :---:     |     :---:     |     :---:     |
| AC Gan   | Balanced     | 0.24     | 0.90    |  0.43   |   0.82    | 
| AC Gan     |Imbalanced       |0.32       | 0.86      | 0.37 | 0.83  |
| DCSN Gan     | Balanced       | 0.33       | 0.86      | 0.43 | 0.80 |
| DCSN Gan     | Imbalanced      | 0.20      | 0.92      | 0.36 | 0.85 |
