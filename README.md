# AIDL21: GAN-based synthetic medical image augmentation

This repository contains different Generative Adversarial Networks (GANs) implementations to explore methods for synthetic data augmentation . They all are dedicated to Artificial Image Synthesis in the context of Medical imaging data.

Our project aims to display the effectiveness of synthetic  data generation as a form of image augmentation technique to improve the predictive classification performance of a scaled up Convolutional Neural Network (CNN) known as EfficientNet, proposed in [ICML 2019](https://arxiv.org/pdf/1905.11946.pdf). We suggest an augmentation scheme that  is  based  on  a combination  of  standard  image  perturbation  and  synthetic  dermatologic  lesion  generation  using  GAN  for improved skin cancer classification.

### About
Final Project for the UPC [Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310402/postgraduate-course-artificial-intelligence-deep-learning/) 2020-2021 online edition, authored by:

* [Alexis Molina](https://www.linkedin.com/in/alexismolinamr/)
* [Antonio Castaño](https://www.linkedin.com/in/antoniocasblan/)
* [Guillermo Oyarzun](https://www.linkedin.com/in/guillermo-oyarzun-altamirano)
* [Marcos Estecha](https://www.linkedin.com/in/marcos-estecha-07a54a113)

Advised by [Santiago Puch](https://www.linkedin.com/in/santipuch)

## Table of Contents <a name="toc"></a>

- [1. Introduction](#1-intro)
    - [1.1. Motivation](#11-motivation)
    - [1.2. Objectives](#12-milestones)
- [2. Corpora](#2-available-datasets)
    - [2.1. Data Description](#21-eda)
    - [2.2. Pre-processing](#22-preprocessing) 
- [3. Deep Neural Networks Models](#3-dnns)    
    - [3.1. Evaluation Metrics](#31-metrics)
    - [3.2. Generative Adversarial Networks (GANs)](#32-gans)
        - [DC-GAN](DC-SN-GAN)
        - [W-GAN](W-GAN)
        - [AC-GAN](AC-GAN)
        - [SN-GAN](DC-SN-GAN)
    - [3.2. EfficientNet](#32-efficientnet) 
- [4. Environment Requirements](#4-envs)
    - [4.1. Software](#41-software)
    - [4.2. Hardware](#42-hardware)
- [5. Results](#5-results)
- [6. Conclusions](#6-conclusions) 
- [7. Acknowledgements](#7-acknowledgements)
 
## 1. Introduction <a name="1-intro"></a>

Over the last decade Deep Neural Networks have produced unprecedented performance on a number of tasks, given sufficient data. One of the main challenges in the medical imaging domain is how to cope with the small datasets and limited amount of annotated samples, especially when employing supervised machine learning algorithms that require labeled data and larger training examples.

In an attempt to overcome this challenge, the researchers adopted data augmentation schemes, commonly including simple modifications of dataset images such as translation, rotation, flip and scale. However, little additional information can be gained from the diversity provided by these techniques.

On the other hand, since their introduction by [Goodfellowet al.](https://papers.nips.cc/paper/5423-generative-adversarial-nets), Generative Adversarial Networks (GANs) have become the defacto standard for high quality image synthesis. There are two general ways in which GANs have been used in medical imaging. The first is focused on the generative aspect and the second one is on the discriminative aspect. Focusing on the first one, GANs can help in exploring and discovering the underlying structure of training data and learning to generate new images. This property makes GANs very promising in coping with data scarcity and patient privacy.


### 1.1. Motivation <a name="11-motivation"></a>

Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective. The advance of medical imaging informatics and the new AI approaches are leading the efforts to improve melanoma diagnosis. Machine learning algorithms require sufficient data volume for attaining successful results. However, there is inherent problematic in the field of medical imaging where abnormal findings are by definition uncommon. The paucity of annotated data and class imbalance of insufficient variability leads to poor classification performance.

In this project we have explored the possibilities of applying different flavors of GANs given their potential as an augmented method for image representation and classification.


### 1.2. Objectives <a name="12-milestones"></a>

The main purpose of this project is to demonstrate the potential solution to the problem of insufficiency data volume in the medical domain. The proposed solution consists of using GANs for synthetic medical data augmentation for improving a CNN-based classifier's performance. To tackle this task, it can be further broken down into the following sub-objectives:
- Explore, clean and process the data that will be used for training and evaluating the implemented Deep Neural Networks.
- Research, develop, implement and train a classifier model. This classifier will be based on a scaled up CNN whose function will be to detect malign dermathological lesions from the different augmented images.
- Perform classifier performance tests. In order to establish a solid base evaluation model to compare with, there wil be necessary to undertake several experiments for fine tuning appropriately the model to our data.
- Research, develop, implement and train a series of GANs-based models to be able to demonstrate how much we can improve the performance of the classifier.
- Carried out a series of experiments comparing the performance of the classifier using standard augmentation over the training data with respect to the performance obtained using the synthetic data from the differents GANs.
- Draw final conclusions from all the experiments conducted and the different improvements attempted.

## 2. Corpora <a name="2-available-datasets"></a>

For training and testing our models, we have used the dataset provided by the healthcare organization for informatics in medical imaging, the [Society for Imaging Informatics in Medicine (SIIM)](https://siim.org/) joined by the [International Skin Imaging Collaboration (ISIC)](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main). 

After some filtering for rebalancing the missing or non-labeled images and cutting off the excess of benign lesions, we finish with a total number of 22,922 images split in Train (16,045 observations) and Test (6,877 observations).

<p align="center">
  <img src="Data/images-sagan/data-tree-background.png">
</p>

### 2.1. Data Description  <a name="21-eda"></a>

Through the API available in the ISIC home page we have been able to download all the images collection with its descriptions associated. The whole database is about 110 gigabytes (GB). The format of the colour images is both JPEG and PNG with a high variety of resolution sizes. Each image has a corresponding JSON-based description file with the image metadata information. From these metadata files we have been conducted a quick Exploratory Data Analysis (EDA) to acquire more awareness of how distributed it is. Initially, there were 27 metadata fields from which we later filtered out and kept only 8 of them. Some meaningful classes worthy to mention are the dcm_name field which identifies the image associated; the benign_malignant class from which we later classify; and finally the diagnosis class which details the diagnosis of the dermatological image lesion is referred to.

<p align="center">
  <img width="90%" height="90%" src="Data/images-sagan/diagnosis.png"/>
</p>


As we mention, we carried out a gently Data Wrangling to obtain some useful insight about the images data handling. From this procedure we have verified that there is a very high target class imbalance which need to take in consideration when modeling.

<p align="center">
  <img width="70%" height="70%" src="Data/images-sagan/sex.png" /> 
  <img width="70%" height="70%" src="Data/images-sagan/age.png" />
</p>

### 2.2. Pre-processing  <a name="22-preprocessing"></a> 

To be able to feed our dataset into the classifier, we must first condition it to the network and to our resource’s limitations.

#### CSV files

As mentioned before, every image comes with a JSON files with relevant information regarding the patient and the skin spot. This files were all put into a CSV file where each column stands for a field from the JSON file. 
In addition to the initial fields, we added “dcm_name” to store the name of the image the data belongs to, and “target” which is set to 0 if the skin spot is benign and to 1 in case it is malignant.

#### Dataset reduction

We reduced the dataset to 5K images to diminish the training cost, keeping the malignant/benign ratio so the results can be escalated to the complete dataset.

Given the size of the image’s directory, we modified it so it only contained the images that were going to be fed into the network, in order not to use more storage than necessary in GCP. We did this through a series of automated functions in python.

#### Data augmentation

We applied several transformations to avoid overfitting when training our network with the reduced dataset. To do that, we have used albumentations’ library due to the large number of augmentations it has available.

The images input size was variable and with a bigger resolution, we resized them to 128x128 to fit the synthetically generated images. Furthermore, we applied techniques involving changing the image’s contrast and brightness scale and rotations. We also normalized its mean and standard deviation and finally we converted the images ton tensors so they can be feed into our model.

Here are some of the images before and after applying the transformations.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37978771/115211333-c0ff6600-a0ff-11eb-8ee7-fbd3dee7f332.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/37978771/115211501-e68c6f80-a0ff-11eb-878f-b2b58ff11400.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/37978771/115211526-ef7d4100-a0ff-11eb-9d35-5751703abaca.png">
</p>

## 3. Deep Neural Networks Models <a name="3-dnns"></a>

Under this section we present all the GAN versions implemented. We approach to the proble with our own variation of implementation of the technique and methodology first introduced in [Frid-Adar et al.](https://arxiv.org/abs/1803.01229) in 2018.

### 3.1. Evaluation Metrics  <a name="31-metrics"></a>

Since we lack from any medical expertise for assessing the quality of the generated images, we have implemented several metrics to measure traits of our output pictures.

- #### Peak Signal-to-Noise Ratio (PSNR)

This metric is used to measure the quality of a given image (noise), which underwent some transformation, compared to the its original (signal). In our case, the original picture is the real batch of images feeded into our network and the noise is represented by a given generated image.

- #### Structural Similarity (SSIM)

SSIM aims to predict the perceived the quality of a digital image. It is a perception based model that computes the degradation in an image comparison as in the preceived change in the structural information. This metric captures the perceptual changes in traits such as luminance and contrast.

- #### Multi-Scale Gradient Magnitude Similarity Deviation (MS GMSD)

MS-GMSD works on a similar version as SSIM, but it also accounts for different scales for computing luminance and incorporates chromatic distortion support.

- #### Mean Deviation Similarity Index (MDSI)

MDSI computes the joint similarity map of two chromatic channels through standard deviation pooling, which serves as an estimate of color changes. 

- #### Haar Perceptural Similarity Index (HaarPSI)

HaarPSI works on the Haar wavelet decomposition and assesses local similarities between two images and the relative importance of those local regions. This metric is the current state-of-the-art as for the agreement with human opinion on image quality. 

- #### Measure Assessment

Measure | Bar | 
:------: | :------:|
PSNR   | Context dependant, generally the higher the better.  | 
SSIM   |  Ranges from 0 to 1, being 1 the best value.     | 
MS-GMSD |  Ranges from 0 to 1, being 1 the best value.    |  
MDSI   |   Ranges from 0 to inf, being 0 the best value.    |
HaarPSI |   Ranges from 0 to 1, being 1 the best value.   |

### 3.2. Generative Adversarial Networks (GANs)  <a name="32-gans"></a>

- [DC-GAN](DC-SN-GAN)<a name="DC-GAN"></a>

A DC-GAN is a specific flavor of GAN dedicated to image generation. The architecture consists on a _Generator_ and a _Discriminator_ built upon four 2d convolutional layers. It was first described by _Radford et. al._ in this [paper](https://arxiv.org/pdf/1511.06434.pdf). The _Discriminator_ in build out of strided convolutions, batch normalization layers and uses Leaky Relu activations. Originally, the input size of the images is 64 and it is already set to process color images (3x64x64). The _Generator_ differs from the _Discriminator_ in the convolutional layers, which are transposed. It has as an input a random vector sampled from a normal distribution which will be transformed by adversarial training into an RGB image of the selected shape.

- [W-GAN](W-GAN)<a name="W-GAN"></a>

The Wasserstein GAN (W-GAN) is a variant to the traditional implementation of the generative adversarial networks for improving the training phase. The adversarial loss aims at finding the Nash equilibrium, which in practice is difficult to achieve and it may result in model oscillation and mode collapse. The idea behind the W-GAN is to stabilize the model training by focusing on minimizing an approximation of the Earth-Mover's distance (EM) rather than the Jensen-Shannon divergence as in the original GAN formulation. Moreover, the discriminator is changed by a critic that scores the realness or fakeness of a given image. 
By doing so, the loss of the discriminator seems to relate with the quality of the images generated by the model. 


- [AC-GAN](AC-GAN)<a name="AC-GAN"></a>

Conditional  GANs  are  an  extension  of  the  GAN model,  that  enable  the  model  to  be  conditioned  on  external information to improve the quality of the generated samples. It changes the discriminator to predict the class label of a given image rather than receive it as input. It has the effect of stabilizing the training process and allowing the generation of large high-quality images whilst learning a representation in the latent space that is independent of the class label.

- [SN-GAN](DC-SN-GAN)<a name="SN-GAN"></a>

The SN-GAN is identical to DC-GAN but implements _Spectral Normalization_ to deal with the issue of exploding gradients in the _Discriminator_. 

Spectral normalization is a weight regularization technique with is applied to the GAN's _Discriminator_ to solve the issue of exploding gradients. Is works stabilizing the training process of the _Discriminator_ through a rescaling the weights of the convolution layers using a norm (spectral norm) which is calculated using the power iteration method. The method is triggered right before the _forward()_ function call.

In more detail, spectral normalization deals with Lipschitz constant as its only hyper-paramenter. This constant refers to a regularization property of continuous functions which bounds its values. More precisely, the Lipschitz constant equals the maximum value of the derivatives of the function. In out particular case, since the activation function is a LeakyRelu, this constant takes the value 1. 

Spectral normalization controls this parameter in the discriminator by bounding it through the spectral norm. The Lipschitz norm ![g_lip](https://user-images.githubusercontent.com/48655676/115139673-7f08ee00-a033-11eb-9495-79dfe24bbc0c.gif) is equivalent to the superior bound of the gradient of the layer ![sup_g](https://user-images.githubusercontent.com/48655676/115155321-b0a7a680-a07f-11eb-9997-861feab751e9.gif), where ![sigma_a](https://user-images.githubusercontent.com/48655676/115139724-c0010280-a033-11eb-9030-70b1c31bee33.gif) is defined as the spectral norm of the matrix A. That gives,

![big_eq](https://user-images.githubusercontent.com/48655676/115139854-a7451c80-a034-11eb-8be2-53549d5fa4af.gif),

which is the largest singuler value of A and **h** is the linear layer.

With the above definition of a linear layer, when passing weights through as ![pass_W](https://user-images.githubusercontent.com/48655676/115139979-49650480-a035-11eb-8527-394564873320.gif), the norm of the layer is defined as,

![big_all_eq](https://user-images.githubusercontent.com/48655676/115140094-d740ef80-a035-11eb-8944-7074fec8c592.gif).

Therefore, spectral normalization of a given passing weight **W** normalizes the weight of each layer and thus, the whole network, mitigaiting explosion gradient problems.

Some works refer to DCGANs that implement spectral normalization as SNGANs, which is also done in this work. SNGAN with the best parameters and implementations described in the respective folder was the one used for the image generation.

### 3.2. EfficientNet 

EfficientNet is a convolutional neuronal network that provides and optimal relation between the number of parameters of a network and its efficiency. It accomplishes this using a mobile size baseline architecture, EfficientNet-b0, and performing a compound scaling to increase its size by replicating the baseline to different MBConv blocks. This network was firstly presented by Google in this [paper](https://arxiv.org/abs/1905.11946). Up to this day, EfficientNet has been scaled from its baseline that contains 5.3M parameters to b7, with 66M parameters.

## 4. Environment Requirements <a name="4-envs"></a>
### 4.1. Software  <a name="41-software"></a>

We selected PyTorch as framwork for our scientific computing package to develop our project. Regarding the image transformations used for standard augmentations, we have selected both Torchvision and Albumentation packages. To approach the imbalance dataset issue we used Torchsampler’s Imbalanced Dataset Sampler library. For visualization, we also used both classical Pyplot and Seaborn packages. For the dataset preprocessing, we made use of the modules available in Scikit-Learn library. Some of the GANs-based implementations developed make use of YAML as the preferred language for defining its configuration parameters files. Lastly, the package Pytorch Image Quality Assessment (PIQA) is used to generate the metrics that evaluate the quality of the synthetic images. And finally, for the model we made use of lukemelas EfficientNet architecture. 
 

### 4.2. Hardware  <a name="42-hardware"></a> 

The GANs were trained using Google Colab. This work environment provided us an easy way to work in teams and to access to GPUs. The Classifier also started as a Google Colab project, however, due to its high computing demands, we were forced to port it to Google Cloud to avoid the time limit of the Colab.  

- **Google Cloud Platform**

To launch the instance we used Cloud Deep Learning VM Image. We created a Linux VM, with a n1-highmem-2 machine and 1 NVIDIA Tesla k80 GPU. In addition to the Instance, we created a bucket in order to upload the Images from the different datasets (the reduced one, and the ones with the GANs) to then move them to the persistent disk. We firstly implemented our code using the Jupyter Notebook function, but since the training process took a long time and Google Cloud Shell would eventually log off, we switched to SSH and launched our script from the terminal.


## 5. Results <a name="5-results"></a>

Architecture | PSNR | SSIM |MS-GMSD |MDSI |HaarPSI |
:------: | :------:| :------:| :------:| :------:|:------:|
DC-GAN |  12.92  | 0.35   | 0.29   |  0.45 | 0.39 |
AC-GAN |   12.17 |  0.32  | 0.27   | 0.49  | 0.31 |
AC-GAN 128 |  11.81  |  0.39     |    0.25   |   0.47    |   0.31     |
W-GAN  |  12.63  |    0.30    |   0.27    |  0.50   |  0.35    |
W-GAN 128 | 12.27 | 0.31 | 0.14 | 0.44 | 0.40 |
SN-GAN |  12.21  |   0.21   |   0.26   |   0.49  |  **0.41**  |  
SN-GAN 128 |  12.18  |   0.24   |   0.15   |   0.52  |  **0.45**  |  


- #### DC-GAN 64x64

![skin_lesions_700_twick](https://user-images.githubusercontent.com/48655676/110391353-a1d4d980-8067-11eb-9eca-4f458fffd203.png)

- #### SN-GAN 64x64

![skin_lesions_800_twick3_sn](https://user-images.githubusercontent.com/48655676/110391188-70f4a480-8067-11eb-9d8b-ce150ef7797b.png)

- #### W-GAN 64x64

![WGAN-64](https://github.com/mestecha/AIDL21-SAGAN/blob/main/W-GAN/wgan-images/gen_wgan_64.png)

- #### AC-GAN 64x64

<p align="center">
  <img src="AC-GAN/acgan-images/64-64-600.png">
</p>

- #### SN-GAN 128x128

![SN_final](https://user-images.githubusercontent.com/48655676/114686469-18be5b80-9d13-11eb-80ae-aa53aa7061e6.png)

- #### W-GAN 128x128

![WGAN-128](https://github.com/mestecha/AIDL21-SAGAN/blob/main/W-GAN/wgan-images/gen_wgan_128.png)

- #### AC-GAN 128x128

<p align="center">
  <img width="50%" height="50%" src="AC-GAN/acgan-images/grid-128-64-600.jpg">
</p>

## 6. Conclusions <a name="6-conclusions"></a>

* **Training GANs** proved to be a **hard task**.
    * Requires a vest amount of **resources**.
    * **Training process** is **not straightforward**.

* **SNGAN outperformed** DCGAN, ACGAN and WGAN.
    * Even though **after huge amount of experimentation** metrics were still far from initial goal.

* On the **GAN training parametrization**:
    * **Batch size** is among the most relevant parameters to reduce training times and improve image quality. The reasonale behind this effect could come from the _Discriminator_ having less examples to generalize its classification of real and fake images.
    * The number of **training epochs** also affects the quality of the generated images. Longer traning usually ends up producing better images even though the two losses did not converge.
    * Another parameter tweak that comes handy when training these architectures is the **size of the latent vector**. With higher sizes the quality of images did not improve, but it did reduce the training time.
    * **Label smoothing** has another critical change that was done in our GANs. It did produce better images and also it did stabilize the training. Mathematically, the class probabilities of the discriminator are, in general, lower when using this technique and thus, it balances the performance of the _Discriminator_ and the _Generator_.
    * **Spectral normalization**, which deals with exploding gradients, did also increase the quality of the generated images. It gave out a new architecture purely based on a DCGAN.
    * **Different learning rates**, more specifically with higher values for the _Discriminator_, did stabilize training and also increased the quality of the images. The explanation behind this behavior is that setting bigger steps for optimizing the loss function of the _Discriminator_  makes this agent more imprecise at the classification task whereas the smaller steps for the _Generator_ gives it a more precise approach to image generation.

* **Different metrics** are sensible to **different aspects** of image quality.
    * Best practice to **use a set** of them to assess the generated images.
    * **Include a metric** based on **human perception**.
 
* Good results for a **lack** of **resources**.
    * Fine-tuned **EfficientNet** achieves **high accuracy** with **reduced dataset**.
    * Dataset with **sysnthetic images** does **not improve accuracy**.
    * **Balanced dataset** with **synthetic images** and no augmentations achieves **good results**.

## 7. Acknowledgements <a name="7-acknowledgements"></a>

We would like to thank all the staff from the Prostgraduate Course on Artificial Intelligence with Deep Learning for all the effort and care that they took and showed preparing the materials and lecture which provided us with the tools to build this project.

We would like to give a special thanks to Santi Puch, our advisor, who provided very helpful advise and spent numerous hours revising our material and searching tricks and tips for our project.

Finally, we would also like to highlight the very useful work of François Rozet, the creator of PIQA. It really helped us to implement an standard of metrics in a really straightforward and clear manner.
