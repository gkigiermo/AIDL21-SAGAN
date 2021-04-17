# AIDL21: GAN-based synthetic medical image augmentation

This repository contains different Generative Adversarial Networks (GANs) implementations to explore methods for synthetic data augmentation . They all are dedicated to Artificial Image Synthesis in the context of Medical imaging data.

Our project aims to display the effectiveness of synthetic  data generation as a form of image augmentation technique to improve the predictive classification performance of a scaled up Convolutional Neural Network (CNN) known as EfficientNet, proposed in [ICML 2019](https://arxiv.org/pdf/1905.11946.pdf). Therefore we suggest an augmentation scheme that  is  based  on  combination  of  standard  image  perturbation  and  synthetic  dermatologic  lesion  generation  using  GAN  for improved skin cancer classification.

### About
Final Project for the UPC [Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310402/postgraduate-course-artificial-intelligence-deep-learning/) 2020-2021 online edition, authored by:

* [Alexis Molina](https://www.linkedin.com/in/alexismolinamr/)
* [Antonio Castaño](https://www.linkedin.com/in/antoniocasblan/)
* [Guillermo Oyarzun](http://www.cttc.upc.edu/?q=user/93)
* [Marcos Estecha](https://www.linkedin.com/in/marcos-estecha-07a54a113)

Advised by [Santiago Puch](https://www.linkedin.com/in/eva-mohedano-261b6889/)

## Table of Contents <a name="toc"></a>

- [1. Introduction](#1-intro)
    - [1.1. Motivation](#11-motivation)
    - [1.2. Objectives](#12-milestones)
- [2. Corpora](#2-available-datasets)
    - [2.1. Data Description](#21-eda)
    - [2.2. Pre-processing](#22-preprocessing) 
- [3. Deep Neural Networks Models](#3-dnns)
    - [3.1. Generative Adversarial Networks (GANs)](#31-gans)
        - [DC-GAN](#DC-SN-GAN)
        - [W-GAN](#W-GAN)
        - [AC-GAN](#AC-GAN)
        - [SN-GAN](#DC-SN-GAN)
    - [3.2. EfficientNet](#32-efficientnet) 
- [4. Environment Requirements](#4-envs)
    - [4.1. Software](#41-software)
    - [4.2. Hardware](#42-hardware)
- [5. Results](#5-results)
    - [5.1. Evaluation Metrics](#51-metrics)
- [6. Conclusions](#6-conclusions) 
 
 
## 1. Introduction <a name="1-intro"></a>

Over the last decade Deep Neural Networks have produced unprecedented performance on a number of tasks, given sufficient data. One of the main challenges in the medical imaging domain is how to cope with the small datasets and limited amount of annotated samples, especially when employing supervised machine learning algorithms that require labeled data and larger training examples.

An attemp to overcome this challenge, the researchers adopted data augmentation schemes, commonly including simple modifications of dataset images such as translation, rotation, flip and scale. However, little additional information can be gained from the diversity provided by these techniques.

On the other hand, since their introduction by [Goodfellowet al.](https://papers.nips.cc/paper/5423-generative-adversarial-nets), Generative Adversarial Networks (GANs) have become the defacto standard for high quality image synthesis. There are two general ways in which GANs have been used in medical imaging. The first is focused on the generative aspect and the second one in the discriminative aspect. Focusing on the first one, GANs can help in exploring and discovering the underlying structure of training data and learning to generate new images. This property makes GANs very promising in coping with data scarcity and patient privacy.


### 1.1. Motivation <a name="11-motivation"></a>

Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective. With the advance of the medical imaging informatics, there is an effort improving the melanoma diagnosis via new AI approaches. For that purpose, sufficient data volume is necessary for training a successful machine learning algorithm for medical image analysis. However, there is inherent problematic problematic in the field of medical imaging where abnormal findings are by definition uncommon. The paucity of annotated ddata and class imbalance of insufficient variability leads to poor classification performance.

In this project we have explored the possibilities of applying different flavors of GANs given their potential to complement image interpretation and augment image representation and classification.


### 1.2. Objectives <a name="12-milestones"></a>

Above all the main purpose of this project is to demonstrate the potential solution to the problem of insufficiency data volume in the medical domain by means of comparing the proposed CNN-based classifier performance using standard against synthetic image augmentation. To tackle this task, it can be further broken down into the following sub-objectives:
- Explore, clean and process the data that will be used for training and evaluating the implemented Deep Neural Networks.
- Research, develop, implement and train a clasiffier model. This classifier will be based on a scaled up CNN whose function will be to detect malign dermathological lesions from the different augmented images.
- Perform classifier performance tests. In order to establish a solid base evaluation model to compare with, there wil be necessary to undertake several experiments for fine tuning the model for the training data appropriately.
- Research, develop, implement and train a series of GANs-based models to be able to demosntrate how much we can improve the performance of the classifier.
- Draw final conclusions from all the experiments conducted. There will be necessary to compare the results obtained from the classifier using the different synthesed images and the different improvements attempted.

## 2. Corpora <a name="2-available-datasets"></a>

### 2.1. Data Description  <a name="21-eda"></a>
### 2.2. Pre-processing  <a name="22-preprocessing"></a> 


## 3. Deep Neural Networks Models <a name="3-dnns"></a>
### 3.1. Generative Adversarial Networks (GANs)  <a name="31-gans"></a>

- [DC-GAN](DC-GAN)<a name="DC-SN-GAN"></a>

A DCGAN is a specific flavor of GAN dedicated to image generation. The architecture consists on a _Generator_ and a _Discriminator_ built upon four 2d convolutional layers. It was first described by _Radford et. al._ in this [paper](https://arxiv.org/pdf/1511.06434.pdf). The _Discriminator_ in build out of strided convolutions, batch normalization layers and uses Leaky Relu activations. Originally, the input size of the images is 64 and it is already set to process color images (3x64x64). The _Generator_ differs from the _Discriminator_ in the convolutional layers, which are transposed. It has as an input a random vector sampled from a normal distribution which will be transformed by adversarial training into an RGB image of the selected shape.


![alt text](https://www.researchgate.net/publication/331282441/figure/download/fig3/AS:729118295478273@1550846756282/Deep-convolutional-generative-adversarial-networks-DCGAN-for-generative-model-of-BF-NSP.png)

- [W-GAN](W-GAN)<a name="W-GAN"></a>
- [AC-GAN](AC-GAN)<a name="AC-GAN"></a>
- [SN-GAN](SN-GAN)<a name="DC-SN-GAN"></a>

The SNGAN is identical to DCGAN but implements _Spectral Normalization_ to deal with the issue of exploding gradients in the _Discriminator_. 

Spectral normalization is a weight regularization technique with is applied to the GAN's _Discriminator_ to solve the issue of exploding gradients. Is works stabilizing the training process of the _Discriminator_ through a rescaling the weights of the convolution layers using a norm (spectral norm) which is calculated using the power iteration method. The method is triggered right before the _forward()_ function call.

Some works refer to DCGANs that implement spectral normalization as SNGANs, which is also done in this work. SNGAN with the best parameters and implementations described in the respective folder was the one used for the image generation.

### 3.2. EfficientNet  <a name="32-efficientnet"></a> 


## 4. Environment Requirements <a name="4-envs"></a>
### 4.1. Software  <a name="41-software"></a>
### 4.2. Hardware  <a name="42-hardware"></a> 


## 5. Results <a name="5-results"></a>
### 5.1. Evaluation Metrics  <a name="51-metrics"></a>

Since we lack from any medical expertise for assessing the quality of the generated images, we have implemented several metrics to measure traits of our output pictures.

#### Peak Signal-to-Noise Ratio (PSNR)

This metric is used to measure the quality of a given image (noise), which underwent some transformation, compared to the its original (signal). In our case, the original picture is the real batch of images feeded into our network and the noise is represented by a given generated image.

#### Structural Similarity (SSIM)

SSIM aims to predict the percieved the quality of a digital image. It is a perception based model that computes the degradation in an image comparison as in the precived change in the structural information. This metric captures the perceptual changes in traits such as luminance and contrast.

#### Multi-Scale Gradient Magnitude Similarity Deviation (MS GMSD)

MS-GMSD works on a similar version as SSIM, but it also accounts for different scales for computing luminance and incorporates chromatic distorsion support.

#### Mean Deviation Similarity Index (MDSI)

MDSI computes the joint similarity map of two chromatic channels through standard deviation pooling, which serves as an estimate of color changes. 

#### Haar Perceptural Similarity Index (HaarPSI)

HaarPSI works on the Haar wavelet decomposition and assesses local similarities between two images and the relative importance of those local regions. This metric is the current state-of-the-art as for the agreement with human opinion on image quality. 

#### Bar of measures

Measure | Bar | 
:------: | :------:|
PSNR   | Context dependant, generally the higher the better.  | 
SSIM   |  Ranges from 0 to 1, being 1 the best value.     | 
MS-GMSD |  Ranges from 0 to 1, being 1 the best value.    |  
MDSI   |   Ranges from 0 to inf, being 0 the best value.    |
HaarPSI |   Ranges from 0 to 1, being 1 the best value.   |

### Obtained metrics and images

Architecture | PSNR | SSIM |MS-GMSD |MDSI |HaarPSI |
:------: | :------:| :------:| :------:| :------:|:------:|
DC-GAN |   12.92  | 0.35   | 0.29   |  0.45 | 0.39 |    
AC-GAN |           |       |       |      |         |
W-GAN |           |        |       |     |         |
SN-GAN |  12.21  |   0.21   |   0.26   |   0.49  |  *0.41*  |  
SN-GAN 128 |  12.18  |   0.24   |   0.15   |   0.52  |  *0.45*  |  


#### DCGAN 64x64
![skin_lesions_700_twick](https://user-images.githubusercontent.com/48655676/110391353-a1d4d980-8067-11eb-9eca-4f458fffd203.png)

### SNGAN 64x64

![skin_lesions_800_twick3_sn](https://user-images.githubusercontent.com/48655676/110391188-70f4a480-8067-11eb-9d8b-ce150ef7797b.png)

### SNGAN 128x128

![SN_final](https://user-images.githubusercontent.com/48655676/114686469-18be5b80-9d13-11eb-80ae-aa53aa7061e6.png)

## 6. Conclusions <a name="6-conclusions"></a>
