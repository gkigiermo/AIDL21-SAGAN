# Wasserstein GAN
## Description

The Wasserstein GAN (WGAN) was first introduced in [Arjovsky et al.](arXiv:1701.07875) with the objective
to improve learning's stability and problems like mode collapse of the traditional GANs.
The proposed GAN consist in using the Earth-Mover distance (Wasserstein-1) for the loss aiming 
at improving the gradient descent.
On the original formulation, a weight clipping was used to achieve 1-Lipchitz functions. 
However, in this work, we have based our implementation on [Gulrajani et al.] (https://arxiv.org/abs/1704.00028)
article that proposes an alternative to the weight clipping, called gradient penalization. i

## Implementation


### Critic

### Gradient Penalty

## Execution

```
python3 image_generation.py input.yaml
```
## New Parameters

The .yaml file described for the DC-GAN must include two new parameters: 

* critic_its = The number of iterations that the generator is frozen

* gp_lamba   = Controls the magnitude of the gradient penaly added to the discrimitor loss

The input file reads:


```
arch: 'WGAN'

path: '/home/name/path/to/images/'
out: '/home/name/path/to/output/images'
run: 'name'
seed: 42
n_gpu: 0

num_epochs: 5
learning_rate: 0.0002
beta_adam: 0.5
batch_size: 16

latent_vector: 256

image_size: 64
loader_workers: 2
number_channels: 3
gen_feature_maps: 64
dis_feature_maps: 64


critic_its : 5
gp_lambda : 10

```
## Results

### Gradient penalty 

### Critic iterations


### 64 x 64

### 128 x 128

