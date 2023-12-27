---
layout: post
title: "A Deterministic View of Diffusion Models"
date: 2023-12-20 
---
*In this post, we present a deterministic perspective on diffusion models. In this approach, neural networks are trained as inverse deterministic functions that progressively denoise corrupted images at each timestep. This method simplifies the derivation of diffusion models, making the process more straightforward and comprehensive.*
<br><br>

In recent years, diffusion models, a novel category of deep generative models $[1]$, have made significant strides in producing high-quality, high-resolution images. Notable examples include GLIDE $[2]$, DALLE-2 $[3]$, Imagen, and the fully open-source Stable Diffusion. These models are traditionally built on the framework of Denoising Diffusion Probabilistic Models (DDPM) $[4]$. In this probabilistic framework, the forward diffusion process is modeled as a Gaussian process with Markovian properties. Conversely, the backward denoising process employs neural networks to estimate the conditional distribution at each timestep. The neural networks involved in the denoising process are trained to minimize the evidence lower bound (ELBO) $[4]$ $[5]$, akin to the approach used in a Variational Autoencoder (VAE).

In this post, we present a deterministic perspective on diffusion models. In this approach, neural networks are trained as inverse deterministic functions that progressively denoise corrupted images at each timestep. This method simplifies the derivation of diffusion models, making the process more straightforward and comprehensive.

<figure align="center">
  <img src="{{site.url}}/figures/deterministic-diffusion.png" width="600" alt> 
  <figcaption> Figure 1. A deterministic view of diffusion models in the forward diffusion process and the backward denoising transformation (Image adapted from [6]). 
  </figcaption> 
</figure>


### **Forward Diffusion Process**

As shown in Figure 1, starting with any input image, denoted as $\mathbf{x}_0$, drawn from a data distribution $p(\mathbf{x})$, the forward process incrementally corrupts the input image for each time steps $t=1,2,\cdots,T$. This corruption is achieved by  progressively adding varying levels of Gaussian noises as
$$
\mathbf{x}_t = \sqrt{\alpha_t } \mathbf{x}_{t-1} + \sqrt{1- \alpha_t } \,  {\boldsymbol \epsilon}_t \;\;\;\;\;\; \forall t=1, 2, \cdots, T
$$
where noise at each timestep ${\boldsymbol \epsilon}_t \sim \mathcal{N}(0, \mathbf{I})$,
adhering to a predefined noise schedule: $\alpha_1, \alpha_2, \ldots, \alpha_T$ (with $1 > \alpha_1 > \alpha_2 > \ldots > \alpha_T \geq 0$). 
This process gradually introduces more noise at each step, leading to a sequence of increasingly corrupted versions of the original image: $\mathbf{x}_0 \to \mathbf{x}_1 \to \mathbf{x}_2 \to  \cdots \to \mathbf{x}_T$. When $T$ is large enough, the last image $\mathbf{x}_T$ approaches to a Gaussian noise, i.e. $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$.

Building on the approach outlined in $[4]$, the above diffusion process can be implemented much more efficiently. Rather than sampling a unique Gaussian noise at each timestep, it is feasible to sample a single Gaussian noise, $ \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$, and employ the subsequent formula to efficiently generate all the corrupted samples in one go (along the left-to-right red dash arrow in Figure 1):
$$
\mathbf{x}_t = f(\mathbf{x}_{0},t) = \sqrt{\bar{\alpha}_t} \mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_t} \,  {\boldsymbol \epsilon} \;\;\;\;\;\; \forall t=1, 2, \cdots, T
$$
where $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. This method streamlines the process, making the generation of corrupted samples more straightforward and less computationally demanding.

Assuming the application of the aforementioned formula in the diffusion process, let's explore the relationship between between  $\mathbf{x}_{t-1}$ and $\mathbf{x}_t$ in this case. This exploration will help us understand how consecutive stages in the diffusion process are interrelated, which is crucial in the subsequent backward denoising process. In fact, it's possible to establish how these two consecutive  samples are connected using two distinct approaches. 

First of all, we have 
$ \mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}
\big[ \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\,   {\boldsymbol \epsilon} \big]
$, and substitute $\mathbf{x}_0$ to further derive the relationship between any two adjacent samples, i.e. $\mathbf{x}_t$ and $\mathbf{x}_{t-1}$, as follows:
$$\begin{aligned}
\mathbf{x}_{t-1}  &= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t-1}} \,   {\boldsymbol \epsilon} \\
&= \frac{1}{\sqrt{\alpha_t}} \big[ \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\,   {\boldsymbol \epsilon} \big] + \sqrt{1 - \bar{\alpha}_{t-1}} \,   {\boldsymbol \epsilon} \\
&= \frac{1}{\sqrt{\alpha_t}} \Big[ \mathbf{x}_t - 
\big( \sqrt{1-\bar{\alpha}_t} - \sqrt{\alpha_t-\bar{\alpha}_t}
\big) {\boldsymbol \epsilon} \Big] \\
&= \frac{1}{\sqrt{\alpha_t}} \Big[ \mathbf{x}_t -  
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t} + \sqrt{\alpha_t-\bar{\alpha}_t}}\,   {\boldsymbol \epsilon} 
\Big] \\
&\approx \frac{1}{\sqrt{\alpha_t}} \Big[ \mathbf{x}_t -  
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,   {\boldsymbol \epsilon} 
\Big] \;\;\;\;\;\;\; (\mathrm{as} \; \alpha_t \ll 1)
\end{aligned}$$

Alternatively, we can have ${\boldsymbol \epsilon} 
= \frac{1}{\sqrt{1 - \bar{\alpha}_{t}}} \big[ \mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_{0}\big]$, and substitute ${\boldsymbol \epsilon}$ into how $\mathbf{x}_{t-1}$ is computed, we have
$$\begin{aligned}
\mathbf{x}_{t-1}  &= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t-1}} \,   {\boldsymbol \epsilon} \\
&= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0}  + \frac{\sqrt{1 - \bar{\alpha}_{t-1}}}{\sqrt{1 - \bar{\alpha}_{t}}}
\big[ \mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_{0}\big] \\
& \approx \mathbf{x}_t + \big[ \sqrt{\bar{\alpha}_{t-1}} - \sqrt{\bar{\alpha}_{t}}\, \big] \mathbf{x}_{0} \;\;\;\;\;\; (\mathrm{as} \;  1- \bar{\alpha}_{t} \approx 1- \bar{\alpha}_{t-1}) \\
&= \mathbf{x}_t + \frac{\bar{\alpha}_{t-1} (1-\alpha_t)}{\sqrt{\bar{\alpha}_{t-1}} + \sqrt{\bar{\alpha}_{t}}} \mathbf{x}_0 \\
& \approx \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} (1- \alpha_t)}{2} \mathbf{x}_0
\end{aligned}$$

### **Backward Denoising Process**

In the backward process, starting from a Gaussian noise $\mathbf{x}_T \sim \cal{N}(0, \mathbf{I})$,  
we gradually recover all corrupted images backwards one by one until we obtain the initial clean image: $\mathbf{x}_T \to \mathbf{x}_{T-1} \to \mathbf{x}_{T-2} \to  \cdots \to \mathbf{x}_1 \to \mathbf{x}_0$.

At each timestep, given the corrupted image $\mathbf{x}_t$, in order to denoise to recover a slightly cleaner version of the image $\mathbf{x}_{t-1}$, we have two choices:

#### **I. Estimating clean image $\mathbf{x}_0$**

In this case, we construct a deep neural network $\boldsymbol \theta$ to approximate the inverse function of the above diffusion mapping $\mathbf{x}_t = f(\mathbf{x}_0, t)$, denoted as
$$
\hat{\mathbf{x}}_0 = f^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t)
$$
which can recover a rough estimate of the clean image $\hat{\mathbf{x}}_0$ from $\mathbf{x}_t$.
In this case, this neural network is learned by minimizing the following objective function:
$$
L_1({\boldsymbol \theta}) = \sum_{\mathbf{x}_0} \sum_{t=1}^T \Big( f^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t) - \mathbf{x}_0\Big)^2
$$

Once we have learned this neural network, we can  derive an estimate of $\mathbf{x}_{t-1}$ from $\mathbf{x}_{t}$  as follows:
$$\begin{aligned}
\mathbf{x}_{t-1} &= \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} (1- \alpha_t)}{2} \hat{\mathbf{x}}_0 \\
&= \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} (1- \alpha_t)}{2} f^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t)
\end{aligned}$$

At last, the sampling process to generate a new image can be described as follows:

1. sample a Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I}) $
2. for $t=T, T-1, \cdots, 1$:
 * 2.1) if $t>1$, sample another noise $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$, else $\mathbf{z}=0$
 * 2.2) $\mathbf{x}_{t-1} = \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} (1- \alpha_t)}{2} f^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t) + \sigma_t  \mathbf{z}$
3. return $\mathbf{x}_0$

#### **Estimating noise ${\boldsymbol \epsilon}$**

In this case, we construct a deep neural network to  $\boldsymbol \theta$ to approximate the inverse function via estimating the noise ${\boldsymbol \epsilon}$ from each corrupted image $\mathbf{x}_t$ at each timestep $t$:
$$
\hat{\boldsymbol \epsilon} = g^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t)
$$
This neural network is learned by minimizing the following objective function:
$$
L_2({\boldsymbol \theta}) = \sum_{\mathbf{x}_0} \sum_{t=1}^T \Big( g^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t) - {\boldsymbol \epsilon}\Big)^2
$$

Once we have learned this neural network, we can  derive an estimate of $\mathbf{x}_{t-1}$ as follows:
$$\begin{aligned}
\mathbf{x}_{t-1} &= \frac{1}{\sqrt{\alpha_t}} \Big[ \mathbf{x}_t -  
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,   {\boldsymbol \epsilon} \Big] \\
&= \frac{1}{\sqrt{\alpha_t}} \Big[ \mathbf{x}_t -  
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,   g^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t) \Big] 
\end{aligned}$$

At last, the sampling process to generate a new image can be described as follows:

1. sample a Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I}) $
2. for $t=T, T-1, \cdots, 1$:
 * 2.1) if $t>1$, sample another noise $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$, else $\mathbf{z}=0$
 * 2.2) $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \big[ \mathbf{x}_t -  
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,   g^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t)
\big] + \sigma_t  \mathbf{z}$
3. return $\mathbf{x}_0$


### **References**


$[1]$ Hui Jiang, *[Machine Learning Fundamentals](https://wiki.eecs.yorku.ca/user/hj/research:mlfbook)*, Cambridge University Press, 2021.

$[2]$ Alex Nichol, Prafulla Dhariwal, *et.al.*, 
*GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models*, [arXiv:2112.10741
](https://arxiv.org/abs/2112.10741), 2021. 

$[3]$  Aditya Ramesh, Prafulla Dhariwal, *et.al.*, *Hierarchical Text-Conditional Image Generation with CLIP Latents*, [arXiv:2204.06125
](https://arxiv.org/abs/2204.06125), 2022. 

$[4]$ Jonathan Ho, Ajay Jain, Pieter Abbeel, *Denoising Diffusion Probabilistic Models*, [arXiv:arXiv:2006.11239
](https://arxiv.org/abs/2006.11239), 2020. 

$[5]$ Calvin Luo, *Understanding Diffusion Models: A Unified Perspective*, [arXiv:arXiv:arXiv:2208.11970
](https://arxiv.org/abs/2208.11970), 2022. 

$[6]$ Sergios Karagiannakos,Nikolas Adaloglou, *How diffusion models work: the math from scratch*, https://theaisummer.com/diffusion-models/.



