---
layout: post
title: "A Deterministic View of Diffusion Models"
date: 2023-12-20 
---
*In this post, we present a deterministic perspective on diffusion models. In this approach, neural networks are trained as inverse deterministic functions that progressively denoise corrupted images at each timestep. This method simplifies the derivation of diffusion models, making the process more straightforward and comprehensive.*
<br><br>

In recent years, diffusion models, a novel category of deep generative models $[1]$, have made significant strides in producing high-quality, high-resolution images. Notable examples include GLIDE $[2]$, DALLE-2 $[3]$, Imagen, and the fully open-source Stable Diffusion. These models are traditionally built on the framework of Denoising Diffusion Probabilistic Models (DDPM) $[4]$. In this framework, the forward diffusion process is modeled as a Gaussian process with Markovian properties. Conversely, the backward denoising process employs neural networks to estimate the conditional distribution at each timestep. The neural networks involved in the denoising process are trained to minimize the evidence lower bound (ELBO) $[4]$ $[5]$, akin to the approach used in a Variational Autoencoder (VAE).

In this post, we present a deterministic perspective on diffusion models. In this approach, neural networks are trained as inverse deterministic functions that progressively denoise corrupted images at each timestep. This method simplifies the derivation of diffusion models, making the process more straightforward and comprehensive.

<figure align="center">
<figcaption> Figure 1. A deterministic view of diffusion models in the forward diffusion process and the backward denoising transformation. 
  </figcaption> 
  <img src="{{site.url}}/figures/.png" width="600" alt> 
</figure>


### **Forward Diffusion Process**

Starting with any input image, denoted as $\mathbf{x}_0$, drawn from the distribution $p(\mathbf{x})$, the forward process as illustrated in Figure 1, incrementally corrupts the input image for each time steps $t=1,2,\cdots,T$. This corruption is achieved by  progressively adding varying levels of Gaussian noise, 
${\boldsymbol \epsilon} \sim \mathcal{N}(0, \mathbf{I})$,
adhering to a predefined noise schedule: $\beta_1, \beta_2, \ldots, \beta_T$ (where $\beta_1 < \beta_2 < \ldots < \beta_T$). 
This process gradually introduces more noise at each step, leading to increasingly corrupted versions of the original image, $\mathbf{x}_0 \to \mathbf{x}_1 \to \mathbf{x}_2 \to  \cdots \to \mathbf{x}_T$.

We further denote ​ $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, the forward diffusion process is conducted as follows:
$$
\mathbf{x}_t = f(\mathbf{x}_{0},t) = \sqrt{\bar{\alpha}_t} \mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_t} \,  {\boldsymbol \epsilon} \;\;\;\;\;\; \forall t=1, 2, \cdots, T
$$

From the above, we have 
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
= \frac{1}{\sqrt{1 - \bar{\alpha}_{t}}} \big[ \mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_{0}\big]$, and substitute ${\boldsymbol \epsilon}$, we have
$$\begin{aligned}
\mathbf{x}_{t-1}  &= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t-1}} \,   {\boldsymbol \epsilon} \\
&= \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0}  + \frac{\sqrt{1 - \bar{\alpha}_{t-1}}}{\sqrt{1 - \bar{\alpha}_{t}}}
\big[ \mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_{0}\big] \\
& \approx \mathbf{x}_t + \big[ \sqrt{\bar{\alpha}_{t-1}} - \sqrt{\bar{\alpha}_{t}}\, \big] \mathbf{x}_{0} \;\;\;\;\;\; ( 1- \bar{\alpha}_{t} \approx 1- \bar{\alpha}_{t-1}) \\
&= \mathbf{x}_t + \frac{\bar{\alpha}_{t-1} (1-\alpha_t)}{\sqrt{\bar{\alpha}_{t-1}} + \sqrt{\bar{\alpha}_{t}}} \mathbf{x}_0 \\
& \approx \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} (1- \alpha_t)}{2} \mathbf{x}_0
\end{aligned}$$

### **Backward Denoising Process (I)**

In the backward process, starting from a Gaussian noise $\mathbf{x}_T \sim \cal{N}(0, \mathbf{I})$,  
we gradually recover all corrupted images bacwards one by one until we obtain a  clean image: $\mathbf{x}_T \to \mathbf{x}_{T-1} \to \mathbf{x}_{T-2} \to  \cdots \to \mathbf{x}_1 \to \mathbf{x}_0$.

At each time step, given any corrupted image $\mathbf{x}_t$, in order to denoise to recover a slighter better version of image $\mathbf{x}_{t-1}$, we have two choices:

#### **Estimating clean image $\mathbf{x}_0$**

In this case, we construct a deep neural network $\boldsymbol \theta$ to approximate the inverse function of the above diffusion mapping $\mathbf{x}_t = f(\mathbf{x}_0, t)$, denoted as
$$
\hat{\mathbf{x}}_0 = f^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t)
$$
so that we can first recover a rough estimate of the clean image $\hat{\mathbf{x}}_0$ from $\mathbf{x}_t$.
In this case, the neural network is learned by minimizing the following objective function:
$$
L_1({\boldsymbol \theta}) = \sum_{\mathbf{x}_0} \sum_{t=1}^T \Big( f^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t) - \mathbf{x}_0\Big)^2
$$

Once we have learned the inverse mapping, we can  derive an estimate of $\mathbf{x}_{t-1}$ as follows:
$$
\mathbf{x}_{t-1} = \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} (1- \alpha_t)}{2} \hat{\mathbf{x}}_0
$$

At last, the sampling process to generate a new image can be described as follows:

1. sample a Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I}) $
2. for $t=T, T-1, \cdots, 1$:
 * 2.1) if $t>1$, sample another noise $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$, else $\mathbf{z}=0$
 * 2.2) $\mathbf{x}_{t-1} = \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} (1- \alpha_t)}{2} f^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t) + \sigma_t  \mathbf{z}$
3. return $\mathbf{x}_0$

#### **Estimating noise ${\boldsymbol \epsilon}$**

In this case, we construct a deep neural network to  $\boldsymbol \theta$ to approximate the inverse function to estimate the noise ${\boldsymbol \epsilon}$ from each corrupted image $\mathbf{x}_t$ at each time step $t$:
$$
\hat{\boldsymbol \epsilon} = g^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t)
$$
In this case, the neural network is learned by minimizing the following objective function:
$$
L_2({\boldsymbol \theta}) = \sum_{\mathbf{x}_0} \sum_{t=1}^T \Big( g^{-1}_{\boldsymbol \theta} (\mathbf{x}_t, t) - {\boldsymbol \epsilon}\Big)^2
$$

Once we have learned the inverse mapping, we can  derive an estimate of $\mathbf{x}_{t-1}$ as follows:
$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \Big[ \mathbf{x}_t -  
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,   {\boldsymbol \epsilon} 
\Big]
$$

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



