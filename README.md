# Diffusing Gaussian Mixtures for Categorical data
This is an implementation of the paper [diffusing Gaussian Mixtures for Categorical data](https://arxiv.org/abs/2106.0606)

Learning a categorical distribution comes with its own set of challenges. A successful approach taken by state-of-the-art works is to cast the problem in a continuous domain to take advantage of the impressive performance of the generative models for continuous data. Amongst them are the recently emerging <b>diffusion probabilistic models</b>, which have the observed advantage of generating high-quality samples. Recent advances for categorical generative models have focused on log likelihood improvements. In this work, we propose a generative model for categorical data based on diffusion models with a focus on high-quality sample generation, and propose sampled-based evaluation methods. 

<p align="center">
<img src="/img/overview.png"/>
</p>
The efficacy of our method stems from performing diffusion in the continuous domain while having its parameterization informed by the structure of the categorical nature of the target distribution. Our method of evaluation highlights the capabilities and limitations of different generative models for generating categorical data. 

## Sphere packing 
To map the data to the continuous space, we use a a sphere packing algorithm to set a flexible and fixed encoding.

<b>Idea: Solve a sphere packing problem</b> $\rightarrow$ Fits  $K$ well-separated balls in a d-dimensional real space:
$$
\mathbf{u}_1
$$

We can then use the solution to this problem to set the mean and variance of an encoding distribution conditioned on the category: 
$
q(\cdot|x)  = \mathcal{N}(\mathbf{u}^*_x, \sigma).
$ where  $\sigma$ is derived from the min distance between $\mathbf{u}^*_1 ,.., \mathbf{u}^*_K$. 

Argmax encoding            |  Learned encoding |  Sphere packing encoding
:-------------------------:|:-------------------------: |:-------------------------:
<img src="/img/argmax.png" width="100"/>  |   <img src="/img/learned.png" width="100"/> |   <img src="/img/sphere.png" width="100"/> 

## Parameterizing the denoising step
This parameterization induces a structure on the targeted distribution, which enables us to design a task-cognizant Gaussian Mixtures denoising function:


$
d_{\theta}(\mathbf{Z}^{t-1} | \mathbf{Z}^t) =
  \prod_{s=1}^S \sum^K_{k=1}  p(\mathbf{Z}^{t-1}_{(s)} | \mathbf{Z}^{t}_{(s)} ,C_k)  p_{\theta}(X_{(s)}=C_k|\mathbf{Z}^t, t)
$
 where $\mathbf{Z}^t_{(s)} \in \mathcal{R}^d$ is the $t$-representation of $x_s$. ($\mathbf{Z}^t \in \mathcal{R}^{S\times d}$). We can obtain 
a closed-from expression for the conditional $p(\mathbf{Z}^{t-1}_{(s)} | \mathbf{Z}^{t}_{(s)} , x_{(s)})$; it's a MV Gaussian with fixed paramters. Hence we only have to learn the  mixture weights $p_{\theta}(X_{(s)}|\mathbf{Z}^t, t)$ to learn the denoising step.



## Team
* [Florence Regol](/docs/members/flo)
* [Mark Coates](/docs/members/mark.md)

## Citation

This project was published at AAAI 2023.

```
@inproceedings{regol2023, 
author={F. Regol and M. Coates}
```

[ArXiv link](https://arxiv.org/abs/2106.0606)
