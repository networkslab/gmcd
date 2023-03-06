# Diffusing Gaussian Mixtures for Categorical data
This is an implementation of the paper [diffusing Gaussian Mixtures for Categorical data](https://arxiv.org/abs/2106.0606)

Learning a categorical distribution comes with its own set of challenges. A successful approach taken by state-of-the-art works is to cast the problem in a continuous domain to take advantage of the impressive performance of the generative models for continuous data. Amongst them are the recently emerging <b>diffusion probabilistic models</b>, which have the observed advantage of generating high-quality samples. Recent advances for categorical generative models have focused on log likelihood improvements. In this work, we propose a generative model for categorical data based on diffusion models with a focus on high-quality sample generation, and propose sampled-based evaluation methods. 

<p align="center">
<img src="/img/overview.png"/>
</p>
The efficacy of our method stems from performing diffusion in the continuous domain while having its parameterization informed by the structure of the categorical nature of the target distribution. Our method of evaluation highlights the capabilities and limitations of different generative models for generating categorical data. 

## Sphere packing 
To map the data to the continuous space, we use a a sphere packing algorithm to set a flexible and fixed encoding.

<b>Idea: Solve a sphere packing problem</b> 
$\rightarrow$ Fits  $K$ well-separated balls in a d-dimensional real space: $ \mathbf{u}_1 \dots $


