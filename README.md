<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/github-markdown-css/2.2.1/github-markdown.css"/>



# Diffusing Gaussian Mixtures for Categorical data
This is an implementation of the paper [diffusing Gaussian Mixtures for Categorical data](https://arxiv.org/abs/2106.0606)

Learning a categorical distribution comes with its own set of challenges. A successful approach taken by state-of-the-art works is to cast the problem in a continuous domain to take advantage of the impressive performance of the generative models for continuous data. Amongst them are the recently emerging <b>diffusion probabilistic models</b>, which have the observed advantage of generating high-quality samples. Recent advances for categorical generative models have focused on log likelihood improvements. In this work, we propose a generative model for categorical data based on diffusion models with a focus on high-quality sample generation, and propose sampled-based evaluation methods. 

<p align="center">
<img src="/img/overview.png"/>
</p>
The efficacy of our method stems from performing diffusion in the continuous domain while having its parameterization informed by the structure of the categorical nature of the target distribution. Our method of evaluation highlights the capabilities and limitations of different generative models for generating categorical data. 




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

