


# Diffusing Gaussian Mixtures for Categorical data
This is a pytorch implementation of the paper: Diffusing Gaussian Mixtures for Categorical data.



# Overview

Learning a categorical distribution comes with its own set of challenges. A successful approach taken by state-of-the-art works is to cast the problem in a continuous domain to take advantage of the impressive performance of the generative models for continuous data. Amongst them are the recently emerging diffusion probabilistic models, which have the observed advantage of generating high-quality samples. Recent advances for categorical generative models have focused on log likelihood improvements. In this work, we propose a generative model for categorical data based on diffusion models with a focus on high-quality sample generation, and propose sampled-based evaluation methods. 

<p align="center">
<img src="/img/overview.png"/>
</p>
The efficacy of our method stems from performing diffusion in the continuous domain while having its parameterization informed by the structure of the categorical nature of the target distribution. Our method of evaluation highlights the capabilities and limitations of different generative models for generating categorical data. 


## Installation
```bash
pip install -r requirements.txt 
```

## Run the demo

```bash
cd gmcd
python run_train.py
```



## Citation

This project was published at AAAI 2023.

```
@inproceedings{regol2023,
	title = {Diffusing Gaussian Mixtures for Generating Categorical Data},
	author={Regol, Florence and Coates, Mark},
	booktitle = {Proc. AAAI Conf. on Artificial Intelligence},
	year = {2023}
}
```


