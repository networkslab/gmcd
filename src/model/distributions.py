

import torch

class PriorDistribution(torch.nn.Module):

	GAUSSIAN = 0


	def __init__(self, **kwargs):
		super().__init__()
		self.distribution = self._create_distribution(**kwargs)


	def _create_distribution(self, **kwargs):
		raise NotImplementedError


	def forward(self, shape=None):
		return self.sample(shape=shape)


	def sample(self, shape=None):
		if shape is None:
			return self.distribution.sample()
		else:
			return self.distribution.sample(sample_shape=shape)


	def log_prob(self, x):
		logp = self.distribution.log_prob(x)
		assert torch.isnan(logp).sum() == 0, "[!] ERROR: Found NaN values in log-prob of distribution.\n" + \
				"NaN logp: " + str(torch.isnan(logp).sum().item()) + "\n" + \
				"NaN x: " + str(torch.isnan(x).sum().item()) + ", X(abs) max: " + str(x.abs().max())
		return logp


	def prob(self, x):
		return self.log_prob(x).exp()


	def icdf(self, x):
		assert ((x < 0) | (x > 1)).sum() == 0, \
			   "[!] ERROR: Found values outside the range of 0 to 1 as input to the inverse cumulative distribution function."
		return self.distribution.icdf(x)


	def cdf(self, x):
		return self.distribution.cdf(x)


	def info(self):
		raise NotImplementedError


	

class GaussianDistribution(PriorDistribution):


	def __init__(self, mu=0.0, sigma=1.0, **kwargs):
		super().__init__(mu=mu, sigma=sigma, **kwargs)
		self.mu = mu
		self.sigma = sigma


	def _create_distribution(self, mu=0.0, sigma=1.0, **kwargs):
		return torch.distributions.normal.Normal(loc=mu, scale=sigma)


	def info(self):
		return "Gaussian distribution with mu=%f and sigma=%f" % (self.mu, self.sigma)
