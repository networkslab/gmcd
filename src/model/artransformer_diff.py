from torch.distributions.utils import logits_to_probs
import torch as th
import torch.distributions as D
import torch.nn.functional as F
from src.model.gaussian_diff import GaussianDiffusion
from src.model.linear_transformer import DenoisingTransformer
from src.model.diff_utils import LossType, ModelMeanType, ModelVarType, extract_into_tensor, get_named_beta_schedule, normal_kl

import numpy as np


class ArTransformerDiffusion(GaussianDiffusion):
    def __init__(
            self, diffusion_params,
            S,
            latent_dim,
            T,
            extActFixed,
            posterior_sample_fun=None,
            figure_path=""):
        super().__init__(
            sequence_length=S,
            latent_dim=latent_dim,
            T=T,
            denoise_fn=None,
            betas=get_named_beta_schedule('cosine', T),
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_LARGE,
            rescale_timesteps=False,  figure_path=figure_path)

        self.extActFixed = extActFixed
        K = extActFixed.fixed_embedding.num_embeddings
        self.loss_type = LossType.NOISY_GUIDED_SHARP
        self.K = K
        self.S = S
        self.T = T
        self.d = latent_dim
        self.alpha = diffusion_params.alpha
        if self.alpha is None:
            self.loss_type = LossType.NLL
        self.posterior_sample = posterior_sample_fun  # p(x|z, t)
        self.nll_loss = th.nn.BCEWithLogitsLoss()
        try:
            self.corrected_var = diffusion_params.corrected_var
        except Exception as e:
            self.corrected_var = False

        self.mixture_weights = DenoisingTransformer(
            K=K, S=S, latent_dim=latent_dim, diffusion_params=diffusion_params)

        # other coeff needed for the posterior N(z_t-1|z_t,x)
        self.sigma_tilde = (self.betas *
                            (1.0 - self.alphas_cumprod_prev)
                            / (1.0 - self.alphas_cumprod)
                            )

    def need_data_init(self):
        return []

    def forward(self, z, ldj=None, reverse=False, x_cat=None, **kwargs):
        batch_size, set_size, hidden_dim = z.size(0), z.size(1), z.size(2)
        if not reverse:

            ldj = z.new_zeros(batch_size, )
            z = z.reshape((batch_size, self.d_in))
            device = z.device
            if self.training:
                t = th.randint(0, self.T, (batch_size, ),
                               device=device).long()
                ldj = -self.training_losses(z, t, x_cat, **kwargs)['loss']
                z = z.reshape((batch_size, set_size, hidden_dim))
                return z, ldj
            else:
                ldj = self.log_likelihood(z)
                return z, ldj

        else:
            ldj = self.nll(z)
            return z, ldj

    def log_likelihood(self, z_0):
        b = z_0.size(0)
        device = z_0.device
        log_likelihood = 0

        for t in range(0, self.num_timesteps):
            t_array = (th.ones(b, device=device) * t).long()
            sampled_z_t = self.q_sample(z_0, t_array)
            kl_approx = self.compute_Lt(
                z_0=z_0,
                z_t=sampled_z_t,
                t=t_array)

            log_likelihood += kl_approx

        qt_mean, _, qt_log_variance = self.q_mean_variance(z_0, t_array)
        # THIS SHOULD BE SUPER SMALL
        kl_prior = -normal_kl(qt_mean, qt_log_variance, mean2=0.0, logvar2=0.0)
        kl_prior = th.sum(kl_prior, dim=1)

        log_likelihood += kl_prior

        return log_likelihood

    def compute_Lt(self, z_0, z_t, t):

        z_t = z_t.reshape(-1, self.S, self.d)
        logits_output = self.mixture_weights(t, z_t)  # get p_theta
        transformer_probs = logits_to_probs(logits_output)

        dist = self.get_zt_given(z_t, t)

        w = self.get_w(dist, z_0, z_t, t)

        terms = transformer_probs * w  # all p_theta * w
        approx_kl = th.log(th.sum(terms, dim=2))  # take log of sum accross k
        approx_kl = th.sum(approx_kl, dim=1)

        z_0 = z_0.reshape(-1, self.S, self.d)
        stacked_z_0 = z_0[:, :, None, :].repeat(
            1,  1, self.K, 1)  # repeat along K
        log_pdf_z0 = dist.log_prob(stacked_z_0)
        terms = transformer_probs * th.exp(log_pdf_z0)
        decoder_nll = -th.log(th.sum(terms, dim=2))
        decoder_nll = th.sum(decoder_nll, dim=1)

        mask = (t == th.zeros_like(t)).float()
        # replace nan to zero as they should not b added in the first place
        loss = mask * th.nan_to_num(decoder_nll) + (1. - mask) * approx_kl

        return loss

    def get_w(self, dist, z_0, z_t, t):
        z_t = z_t.reshape(-1, self.S * self.d)
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=z_0, x_t=z_t, t=t
        )
        true_mean = true_mean.reshape(-1, self.S, self.d)
        true_log_variance_clipped = true_log_variance_clipped.reshape(
            -1, self.S, self.d)
        stacked_true_mean = true_mean[:, :, None, :].repeat(
            1,  1, self.K, 1)  # repeat along K
        stacked_true_log_variance_clipped = true_log_variance_clipped[:, :, None, :].repeat(
            1,  1, self.K, 1)  # repeat along S

        kl = normal_kl(stacked_true_mean, stacked_true_log_variance_clipped,
                       dist.mean, th.log(dist.variance))
        kl = th.sum(kl, dim=3)
        w = th.exp(-kl)
        return w

    def training_losses(self, z_0, t, x_cat, **kwargs):

        noise = th.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise=noise)
        z_t = z_t.reshape(-1, self.S, self.d)
        # to do, z_t from z_t-1
        check_fraction_same_x = False
        if self.loss_type in [LossType.NOISY_GUIDED_SHARP]:

            dist = self.get_zt_given(z_t, t)
            w = self.get_w(dist, z_0, z_t, t).view(-1, self.S)
            w = w.permute(1, 0)
            norm_w = th.sum(w, dim=0).view(-1)  # sum over k
            p_w = th.div(w, norm_w).permute(1, 0).view(-1, self.S, self.K)
            if self.loss_type == LossType.NOISY_GUIDED_SHARP:
                p_w_power = p_w**self.alpha
                p_w_power = p_w_power.reshape(-1, self.K)
                p_w_power = p_w_power.permute(1, 0)
                norm_w = th.sum(p_w_power, dim=0).view(-1)
                p_w = th.div(p_w_power, norm_w).permute(
                    1, 0).view(-1, self.S, self.K)

            c = D.categorical.Categorical(p_w)
            w = c.sample()
            mask = (t == th.zeros_like(t)).int().view(-1, 1)
            # at 0 we take x as is
            w = mask * x_cat + (1 - mask) * w
        else:
            w = x_cat
        if check_fraction_same_x:
            self.check_fraction_same_x(w, x_cat, t)

       
       
        logits_output = self.mixture_weights(t, z_t)
        terms = {}
        transformer_probs = logits_to_probs(logits_output)
       

        logits_output_flat = logits_output.reshape(-1, self.K)
        w = x_cat
        w_flat = w.view(-1)
        w_flat = F.one_hot(w_flat, num_classes=self.K).type(th.float32)
        neg_log_likelihood = F.binary_cross_entropy_with_logits(
            logits_output_flat, w_flat, reduction='none')
        neg_log_likelihood = th.sum(
            neg_log_likelihood.view(-1, self.S, self.K), dim=2)
        terms['loss'] = th.sum(neg_log_likelihood, dim=1)

        return terms

    def inspect_mix_pis(self, mixing_logits, t, x_cat=None):
        name = "entropy_transformer.pdf"
        probs = logits_to_probs(mixing_logits)
        self.check_entropy(probs, t, name)

    @th.no_grad()
    def sample(self, num_samples,  watch_z_t=False):

        device = next(self.mixture_weights.parameters()).device
        shape = (num_samples, self.d_in)
        z = th.randn(*shape, device=device)
        z_T = z
        indices = list(range(self.num_timesteps))[::-1]
        t_to_check = [int(self.num_timesteps/2)]
        return_dict = {}

        for i in indices:

            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(z, t)
                z = out["sample"]
                pi = out["logits"]
                x_w = out["sampled_w"]

        return_dict['z_T'] = z_T
        return_dict.update(out)
        return return_dict

    def p_sample(self, z_t, t):  # denoising step
        # x \sim p(X|z^t)
        # z^t-1 \sim p(Z^t-1|X=x)
        z_t = z_t.reshape(-1, self.S, self.d)
        b = z_t.shape[0]

        logits = self.mixture_weights(t, z_t)  # B, S, K
        p_w_given_past = D.categorical.Categorical(logits=logits)
        w = p_w_given_past.sample()

        norm_of_w = self.get_zt_given(z_t, t, w)
        sample = norm_of_w.sample().type(th.float32)
        return {"sample": sample, "sampled_w": w, 'logits': logits}

    def info(self):
        return 'Task-cognizant Diffusion Model'

    # norm [b, s, k, d] or # norm [b, s, d]
    def get_zt_given(self,  z_t, t, x=None):
        b = z_t.shape[0]
        device = z_t.device
        mu_k, var = self.extActFixed.get_mean_var(device)
        var = var[0, 0]
        shape = (b, self.S, self.K, self.d)
        posterior_mean_coef1 = extract_into_tensor(self.posterior_mean_coef1,
                                                   t, (b, self.K, self.d))
        stacked_means = mu_k * posterior_mean_coef1  # coef * mu_k, [b,K,d]
        stacked_means = stacked_means[:, None, :, :].repeat(
            1,  self.S, 1, 1)  # repeat along S

        if x is not None:
            shape = (b, self.S, self.d)
            x = x.reshape(b*self.S)
            stacked_means = stacked_means.reshape(b*self.S, self.K, self.d)
            index = th.arange(start=0, end=b*self.S, dtype=th.long)
            stacked_means = stacked_means[index, x, :]
            stacked_means = stacked_means.reshape(
                b, self.S, self.d)  # only one normal per S

        posterior_mean_coef2 = extract_into_tensor(self.posterior_mean_coef2,
                                                   t, z_t.shape)

        z_term = posterior_mean_coef2 * z_t  # coef * z_t [b, S, d]
        if x is None:
            z_term = z_term[:, :, None, :].repeat(
                1,  1, self.K, 1)  # repeat along K

        mean_sk_given_z = stacked_means + z_term

        posterior_mean_coef1 = extract_into_tensor(self.posterior_mean_coef1,
                                                   t, shape)
        sigma_tilde = extract_into_tensor(self.sigma_tilde,
                                          t, shape)

        if self.corrected_var:
            stacked_std = th.sqrt(posterior_mean_coef1 **
                                  2 * var**2 + sigma_tilde)
        else:
            stacked_std = posterior_mean_coef1 * var + sigma_tilde

        # norm [b, s, k, d] or # norm [b, s, d]
        return D.Independent(D.Normal(mean_sk_given_z, stacked_std), 1)
