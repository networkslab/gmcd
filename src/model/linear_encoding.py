import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from src.metrics import one_hot
from src.model.fixed_encoder import ExtActFixed
from src.model.distributions import GaussianDistribution


class LinearCategoricalEncoding(nn.Module):
    """
    Simple encoder q(Z|X). q(Z|X) = \prod^S_s q(Z_s|X_s), q(Z_s|X_s) = Norm(mu_XS, sigma)
    """

    def __init__(self,
                 run_config,
                 dataset_class,
                 K):
        super().__init__()
        self.dataset_class = dataset_class
        self.encoding_dim = run_config.encoding_dim
        self.K = K
        self.prior_distribution = GaussianDistribution(
            mu=0.0, sigma=1.0)  # Prior distribution used for sampling
        self.fixed_encoder = ExtActFixed(d=self.encoding_dim, K=K)
        self.encoder_layers = nn.ModuleList([self.fixed_encoder])

        # Uniform prior over the categories.
        category_prior = torch.zeros(self.K,
                                     dtype=torch.float32)

        self.register_buffer(
            "category_prior", F.log_softmax(category_prior, dim=-1))

    def forward(self,
                z,
                ldj=None,
                reverse=False,
                beta=1,
                delta=0.0,
                channel_padding_mask=None,
                **kwargs):
        # We reshape z into [batch, 1, ...] as every categorical variable is considered to be independent.
        batch_size, seq_length = z.size(0), z.size(1)
        z = z.reshape((batch_size * seq_length, 1) + z.shape[2:])
        if channel_padding_mask is not None:
            channel_padding_mask = channel_padding_mask.reshape(
                batch_size * seq_length, 1, -1)
        else:
            channel_padding_mask = z.new_ones((batch_size * seq_length, 1, 1),
                                              dtype=torch.float32)

        ldj_loc = z.new_zeros(z.size(0), dtype=torch.float32)
        detailed_ldj = {}

        if not reverse:
            # z is of shape [Batch, SeqLength]
            # Renaming here for better readability (what is discrete and what is continuous)
            z_categ = z
            # 1.) Forward pass of current token flow
            z_cont = self.prior_distribution.sample(
                shape=(batch_size * seq_length, 1,
                       self.encoding_dim)).to(z_categ.device)  # z \sim LogDis(0,1)

            init_log_p = self.prior_distribution.log_prob(z_cont).sum(
                dim=[1, 2])
            z_cont, ldj_forward = self._flow_forward(z_cont,
                                                     z_categ,
                                                     reverse=False)
            # Z CONT is sampled from z \sim q(z|x)

            # 2.) Approach-specific calculation of the posterior log p(x|z)
            class_prob_log = self.get_class_prob_log(z_categ,
                                                     z_cont,
                                                     init_log_p=init_log_p,
                                                     ldj_forward=ldj_forward)

            # 3.) Calculate final LDJ
            # class_prob_log =log(p(x_i|z_i))
            # init_log_p - ldj_forward =  log q(z_i|x_i)
            ldj_loc = (beta * class_prob_log - (init_log_p - ldj_forward))
            ldj_loc = ldj_loc * channel_padding_mask.squeeze()
            z_cont = z_cont * channel_padding_mask
            z_out = z_cont

        else:
            # z is of shape [Batch * seq_len, 1, D]
            assert z.size(
                -1
            ) == self.encoding_dim, "[!] ERROR in categorical decoding: Input must have %i latent dimensions but got %i" % (
                self.encoding_dim, z.shape[-1])

            z_cont = z

            z_out = self._posterior_sample(z_cont)
        # Reshape output back to original shape
        if not reverse:
            z_out = z_out.reshape(batch_size, seq_length, -1)
        else:
            z_out = z_out.reshape(batch_size, seq_length)
        ldj_loc = ldj_loc.reshape(batch_size, seq_length).sum(dim=-1)

        # Add LDJ
        if ldj is not None:
            ldj = ldj + ldj_loc
        else:
            ldj = ldj_loc

        return z_out, ldj, detailed_ldj

    def _flow_forward(self, z_cont, z_categ, reverse, **kwargs):
        ldj = z_cont.new_zeros(z_cont.size(0), dtype=torch.float32)
        for flow in (self.encoder_layers
                     if not reverse else reversed(self.encoder_layers)):
            z_cont, ldj = flow(z_cont,
                               z_categ,
                               ldj,
                               reverse=reverse,
                               **kwargs)
        return z_cont, ldj

    def get_class_prob_log(self,
                           x,
                           z_cont,
                           init_log_p=None,
                           ldj_forward=None):  # log(p(x|z))

        # z has not be passed through the flow q(x|z) yet
        if init_log_p is None or ldj_forward is None:
            init_log_p = self.prior_distribution.log_prob(z_cont).sum(
                dim=[1, 2])
            z_cont, ldj_forward = self._flow_forward(z_cont, x, reverse=False)

        class_prior_log = torch.take(self.category_prior, x.squeeze(dim=-1))
        log_point_prob = init_log_p - ldj_forward + \
            class_prior_log  # log q(z_i|x_i) + log prior(x_i)
        class_prob_log = self._calculate_true_posterior(
            z_cont, x, log_point_prob)

        return class_prob_log

    def _calculate_true_posterior(self, z_cont, z_categ, log_point_prob,
                                  **kwargs):
        # Run backward pass of *all* class-conditional flows
        z_back_in = z_cont.expand(-1, self.K,
                                  -1).reshape(-1, 1, z_cont.size(2))
        sample_categ = torch.arange(self.K,
                                    dtype=torch.long).to(z_cont.device)
        sample_categ = sample_categ[None, :].expand(z_categ.size(0),
                                                    -1).reshape(-1, 1)

        z_back, ldj_backward = self._flow_forward(z_back_in,
                                                  sample_categ,
                                                  reverse=True,
                                                  **kwargs)
        back_log_p = self.prior_distribution.log_prob(z_back).sum(dim=[1, 2])

        # Calculate the denominator (sum of probabilities of all classes)
        flow_log_prob = back_log_p + ldj_backward
        log_prob_denominator = flow_log_prob.view(
            z_cont.size(0), self.K) + self.category_prior[None, :]
        # Replace log_prob of original class with forward probability
        # This improves stability and prevents the model to exploit numerical errors during inverting the flows
        orig_class_mask = one_hot(z_categ.squeeze(),
                                  num_classes=log_prob_denominator.size(1))
        log_prob_denominator = log_prob_denominator * \
            (1 - orig_class_mask) + \
            log_point_prob.unsqueeze(dim=-1) * orig_class_mask
        # Denominator is the sum of probability -> turn log to exp, and back to log
        log_denominator = torch.logsumexp(log_prob_denominator, dim=-1)

        # Combine nominator and denominator for final prob log
        class_prob_log = (log_point_prob - log_denominator)
        return class_prob_log

    def _decoder_sample(self, z_cont, **kwargs):
        return self.decoder(z_cont).argmax(dim=-1)

    def _posterior_sample(self, z_cont, **kwargs):  # [Batch * seq_len, 1, D]
        # Run backward pass of *all* class-conditional flows
        z_back_in = z_cont.expand(-1, self.K,
                                  -1).reshape(-1, 1, z_cont.size(2))
        sample_categ = torch.arange(self.K,
                                    dtype=torch.long).to(z_cont.device)
        sample_categ = sample_categ[None, :].expand(z_cont.size(0),
                                                    -1).reshape(-1, 1)

        z_back, ldj_backward = self._flow_forward(z_back_in,
                                                  sample_categ,
                                                  reverse=True,
                                                  **kwargs)
        back_log_p = self.prior_distribution.log_prob(z_back).sum(dim=[1, 2])

        # Calculate the log probability for each class
        flow_log_prob = back_log_p + ldj_backward
        log_prob_denominator = flow_log_prob.view(
            z_cont.size(0), self.K) + self.category_prior[None, :]
        argmax = True
        if argmax:
            return log_prob_denominator.argmax(dim=-1)
        else:  # create a categorical, then sample from it.
            p_x_given_z = D.categorical.Categorical(
                logits=log_prob_denominator)
            x = p_x_given_z.sample()
            return x
