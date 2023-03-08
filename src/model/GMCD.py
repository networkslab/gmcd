
from torch import nn
from src.model.linear_encoding import LinearCategoricalEncoding
from src.model.artransformer_diff import ArTransformerDiffusion
 

class GMCD(nn.Module):
    def __init__(self,  run_config, dataset_class, name="GMCD", figure_path=""):
        super().__init__()
        self.figure_path = figure_path
        self.name = name
        self.run_config = run_config
        self.dataset_class = dataset_class
        self.S = self.run_config.S
        self.K = self.run_config.K
        self.encoding_dim = self.run_config.encoding_dim
        self.linear_encoding_layer = LinearCategoricalEncoding(self.run_config, dataset_class=self.dataset_class, K=self.K)
        fixed_encoder = self.linear_encoding_layer.fixed_encoder
        T = int(self.run_config.T)

        posterior_sample_fun = self.linear_encoding_layer._posterior_sample
        self.z_0_layer = ArTransformerDiffusion(self.run_config, self.S, self.encoding_dim, T, fixed_encoder,
                                                posterior_sample_fun=posterior_sample_fun,
                                                figure_path=figure_path)

    def sample(self, num_samples, watch_z_t=False, **kwargs):
        out_sample = self.z_0_layer.sample(
            num_samples, watch_z_t=watch_z_t)
        z_0 = out_sample['sample']
        z_0 = z_0.reshape((num_samples, self.S, self.encoding_dim))

        out_sample['x'], _, _ = self.linear_encoding_layer(
            z_0, ldj=None, reverse=True)
        return out_sample

    def forward(self, z, ldj=None, reverse=False,  **kwargs):

        if not reverse:
            x_cat = z
            z_0, encoder_ldj, _ = self.linear_encoding_layer(
                z, reverse=False, x_cat=x_cat, **kwargs)
            z, fcdm_ldj = self.z_0_layer(
                z_0, reverse=reverse, x_cat=x_cat, **kwargs)
            ldj = encoder_ldj + fcdm_ldj
        if reverse:
            z_0, fcdm_ldj = self.z_0_layer(
                z, reverse=True, x_cat=None, **kwargs)
            x_cat, encoder_ldj = self.linear_encoding_layer(
                z, reverse=True, x_cat=None, **kwargs)
        ldj = encoder_ldj + fcdm_ldj
        return ldj

    def need_data_init(self):
        return False
