from src.run_config import RunConfig


class SyntheticRunConfig(RunConfig):
    def __init__(self,
                 dataset,
                 S,
                 K=None,  # by default, K=S
                 encoding_dim=3,
                 max_iterations=1000,
                 T=10) -> None:
        super().__init__(encoding_dim, max_iterations, T)
        self.S = S  # Number of elements in the sets.
        self.K = K
        self.encoding_dim = encoding_dim
        self.eval_freq = 500
        self.dataset = dataset

        if self.K is not None:
            self.dataset = "synthetic_high_K"
        else:
            self.K = self.S

        self.transformer_dim = 64

        super().set_dependent_config()
