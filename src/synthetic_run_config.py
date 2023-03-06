from src.run_config import RunConfig


class SyntheticRunConfig(RunConfig):
    def __init__(self,
                 model,
                 dataset,
                 S,
                 K=None,  # by default, K=S
                 encoding_dim=3,
                 var_coef=0.5,
                 max_iterations=10000,
                 T=10,
                 max_train_time=None,
                 corrected_var=True) -> None:
        super().__init__(model, encoding_dim, var_coef,
                         max_iterations, T, max_train_time)
        self.S = S  # Number of elements in the sets.
        self.K = K
        self.eval_freq = 500
        if self.S == 16:
            self.batch_size = 1024
            self.eval_freq = 500  # Frequency the model should be evaluated
        elif self.S == 8:
            self.eval_freq = 500
        self.dataset = dataset
        if self.K is not None:
            self.dataset = "synthetic_high_K"
        else:
            self.K = self.S

        self.transformer_dim = 64
        self.not_doing = False
        self.corrected_var = corrected_var

        super().set_dependent_config()
