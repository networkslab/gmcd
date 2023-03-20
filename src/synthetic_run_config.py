from src.run_config import RunConfig


class SyntheticRunConfig(RunConfig):
    def __init__(self,
                 dataset,
                 S) -> None:
        super().__init__()
        self.S = S  # Number of elements in the sets.
        self.K = S
       
        self.eval_freq = 500
        self.dataset = dataset

        if self.K == 6:
            self.T = 10
            self.diffusion_steps = self.T
            self.batch_size = 1024
            self.encoding_dim = 6
            self.max_iterations = 1000
            self.transformer_dim = 64
        
        super().set_dependent_config()
