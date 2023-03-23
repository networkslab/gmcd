import numpy as np
import random
import torch
"""
Default config. Will usually be overriden by a task specific config class.
"""


class RunConfig():
    def __init__(self) -> None:
        self.set_boring_config()

        self.seed = 3  # 0

        # Maximum number of epochs to train
        self.print_freq = 100  # Frequency loss information
        self.eval_freq = 5000  # Frequency the model should be evaluated
        self.batch_size = 1024  # Batch size
        self.learning_rate = 7.5e-4  # Learning rate of the optimizer
        self.scale_loss = 1e-4
        self.diffusion_model = 'transformer'
        self.input_dp_rate = 0.2
        self.transformer_heads = 8
        self.transformer_depth = 2
        self.transformer_blocks = 1
        self.transformer_local_heads = 4
        self.transformer_local_size = 64
        self.transformer_reversible = False
        self.alpha = 1.1

    # config that are linked to other config, to be called if we change some values

    def set_dependent_config(self):
        self.set_seed()

    def set_seed(self):
        # Set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_description(self):
        description = '_encod_' + str(
            self.encoding_dim) + '_lr_' + str(self.learning_rate)
        fcdm_description = self.fcdmconfig.get_description()
        description = description + '_' + fcdm_description
        return description

    def set_boring_config(self):
        # In which frequency the model should be saved (in number of iterations). Default: 10,000
        self.save_freq = 1e3
        # Whether to use all GPUs available or only one.
        # Folder(name) where checkpoints should be saved
        self.checkpoint_path = None
        # Tries to find parameter file in checkpoint path, and loads all given parameters from there
        self.load_config = True
        # If selected, no training is performed but only an evaluation will be executed.
        self.only_eval = False
        self.clean_up = False  # Whether to remove all files after finishing or not
        # Decay of learning rate of the optimizer, applied after \"lr_decay_step\" training iterations.
        self.lr_decay_factor = 0.99997
        self.lr_decay_step = 1  # Number of steps after which learning rate should be decreased
        # Minimum learning rate that should be scheduled. Default: no limit.
        self.lr_minimum = 0.0
        self.weight_decay = 0.0  # Weight decay of the optimizer",
        # Which optimizer to use. 0: SGD, 1: Adam, 2: Adamax, 3: RMSProp, 4: RAdam, 5: Adam Warmup
        self.optimizer = 4
        self.momentum = 0.0  # Apply momentum to SGD optimizer"
        self.beta1 = 0.9  # Value for beta 1 parameter in Adam-like optimizers
        self.beta2 = 0.999  # Value for beta 2 parameter in Adam-like optimizers
        # If Adam with Warmup is selected, this value determines the number of warmup iterations to use
        self.warmup = 2000

        # Value of the parameter beta which should be reached for t->infinity.
        self.beta_scheduler_end_val = 1.0
        self.beta_scheduler_start_val = 2.0  # Value of the parameter beta at t=0
        # Step size which should be used in the scheduler for beta.
        self.beta_scheduler_step_size = 5000
        # Logit which should be used in the scheduler for beta.
        self.beta_scheduler_logit = 2
        # Delay which should be used in the scheduler for.
        self.beta_scheduler_delay = 0

        # Add parameters for categorical encoding
        # If selected, variational dequantization is used for encoding categorical data.", action="store_true")
        # self.encoding_dequantization = False
        # If selected, the encoder distribution is joint over categorical variables.", action="store_true")
        # self.encoding_variational = False

        # Flow parameters
        # Number of flows used in the embedding layer.
        # self.encoding_num_flows = 0
        # Number of hidden layers of flows used in the parallel embedding layer.
        # self.encoding_hidden_layers = 2
        # Hidden size of flows used in the parallel embedding layer.
        # self.encoding_hidden_size = 128
        # Number of mixtures used in the coupling layers (if applicable).
        # self.encoding_num_mixtures = 8
