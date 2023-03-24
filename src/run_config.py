import numpy as np
import random
import torch
"""
Default config. Will usually be overriden by a task specific config class.
"""


class RunConfig():
    def __init__(self) -> None:
        self.set_fixed_config()

        self.seed = 2  # 0

        # Maximum number of epochs to train
        self.print_freq = 100  # Frequency print loss
        self.eval_freq = 500  # Frequency evaluation (with samples)
        self.learning_rate = 7.5e-4  # Learning rate of the optimizer
       # self.scale_loss = 1e-4
        self.diffusion_model = 'transformer'
        self.alpha = None

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

    def set_fixed_config(self):
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

     