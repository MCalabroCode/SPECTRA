import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set seed for reproducibility across:
    - python
    - numpy
    - torch (cpu + cuda)

    deterministic=True makes training slower but reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True