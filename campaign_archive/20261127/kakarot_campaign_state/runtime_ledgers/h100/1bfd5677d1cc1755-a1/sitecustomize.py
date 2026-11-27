"""Keep concurrent campaign jobs from oversubscribing host CPU threads."""
import os
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
