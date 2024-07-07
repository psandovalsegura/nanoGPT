from typing import Tuple, Callable, Any
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------

class ActivationStrategy(ABC):
    @abstractmethod
    def get_hooks(self) -> Tuple[Callable, Callable]:
        """
        An ActivationStrategy should return two hooks with the following signatures:
        - pack_hook(tensor: Tensor) -> Any
        - unpack_hook(Any) -> Tensor
        where the return value of pack_hook is a valid input to unpack_hook.
        """
        pass

class TopKActivation(ActivationStrategy):
    """
    Keep top k activations.
    """
    def __init__(self, k: int):
        self.k = k

    def get_hooks(self) -> Tuple[Callable, Callable]:
        def pack_hook(tensor: torch.Tensor) -> Any:
            # Packs tensor as (values, indices, original_size)
            original_size = tensor.size()
            flat_tensor = tensor.flatten()
            values, indices = torch.topk(flat_tensor, self.k)
            return (values, indices, original_size)

        def unpack_hook(packed_tesor: Any) -> torch.Tensor:
            values, indices, original_size = packed_tesor
            tensor = torch.zeros(original_size, device=values.device)
            tensor.put_(indices, values)
            return tensor

        return pack_hook, unpack_hook

# -----------------------------------------------------------------------------

def apply_activation_strategy(model: nn.Module, strategy: ActivationStrategy, check_fn=lambda _: True):
    """
    Apply ActivationStrategy to modules within `model` based on a user-defined configuration.

    For each module within `model`, the `check_fn` is used to decide
    whether `module` should be wrapped or not.

    Note::
        This function modifies `model` in place and replaces the forward method with an
        equivalent forward that uses the torch.autograd.graph.saved_tensors_hooks
        context manager.
    """
    pack_hook, unpack_hook = strategy.get_hooks()

    class ActivationStrategyWrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                return self.module(*args, **kwargs)

    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if check_fn(module):
            setattr(model, name, ActivationStrategyWrapper(module))
    