from typing import Tuple, Callable, Any
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
debug = False

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
            # unless num elements < k then just pack the tensor
            if tensor.numel() <= self.k:
                return tensor
            original_size = tensor.size()
            if debug: print(f"Packing tensor of size {original_size}")
            flat_tensor = tensor.flatten()
            values, indices = torch.topk(flat_tensor, self.k)
            return (values, indices, original_size)

        def unpack_hook(packed_tesor: Any) -> torch.Tensor:
            # Unpacks tensor from (values, indices, original_size)
            # unless packed tensor is a tensor then just return it
            if isinstance(packed_tesor, torch.Tensor):
                return packed_tesor
            values, indices, original_size = packed_tesor
            if debug: print(f"Unpacking tensor of size {original_size}")
            tensor = torch.zeros(original_size, dtype=values.dtype, device=values.device)
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
            if debug: print("Calling wrapper forward")
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                return self.module(*args, **kwargs)

    # GPT contains nn.ModuleDict which we cannot mutate during iteration
    _recursively_wrap(model, ActivationStrategyWrapper, check_fn)

    if debug: print(f"Wrapped {sum(1 for m in model.modules() if isinstance(m, ActivationStrategyWrapper))} modules")

def _recursively_wrap(module: nn.Module, wrapper: nn.Module, check_fn: Callable):
    for name, submodule in module.named_children():
        if check_fn(submodule):
            setattr(module, name, wrapper(submodule))
        else:
            _recursively_wrap(submodule, wrapper, check_fn)