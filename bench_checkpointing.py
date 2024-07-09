"""
A much shorter version of train.py for benchmarking with activation checkpointing
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT
from datetime import datetime
from torch.autograd.profiler import record_function
from pytorch_memlab.utils import readable_size
from custom_timeline import custom_memory_timeline
import functools

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = False
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
profile = True # use pytorch profiler, or else generate a snapshot
log_dir = "./bench_log"
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

init_distributed(0, 1)

if not profile:
    torch.cuda.memory._record_memory_history(max_entries=100_000)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
time_format_str: str = "%b_%d_%H_%M_%S"
filename = f"checkpoint_batch_{batch_size}_block_{block_size}"
timestamp = datetime.now().strftime(time_format_str)
filename = f"{filename}_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# data loading init
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# model init
gptconf = GPTConfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layer = 12, n_head = 12, n_embd = 768, # size of the model
    dropout = 0, # for determinism
    bias = bias,
)
model = GPT(gptconf)
model.to(device)

def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")

wrap_class_str = 'Block'
wrap_class = get_block_class_from_model(model, wrap_class_str)
model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)
rank = 0
shared_fsdp_kwargs = dict(
    auto_wrap_policy=model_auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=False),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=rank,
    ignored_modules=None,
    limit_all_gathers=False,
    use_orig_params=True,
    sync_module_states=False
)

print('Sharding policy...')
# mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
mp_dtype = None
policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
model = FSDP(model, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

try:
    # use activation checkpointing, according to:
    # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
    #
    # first, verify we have FSDP activation support ready by importing:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        apply_activation_checkpointing,
        CheckpointImpl,
    )
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
except Exception as e:
    print('FSDP activation checkpointing not available:', e)
else:
    check_fn = lambda submodule: isinstance(submodule, wrap_class)
    print('Applying activation checkpointing wrapper to policy...')
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
    print('FSDP activation checkpointing enabled!')

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0

if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active

    # trace handler from: https://pytorch.org/blog/understanding-gpu-memory-1/?ref=alexdremov.me
    def trace_handler(prof: torch.profiler.profile):
        # Construct the memory timeline image.
        env_title = f"torch version: {torch.__version__}, torch.compile: {compile}"
        title = f"Checkpointing at {wrap_class_str} Modules" \
                f"\n{env_title}"
        custom_memory_timeline(profile=prof, 
                            path=f"{log_dir}/{filename}",
                            device_str="cuda:0",
                            title=title,
                            ignore_categories=None)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True, # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False, # only for torchscript models atm
    ) as prof:
        
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                loss = model(X, Y)[1]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step() # notify the profiler at end of each step

else:

    # generate snapshot
    wait, warmup, active = 1, 1, 1
    num_steps = wait + warmup + active

    X, Y = get_batch('train')
    for k in range(num_steps):
        with ctx:
            loss = model(X, Y)[1]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lossf = loss.item()
        print(f"{k}/{num_steps} loss: {lossf:.4f}")

    # Save memory snapshot
    try:
       torch.cuda.memory._dump_snapshot(f"{log_dir}/{filename}.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")

    # Stop recording memory snapshot history.
    torch.cuda.memory._record_memory_history(enabled=None)