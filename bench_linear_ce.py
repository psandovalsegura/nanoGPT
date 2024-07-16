"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
import torch.nn as nn
from model import GPTConfig, GPT
from datetime import datetime
from torch.autograd.profiler import record_function
from pytorch_memlab.utils import readable_size
from custom_timeline import custom_memory_timeline

# -----------------------------------------------------------------------------
batch_size = 128
block_size = 128
bias = False
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
profile = True # use pytorch profiler, or else generate a snapshot
log_dir = "./bench_log"
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

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
filename = f"linearce_batch_{batch_size}_block_{block_size}"
timestamp = datetime.now().strftime(time_format_str)
filename = f"{filename}_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# data loading init
n_embd = 768
vocab_size = 50304
x = torch.randn(batch_size, block_size, n_embd).to(ptdtype).to(device)
y = torch.randint(vocab_size, (batch_size, block_size), device=device)
get_batch = lambda split: (x, y)

# model init
model = nn.Linear(n_embd, 50304).to(ptdtype).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
loss_fn = nn.CrossEntropyLoss()

def f(m, x, label):
    linear_outputs = m(x)
    linear_outputs_reshaped = linear_outputs.view(-1, vocab_size)
    label_reshaped = label.view(-1)
    return loss_fn(linear_outputs_reshaped, label_reshaped)

if compile:
    print("Compiling model...")
    f = torch.compile(f)

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
        title = f"Standard Training Iterations" \
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
                loss = f(model, X, Y)
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
            loss = f(model, X, Y)
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