"""
A much shorter version of train.py for benchmarking
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

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = False
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
profile = True # use pytorch profiler, or just simple benchmarking?
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def log_mem(desc):
    mem = torch.cuda.memory_allocated()
    print(f"==> {desc}: {readable_size(mem)}")


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
    TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
    LOG_DIR = "./bench_log"
    # Prefix for file names.
    filename = f"bench_batch_{batch_size}_block_{block_size}"
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{filename}_{timestamp}"
    os.makedirs(LOG_DIR, exist_ok=True)
    def trace_handler(prof: torch.profiler.profile):
        # Construct the trace file.
        # prof.export_chrome_trace(f"{LOG_DIR}/{file_prefix}.json.gz")

        # Construct the memory timeline file.
        # prof.export_memory_timeline(f"{LOG_DIR}/{file_prefix}.html", device="cuda:0")
        custom_memory_timeline(profile=prof, 
                            path=f"{LOG_DIR}/{file_prefix}",
                            device_str="cuda:0",
                            ignore_categories=None)
                            # ignore_categories=['None'])
                            # ignore_categories=['None', 'Category.PARAMETER', 'Category.OPTIMIZER_STATE', 'Category.INPUT', 'Category.TEMPORARY', 'Category.AUTOGRAD_DETAIL', 'Category.GRADIENT'])   

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
        # log_mem("Before get batch")
        X, Y = get_batch('train')
        # log_mem("After get batch")
        for k in range(num_steps):
            with ctx:
                # log_mem(f"Step {k} before forward")
                logits, loss = model(X, Y)
                # log_mem(f"Step {k} after forward")
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            # log_mem(f"Step before backward")
            loss.backward()
            # log_mem(f"Step after backward")
            optimizer.step()
            # log_mem(f"Step after optim step")
            lossf = loss.item()
            # print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step() # notify the profiler at end of each step

else:

    # simple benchmarking
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
        t0 = time.time()
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1-t0
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        if stage == 1:
            print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
