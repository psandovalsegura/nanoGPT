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
batch_size = 128
block_size = 512
bias = False
real_data = False
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
filename = f"nograd_batch_{batch_size}_block_{block_size}"
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

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

# because optimizer will not initialize in no_grad env, we create a tensor of the same size
# adamw uses 8 bytes for each parameter
# num_model_params = model.get_num_params(non_embedding=False)
# optimizer_tensor = torch.randn(num_model_params * 8, dtype=torch.float32, device=device)
# model.synthetic_optimizer_tensor = optimizer_tensor
# print(f"Using {readable_size(num_model_params * 8)} for substitute optimizer state")
optimizer.load_state_dict(torch.load(f"optimizer_batch_{batch_size}_block_{block_size}.pth"))

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
        title = f"No Grad Iterations" \
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
            with torch.no_grad():
                with ctx:
                    loss = model(X, Y)
                # loss.backward()
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
        with torch.no_grad():
            with ctx:
                loss = model(X, Y)
            # loss.backward()
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