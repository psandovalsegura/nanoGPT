# (c) Meta Platforms, Inc. and affiliates. 
import logging
from datetime import datetime, timedelta

import torch

from torch.autograd.profiler import record_function
from custom_timeline import custom_memory_timeline

from contextlib import nullcontext
from model import GPTConfig, GPT

logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
LOG_DIR = "./bench_log"
def trace_handler(prof: torch.profiler.profile, warmup=False, batch_size=None, block_size=None):
   # Prefix for file names.
   filename = f"bench_gpt_batch_{batch_size}_block_{block_size}"
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{filename}_{timestamp}"

   # Construct the trace file.
   # prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   # prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
   if not warmup:
      custom_memory_timeline(profile=prof, 
                              path=f"{LOG_DIR}/{file_prefix}",
                              device_str="cuda:0",
                              ignore_categories=None)

def run_gpt(num_iters=5, device="cuda:0", warmup=False):
   # -----------------------------------------------------------------------------
   batch_size = 64
   block_size = 256
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
   
   # alternatively, if fixed data is desired to not care about data loading
   x = torch.randint(50304, (batch_size, block_size), device=device)
   y = torch.randint(50304, (batch_size, block_size), device=device)
   get_batch = lambda split: (x, y)
   X, Y = get_batch("train")
   
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

   with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
       record_shapes=True,
       profile_memory=True,
       with_stack=True,
       on_trace_ready=lambda p: trace_handler(p, warmup=warmup, batch_size=batch_size, block_size=block_size),
   ) as prof:
       for _ in range(num_iters):
           prof.step()
           with record_function("## forward ##"):
               logits, loss = model(X, Y)

           with record_function("## backward ##"):
               loss.backward()

           with record_function("## optimizer ##"):
               optimizer.step()
               optimizer.zero_grad(set_to_none=True)

if __name__ == "__main__":
    # Warm up
    run_gpt(warmup=True)
    # Run the GPT model
    run_gpt()