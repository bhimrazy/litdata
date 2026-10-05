import faulthandler
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank):
    faulthandler.dump_traceback_later(60, exit=True)
    t = time.time()
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29533", rank=rank, world_size=2)
    print(f"rank{rank} init {time.time() - t:.2f}s", flush=True)
    t = time.time()
    dist.barrier()
    print(f"rank{rank} barrier {time.time() - t:.2f}s", flush=True)
    x = torch.ones(1)
    t = time.time()
    dist.all_reduce(x)
    print(f"rank{rank} all_reduce={x.item()} {time.time() - t:.2f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    print("GLOO_SOCKET_IFNAME =", os.environ.get("GLOO_SOCKET_IFNAME"), "torch", torch.__version__, flush=True)
    mp.spawn(run, nprocs=2)
    print("gloo OK", flush=True)
