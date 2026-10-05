"""Replays tests/streaming/test_cache.py::test_cache_for_image_dataset_distributed with stage timings."""

import faulthandler
import os
import sys
import tempfile
import time
from functools import partial

sys.path.insert(0, os.getcwd())
import tests.streaming.test_cache as tc

T0 = time.time()


def log(msg):
    print(f"[pid {os.getpid()} +{time.time() - T0:6.1f}s] {msg}", flush=True)


_orig_iter = tc.CacheDataLoader.__iter__


def _iter(self):
    log(f"DataLoader iter start (shuffle={getattr(self.batch_sampler, '_shuffle', '?')})")
    n = 0
    for b in _orig_iter(self):
        n += 1
        yield b
    log(f"DataLoader iter end, {n} batches")


tc.CacheDataLoader.__iter__ = _iter


def run(fabric, tmpdir):
    faulthandler.dump_traceback_later(int(os.environ.get("DUMP_AFTER", "90")), exit=False)
    _b = fabric.barrier

    def barrier(*a, **k):
        log("barrier enter")
        r = _b(*a, **k)
        log("barrier exit")
        return r

    fabric.barrier = barrier
    log(f"rank {fabric.global_rank} start")
    tc._cache_for_image_dataset(2, tmpdir, fabric=fabric)
    log(f"rank {fabric.global_rank} done")


if __name__ == "__main__":
    from lightning.fabric import Fabric

    log(f"GLOO_SOCKET_IFNAME={os.environ.get('GLOO_SOCKET_IFNAME')}")
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "cache"))
    Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn").launch(partial(run, tmpdir=d))
    log("ALL DONE")
