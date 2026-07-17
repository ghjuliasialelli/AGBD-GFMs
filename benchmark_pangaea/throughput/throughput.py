"""Measure inference throughput (samples/sec) of a model on a given dataset.

Usage:
    python throughput.py \
        task=regression dataset=agbdlite encoder=gfmswin \
        decoder=reg_upernet preprocessing=reg_default criterion=mse \
        batch_size=64 \
        --warmup 20 --iterations 100

    # Use random dummy data (no dataset loading):
    python throughput.py \
        task=regression dataset=agbdlite encoder=gfmswin \
        decoder=reg_upernet preprocessing=reg_default criterion=mse \
        batch_size=64 --dummy

All Hydra overrides (task, dataset, encoder, etc.) work as in run.py.
Extra CLI flags (--warmup, --iterations, --dummy) control the benchmark.
"""

import argparse
import sys
import threading
import time

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from pangaea.datasets.base import GeoFMDataset, RawGeoFMDataset
from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder
from pangaea.utils.collate_fn import get_collate_fn


def parse_extra_args():
    """Parse --warmup, --iterations, and --dummy before Hydra consumes the rest."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--warmup", type=int, default=20,
                        help="Number of warm-up forward passes (discarded).")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of timed forward passes.")
    parser.add_argument("--dummy", action="store_true",
                        help="Use random dummy data instead of the real dataset.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for benchmarking.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of DataLoader workers.")
    args, remaining = parser.parse_known_args()
    # Put remaining args back so Hydra can parse them
    sys.argv = [sys.argv[0]] + remaining
    return args


extra_args = parse_extra_args()


def make_dummy_batch(encoder, batch_size, device):
    """Create a single random batch matching the encoder's expected input."""
    input_size = encoder.input_size
    image = {}
    for modality, bands in encoder.input_bands.items():
        image[modality] = torch.randn(batch_size, len(bands), input_size, input_size,
                                      device=device)
    target = torch.randn(batch_size, 1, input_size, input_size, device=device)
    return image, target


class PrefetchIterator:
    """Prefetches the next batch on a background thread so data loading
    is overlapped with GPU compute."""

    def __init__(self, loader, device):
        self._loader = loader
        self._device = device
        self._loader_iter = iter(loader)
        self._next_batch = None
        self._thread = None
        # Kick off the first prefetch
        self._prefetch()

    def _load_next(self):
        try:
            data = next(self._loader_iter)
        except StopIteration:
            self._loader_iter = iter(self._loader)
            data = next(self._loader_iter)
        image = {k: v.to(self._device, non_blocking=True) for k, v in data["image"].items()}
        target = data["target"].to(self._device, non_blocking=True)
        self._next_batch = (image, target)

    def _prefetch(self):
        self._thread = threading.Thread(target=self._load_next, daemon=True)
        self._thread.start()

    def next(self):
        """Return the prefetched batch and start loading the next one."""
        self._thread.join()
        image, target = self._next_batch
        self._prefetch()
        return image, target


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    warmup = extra_args.warmup
    iterations = extra_args.iterations
    dummy = extra_args.dummy
    batch_size = extra_args.batch_size
    num_workers = extra_args.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("WARNING: No GPU detected. Throughput numbers on CPU are not meaningful.")
        exit(1)

    # ── Build model ──────────────────────────────────────────────────────
    encoder: Encoder = instantiate(cfg.encoder)

    #if encoder.model_name != "Prithvi":
    #   encoder.load_encoder_weights(None)  # None logger → silent

    decoder: Decoder = instantiate(cfg.decoder, encoder=encoder)
    decoder.to(device)
    decoder.eval()

    n_params = sum(p.numel() for p in decoder.parameters())
    n_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Model        : {decoder.model_name}  (encoder: {encoder.model_name})")
    print(f"Parameters   : {n_params:,} total, {n_trainable:,} trainable")

    if dummy:
        # ── Dummy data mode ──────────────────────────────────────────────
        image, target = make_dummy_batch(encoder, batch_size, device)
        output_shape = target.shape[-2:]

        print(f"Data         : dummy (random tensors)")
        print(f"Batch size   : {batch_size}")
        print(f"Warm-up      : {warmup} forward passes")
        print(f"Iterations   : {iterations} forward passes")
        print()

        print("Warming up …")
        with torch.no_grad():
            for _ in range(warmup):
                _ = decoder(image, output_shape=output_shape)

        if device.type == "cuda":
            torch.cuda.synchronize()

        print("Benchmarking …")
        if device.type == "cuda":
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = decoder(image, output_shape=output_shape)

    else:
        # ── Real dataset mode (with prefetching) ─────────────────────────
        preprocessor = instantiate(
            cfg.preprocessing.test,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )
        raw_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
        dataset = GeoFMDataset(raw_dataset, preprocessor)

        modalities = list(encoder.input_bands.keys())
        collate_fn = get_collate_fn(modalities)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            shuffle=True,
        )

        print(f"Dataset      : {cfg.dataset.dataset_name}  (split=test, {len(dataset)} samples)")
        print(f"Batch size   : {batch_size}")
        print(f"Warm-up      : {warmup} forward passes")
        print(f"Iterations   : {iterations} forward passes")
        print()

        prefetcher = PrefetchIterator(loader, device)

        print("Warming up …")
        with torch.no_grad():
            for _ in range(warmup):
                image, target = prefetcher.next()
                _ = decoder(image, output_shape=target.shape[-2:])

        if device.type == "cuda":
            torch.cuda.synchronize()

        print("Benchmarking …")
        if device.type == "cuda":
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                image, target = prefetcher.next()
                _ = decoder(image, output_shape=target.shape[-2:])

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t_start

    # ── Report ───────────────────────────────────────────────────────────
    total_samples = iterations * batch_size
    throughput = total_samples / elapsed
    ms_per_sample = (elapsed / total_samples) * 1000
    ms_per_batch = (elapsed / iterations) * 1000

    print()
    print("═" * 50)
    print(f"  Total time     : {elapsed:.2f} s")
    print(f"  Batches        : {iterations}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Throughput     : {throughput:.1f} samples/s")
    print(f"  Latency/sample : {ms_per_sample:.2f} ms")
    print(f"  Latency/batch  : {ms_per_batch:.2f} ms")
    print("═" * 50)

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"  Peak GPU mem   : {peak_mem:.2f} GB")
        print("═" * 50)


if __name__ == "__main__":
    main()
