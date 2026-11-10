"""Short CIFAR-10 training runs. Run `python benchmark.py --help`."""

import argparse
import csv
from datetime import datetime
import gc
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model, estimate_complexity, resolve_model_size


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["resnet18", "vit", "mlp"], default="resnet18")
    p.add_argument("--environment", default="local")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--timing-scope", choices=["compute-only", "end-to-end"], default="compute-only")
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resident-batches", type=int, default=4)
    p.add_argument("--threads", type=int)
    p.add_argument("--target-params", type=int)
    p.add_argument("--vit-depth", type=int, default=6)
    p.add_argument("--vit-heads", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--vit-embed-dim", type=int)
    p.add_argument("--vit-mlp-ratio", type=float, default=4.0)
    p.add_argument("--mlp-depth", type=int, default=3)
    p.add_argument("--mlp-width", type=int)
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--capture", action="store_true", help="Mark measured steps for an external ncu capture")
    p.add_argument("--run-dir", help="Exact result directory; used by run.sh to colocate Nsight files")
    args = p.parse_args()
    for key in ("batch_size", "steps", "repetitions", "resident_batches", "threads", "target_params",
                "vit_depth", "vit_heads", "patch_size", "vit_embed_dim", "mlp_depth", "mlp_width"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            p.error(f"{key} must be positive")
    if min(args.warmup_steps, args.num_workers, args.seed) < 0 or args.seed >= 2**32:
        p.error("warmup/workers must be nonnegative and seed must be in [0, 2**32)")
    if not math.isfinite(args.vit_mlp_ratio) or args.vit_mlp_ratio <= 0:
        p.error("vit-mlp-ratio must be finite and positive")
    if not args.environment.strip():
        p.error("environment cannot be empty")
    if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", args.device):
        p.error("device must be auto, cpu, cuda, or cuda:N")
    if args.model == "resnet18" and (args.batch_size < 2 or any(
            x is not None for x in (args.target_params, args.vit_embed_dim, args.mlp_width))):
        p.error("ResNet-18 requires batch-size >= 2 and has fixed dimensions")
    if args.model == "vit":
        if 32 % args.patch_size or (args.vit_embed_dim and args.vit_embed_dim % args.vit_heads):
            p.error("patch-size must divide 32 and embedding width must be divisible by heads")
        if args.mlp_width is not None:
            p.error("mlp-width applies only to MLP")
    elif args.vit_embed_dim is not None:
        p.error("vit-embed-dim applies only to ViT")
    if args.target_params is not None and (args.vit_embed_dim is not None or args.mlp_width is not None):
        p.error("target-params and explicit width are mutually exclusive")
    if args.capture and (args.repetitions != 1 or args.timing_scope != "compute-only"):
        p.error("capture requires repetitions=1 and timing-scope=compute-only")
    return args


def configure_runtime(device_name, threads=None):
    if threads is not None:
        torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def new_run_directory(environment, model, device, output_dir):
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    label = re.sub(r"[^\w.-]+", "-", f"{environment}_{gpu}_{model}").strip("-.")
    while True:
        now = datetime.now().astimezone()
        stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        directory = Path(output_dir) / f"{label}_{stamp}"
        if directory.exists():
            directory = Path(output_dir) / f"{label}_{stamp}-{now.microsecond:06d}"
        try:
            directory.mkdir(parents=True, exist_ok=False)
            return directory
        except FileExistsError:
            continue


def collect_metadata(environment, device):
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    driver = None
    if device.type == "cuda":
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return {"created_at": datetime.now().astimezone().isoformat(), "environment": environment,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
            "cpu": cpu, "logical_cpus": os.cpu_count(), "device": str(device), "threads": torch.get_num_threads(),
            "hostname": platform.node(), "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "driver": driver, "precision": "float32", "tf32": False}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def progress(total, label, enabled):
    """Yield step numbers; display updates happen after the timed step returns."""
    started = updated = time.monotonic()
    tty = sys.stderr.isatty()
    def show(count):
        elapsed = time.monotonic() - started
        eta = f"{elapsed * (total - count) / count:.0f}s" if count else "?"
        filled = 20 * count // total
        text = f"{label} [{'#' * filled}{'-' * (20 - filled)}] {count}/{total} elapsed={elapsed:.1f}s ETA={eta}"
        print(f"\r{text}\033[K" if tty else text, end="" if tty else "\n", file=sys.stderr, flush=True)
    if enabled and total:
        show(0)
    try:
        for index in range(total):
            yield index
            if enabled and (time.monotonic() - updated >= 1 or index + 1 == total):
                show(index + 1)
                updated = time.monotonic()
    finally:
        if enabled and total and tty:
            print(file=sys.stderr, flush=True)


def batches(loader, device):
    iterator = iter(loader)  # Initial iterator/worker setup stays outside timing, even with zero warmup.
    def stream():
        current = iterator
        while True:
            for images, targets in current:
                yield (images.to(device, non_blocking=device.type == "cuda"),
                       targets.to(device, non_blocking=device.type == "cuda"))
            current = iter(loader)
    return stream()


def training_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(batch[0]), batch[1])
    loss.backward()
    optimizer.step()
    return loss.detach()


def summarize(rows, repetitions, batch_size):
    if not rows:
        return {}
    times = [r["seconds"] for r in rows]
    means = [r["mean_step_seconds"] for r in repetitions]
    return {"measured_steps": len(times), "completed_repetitions": len(repetitions),
            "mean_step_seconds": statistics.mean(times), "median_step_seconds": statistics.median(times),
            "step_seconds_std": statistics.stdev(times) if len(times) > 1 else 0,
            "images_per_second": batch_size * len(times) / sum(times),
            "repetition_mean_seconds_std": statistics.stdev(means) if len(means) > 1 else 0,
            "peak_cuda_allocated_bytes": max((r["peak_cuda_allocated_bytes"] or 0 for r in repetitions), default=0) or None}


def main():
    args = parse_args()
    rows = []
    directory = None
    result = {"config": vars(args), "mode": "capture" if args.capture else "benchmark", "status": "error",
              "repetitions": [], "roofline": {"status": "roofline-incomplete",
              "reason": "Measured DRAM counters and a matching calibration are required; use roofline.py plot"}}
    try:
        device = configure_runtime(args.device, args.threads)
        if args.run_dir:
            directory = Path(args.run_dir)
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "summary.json").exists():
                raise FileExistsError("run-dir already contains summary.json; choose a new directory")
        else:
            directory = new_run_directory(args.environment, args.model, device, args.output_dir)
        result["metadata"] = collect_metadata(args.environment, device)
        if args.capture and device.type != "cuda":
            raise ValueError("capture requires CUDA")
        spec = resolve_model_size(args)
        result["model"] = spec
        if args.progress:
            print(f"{args.model}: loading CIFAR-10...", file=sys.stderr, flush=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        dataset = datasets.CIFAR10(args.data_dir, train=True, download=args.download, transform=transform)
        if args.batch_size > len(dataset):
            raise ValueError("batch-size exceeds the CIFAR-10 training split")
        for repetition in range(args.repetitions):
            seed = (args.seed + repetition) % 2**32
            seed_everything(seed)
            label = f"{args.model} repeat {repetition + 1}/{args.repetitions}"
            if args.progress:
                print(f"{label}: preparing model and batches...", file=sys.stderr, flush=True)
            with torch.device(device):
                model, _ = build_model(args, spec)
                model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0, foreach=False)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda",
                                persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(seed))
            source = batches(loader, device)
            if args.timing_scope == "compute-only":
                source = itertools.cycle([next(source) for _ in range(min(args.resident_batches, len(loader)))])
            if repetition == 0:
                result["complexity"] = estimate_complexity(model, args.batch_size, device)
            for _ in progress(args.warmup_steps, f"{label} warmup", args.progress):
                training_step(model, optimizer, next(source))
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            current = []
            if args.capture:
                torch.cuda.cudart().cudaProfilerStart()
            try:
                for index in progress(args.steps, f"{label} measured", args.progress):
                    if args.timing_scope == "compute-only":
                        batch = next(source)  # Resident lookup is outside timing.
                    synchronize(device)
                    if device.type == "cuda" and args.timing_scope == "compute-only":
                        start_event.record()
                    else:
                        started = time.perf_counter()
                    if args.timing_scope == "end-to-end":
                        batch = next(source)  # Fetch and transfer are inside timing.
                    loss = training_step(model, optimizer, batch)
                    if device.type == "cuda" and args.timing_scope == "compute-only":
                        end_event.record()
                        end_event.synchronize()
                        seconds = start_event.elapsed_time(end_event) / 1000
                    else:
                        synchronize(device)
                        seconds = time.perf_counter() - started
                    if seconds <= 0:
                        raise RuntimeError("Timer returned a nonpositive duration")
                    row = {"repetition": repetition, "step": index, "seed": seed,
                           "seconds": seconds, "images_per_second": args.batch_size / seconds}
                    current.append(row)
                    rows.append(row)
            finally:
                if args.capture:
                    torch.cuda.cudart().cudaProfilerStop()
            final_loss = float(loss.cpu())
            if not math.isfinite(final_loss):
                raise RuntimeError("Training produced a nonfinite loss")
            mean = statistics.mean(r["seconds"] for r in current)
            result["repetitions"].append({"repetition": repetition, "seed": seed, "mean_step_seconds": mean,
                "images_per_second": args.batch_size / mean, "final_loss": final_loss,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
            del model, optimizer, loader, source, batch, loss
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result["status"] = "success"
    except (ValueError, RuntimeError, MemoryError, OSError) as exc:
        # Never overwrite a previous run when an exact directory was reused.
        if isinstance(exc, FileExistsError):
            print(str(exc), file=sys.stderr)
            return 2
        oom = isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) or "out of memory" in str(exc).lower() or "can't allocate memory" in str(exc).lower()
        result.update(status="out-of-memory" if oom else "error", error=str(exc))
    if directory is None:
        directory = new_run_directory(args.environment, args.model, torch.device("cpu"), args.output_dir)
    result["summary"] = summarize(rows, result["repetitions"], args.batch_size)
    result["timing_method"] = "CUDA events with per-step completion" if result.get("metadata", {}).get("device", "").startswith("cuda") and args.timing_scope == "compute-only" else "synchronized host wall clock"
    write_csv(directory / "steps.csv", rows, ["repetition", "step", "seed", "seconds", "images_per_second"])
    write_json(directory / "summary.json", result)
    print(f"status={result['status']} results={directory}")
    if result.get("model"):
        print(f"parameters={result['model']['actual_params']} requested={result['model']['requested_params']}")
    if result["summary"]:
        print(f"mean_step_ms={result['summary']['mean_step_seconds'] * 1000:.3f} images/s={result['summary']['images_per_second']:.2f}")
    if result.get("error"):
        print(result["error"], file=sys.stderr)
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
