"""Calibrate ceilings, plot raw Nsight counters, or compare saved training runs."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import collect_metadata, configure_runtime, seed_everything, synchronize, write_csv, write_json


NCU_METRICS = (
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
)


def _number(value):
    value = float((value or "").replace(",", "").strip())
    if not math.isfinite(value) or value < 0:
        raise ValueError("Counter values must be finite and nonnegative")
    return value


def parse_ncu_csv(path):
    """Read Nsight CSV with either metric columns or individual metric rows."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header = next((i for i, line in enumerate(lines)
                   if ("Metric Name" in line and "Metric Value" in line)
                   or ("Kernel Name" in line and "Process ID" in line and "Stream" in line)), None)
    if header is None:
        raise ValueError("Nsight CSV has no raw metric header; check ncu.csv for permissions/unsupported metrics")
    rows = csv.DictReader(io.StringIO("\n".join(lines[header:])))
    id_columns = ("Process ID", "Context", "Stream", "ID")
    launches = {}
    if "Metric Name" in rows.fieldnames:
        for row in rows:
            metric = row.get("Metric Name")
            if metric not in NCU_METRICS:
                continue
            key = tuple(row.get(name, "") for name in id_columns)
            launch = launches.setdefault(key, {"kernel": row.get("Kernel Name", "unknown"), "metrics": {}, "units": {}})
            if metric in launch["metrics"]:
                raise ValueError(f"Duplicate metric {metric} for launch {key}")
            launch["metrics"][metric] = _number(row.get("Metric Value"))
            launch["units"][metric] = (row.get("Metric Unit") or "").strip()
    else:
        missing = set(NCU_METRICS) - set(rows.fieldnames)
        if missing:
            raise ValueError(f"Missing counter columns: {sorted(missing)}")
        unit_row = next(rows, {})
        units = {name: (unit_row.get(name) or "").strip() for name in NCU_METRICS}
        for row in rows:
            if not row.get("Kernel Name"):
                continue
            key = tuple(row.get(name, "") for name in id_columns)
            if key in launches:
                raise ValueError(f"Duplicate kernel launch {key}")
            launches[key] = {"kernel": row["Kernel Name"],
                             "metrics": {name: _number(row.get(name)) for name in NCU_METRICS},
                             "units": units}
    if not launches:
        raise ValueError("No requested hardware counters were collected")
    points = []
    time_units = {"second": 1.0, "s": 1.0, "msecond": 1e-3, "ms": 1e-3,
                  "usecond": 1e-6, "us": 1e-6, "nsecond": 1e-9, "ns": 1e-9}
    for key, launch in launches.items():
        metrics, units = launch["metrics"], launch["units"]
        missing = set(NCU_METRICS) - set(metrics)
        if missing:
            raise ValueError(f"Missing counters for kernel {launch['kernel']}: {sorted(missing)}")
        if units[NCU_METRICS[0]] not in time_units:
            raise ValueError(f"Unsupported duration unit: {units[NCU_METRICS[0]]}")
        for name in NCU_METRICS[1:3]:
            if units[name] not in ("byte", "bytes", "B"):
                raise ValueError("Expected base byte units; collect with --print-units base")
        for name in NCU_METRICS[3:]:
            if units[name] not in ("inst", "instruction", "instructions"):
                raise ValueError("Expected base instruction units; collect with --print-units base")
        seconds = metrics[NCU_METRICS[0]] * time_units[units[NCU_METRICS[0]]]
        if seconds <= 0:
            raise ValueError("Kernel duration must be positive")
        flops = metrics[NCU_METRICS[3]] + metrics[NCU_METRICS[4]] + 2 * metrics[NCU_METRICS[5]]
        traffic = metrics[NCU_METRICS[1]] + metrics[NCU_METRICS[2]]
        points.append({"launch_id": ":".join(key), "kernel": launch["kernel"], "flops": flops,
                       "dram_bytes": traffic, "seconds": seconds, "raw_metrics": metrics,
                       "plot_eligible": flops > 0 and traffic > 0,
                       "exclusion_reason": None if flops > 0 and traffic > 0 else "zero FP32 work or zero DRAM traffic"})
    return points


def roofline_point(flops, dram_bytes, seconds):
    if any(not math.isfinite(v) or v <= 0 for v in (flops, dram_bytes, seconds)):
        raise ValueError("FLOPs, DRAM bytes, and duration must be finite and positive")
    return {"arithmetic_intensity": flops / dram_bytes, "flops_per_second": flops / seconds}


def roofline_limit(intensity, peak_flops, bandwidth):
    if any(not math.isfinite(v) or v <= 0 for v in (intensity, peak_flops, bandwidth)):
        raise ValueError("Intensity, compute ceiling, and bandwidth must be finite and positive")
    return min(peak_flops, bandwidth * intensity)


def calibrate(device_name="auto", environment="local", threads=None, gemm_size=2048,
              buffer_mib=128, warmup=5, iterations=10):
    import torch
    if min(gemm_size, buffer_mib, iterations) <= 0 or warmup < 0:
        raise ValueError("Calibration sizes/iterations must be positive and warmup nonnegative")
    device = configure_runtime(device_name, threads)
    seed_everything(42)

    def measure(operation):
        for _ in range(warmup):
            operation()
        synchronize(device)
        times = []
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(iterations):
            if device.type == "cuda":
                start.record()
                operation()
                end.record()
                end.synchronize()
                seconds = start.elapsed_time(end) / 1000
            else:
                start = time.perf_counter()
                operation()
                seconds = time.perf_counter() - start
            if seconds <= 0:
                raise ValueError("Calibration timer returned a nonpositive duration")
            times.append(seconds)
        return times

    with torch.no_grad():
        a = torch.randn(gemm_size, gemm_size, device=device)
        b = torch.randn_like(a)
        c = torch.empty_like(a)
        compute_times = measure(lambda: torch.mm(a, b, out=c))
        del a, b, c
        elements = buffer_mib * 1024 * 1024 // 4
        source = torch.ones(elements, dtype=torch.float32, device=device)
        target = torch.empty_like(source)
        memory_times = measure(lambda: target.copy_(source))
    return {"status": "calibrated", "metadata": collect_metadata(environment, device),
            "precision": "float32", "peak_flops_per_second": 2 * gemm_size**3 / statistics.median(compute_times),
            "memory_bandwidth_bytes_per_second": 2 * elements * 4 / statistics.median(memory_times),
            "method": "empirical sustained FP32 GEMM and device-local copy; median warm timings; lower-bound ceilings, not vendor peaks",
            "gemm_size": gemm_size, "buffer_mib_per_array": buffer_mib,
            "compute_seconds": compute_times, "copy_seconds": memory_times,
            "cache_caveat": "Copy working set must exceed the relevant last-level cache; otherwise this is a cache-bandwidth ceiling"}


def metadata_key(metadata):
    return tuple(metadata[key] for key in ("environment", "gpu", "gpu_memory_bytes", "cpu", "threads",
                                          "torch", "cuda_runtime", "precision", "tf32"))


def plot_rooflines(runs, calibration_paths, output):
    calibrations = {}
    for path in calibration_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["status"] != "calibrated" or data["precision"] != "float32":
            raise ValueError("A completed FP32 calibration is required")
        roofline_limit(1, data["peak_flops_per_second"], data["memory_bandwidth_bytes_per_second"])
        key = metadata_key(data["metadata"])
        if key in calibrations:
            raise ValueError("Provide only one calibration for each hardware/environment configuration")
        calibrations[key] = data
    points, used = [], {}
    excluded = 0
    for run in runs:
        run = Path(run)
        data = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if data["status"] != "success" or data["mode"] != "capture":
            raise ValueError(f"A successful --capture run is required: {run}")
        key = metadata_key(data["metadata"])
        if key not in calibrations:
            raise ValueError(f"No matching environment/hardware/software calibration for {run}")
        used[key] = calibrations[key]
        for kernel in parse_ncu_csv(run / "ncu.csv"):
            if not kernel["plot_eligible"]:
                excluded += 1
                continue
            points.append({"run": str(run), "environment": key[0], "gpu": key[1],
                           "model": data["model"]["model"], "launch_id": kernel["launch_id"],
                           "kernel": kernel["kernel"], "flops": kernel["flops"],
                           "dram_bytes": kernel["dram_bytes"], "seconds": kernel["seconds"],
                           **roofline_point(kernel["flops"], kernel["dram_bytes"], kernel["seconds"])})
    if not points:
        raise ValueError("No kernels with both nonzero FP32 work and DRAM traffic")
    ridges = [d["peak_flops_per_second"] / d["memory_bandwidth_bytes_per_second"] for d in used.values()]
    x = np.geomspace(min(min(p["arithmetic_intensity"] for p in points) / 4, min(ridges) / 10),
                     max(max(p["arithmetic_intensity"] for p in points) * 4, max(ridges) * 10), 300)
    figure, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for key, data in used.items():
        y = np.minimum(data["peak_flops_per_second"], x * data["memory_bandwidth_bytes_per_second"])
        ax.loglog(x, y / 1e9, label=f"{key[0]}/{key[1]} empirical ceiling")
    for run in sorted({p["run"] for p in points}):
        selected = [p for p in points if p["run"] == run]
        first = selected[0]
        ax.scatter([p["arithmetic_intensity"] for p in selected], [p["flops_per_second"] / 1e9 for p in selected],
                   s=20, alpha=0.65, label=f"{first['environment']}/{first['model']} ({Path(run).name})")
    ax.set(xscale="log", yscale="log", xlabel="FP32 FLOP / DRAM byte", ylabel="FP32 GFLOP/s",
           title="Kernel roofline: measured scalar work and empirical ceilings")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize="x-small")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    write_csv(output.with_suffix(".csv"), points)
    print(f"roofline complete: {output}; excluded {excluded} zero-work/traffic kernels")


def plot_results(inputs, output_dir):
    paths = set()
    for source in inputs:
        path = Path(source)
        if not path.exists():
            raise ValueError(f"Result source does not exist: {path}")
        paths.update(path.rglob("summary.json") if path.is_dir() else [path])
    rows = []
    for path in sorted(paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "success" or data.get("mode") != "benchmark" or not data.get("summary"):
            print(f"Skipping {path}: not a successful unprofiled benchmark")
            continue
        config, summary, metadata = data["config"], data["summary"], data["metadata"]
        peak = summary["peak_cuda_allocated_bytes"]
        rows.append({"result_file": str(path), "environment": config["environment"], "gpu": metadata["gpu"],
                     "model": config["model"], "batch_size": config["batch_size"], "timing_scope": config["timing_scope"],
                     "actual_params": data["model"]["actual_params"], "requested_params": data["model"]["requested_params"],
                     "mean_step_ms": summary["mean_step_seconds"] * 1000,
                     "repetition_std_ms": summary["repetition_mean_seconds_std"] * 1000,
                     "images_per_second": summary["images_per_second"],
                     "peak_cuda_allocated_mib": peak / 2**20 if peak is not None else None})
    if not rows:
        raise ValueError("No successful unprofiled summaries in the new format")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "combined.csv", rows)
    labels = [f"{r['environment']}/{r['gpu']}/{r['model']}\nb={r['batch_size']} {r['timing_scope']} {r['actual_params']/1e6:.2f}M" for r in rows]
    for field, ylabel, filename in (("images_per_second", "Throughput (images/s)", "throughput.png"),
                                    ("mean_step_ms", "Step latency (ms)", "latency.png"),
                                    ("peak_cuda_allocated_mib", "Peak CUDA allocated memory (MiB)", "memory.png")):
        figure, ax = plt.subplots(figsize=(max(8, min(30, len(rows) * 0.7)), 6), layout="constrained")
        selected = [(i, r) for i, r in enumerate(rows) if r[field] is not None]
        if selected:
            ax.bar([i for i, _ in selected], [r[field] for _, r in selected],
                   yerr=[r["repetition_std_ms"] for _, r in selected] if field == "mean_step_ms" else None, capsize=3)
        else:
            ax.text(0.5, 0.5, "CUDA memory measurement unavailable", ha="center", transform=ax.transAxes)
        ax.set_xticks(range(len(rows)), labels, rotation=90, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
    figure, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    keys = ("environment", "gpu", "model", "batch_size", "timing_scope")
    for group in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected = sorted([r for r in rows if tuple(r[k] for k in keys) == group], key=lambda r: r["actual_params"])
        ax.plot([r["actual_params"] / 1e6 for r in selected], [r["images_per_second"] for r in selected],
                "o-", label=f"{group[0]}/{group[1]}/{group[2]} b={group[3]} {group[4]}")
    ax.set(xlabel="Actual parameters (millions)", ylabel="Throughput (images/s)", title="Model-size scaling")
    ax.grid(alpha=0.2)
    ax.legend(fontsize="x-small")
    figure.savefig(output / "model_size.png", dpi=160)
    plt.close(figure)
    print(f"Figures and combined.csv saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("calibrate")
    c.add_argument("--device", default="auto")
    c.add_argument("--environment", default="local")
    c.add_argument("--threads", type=int)
    c.add_argument("--gemm-size", type=int, default=2048)
    c.add_argument("--buffer-mib", type=int, default=128)
    c.add_argument("--warmup", type=int, default=5)
    c.add_argument("--iterations", type=int, default=10)
    c.add_argument("--output", required=True)
    p = commands.add_parser("plot")
    p.add_argument("--runs", nargs="+", required=True, help="Capture directories containing summary.json and ncu.csv")
    p.add_argument("--calibration", nargs="+", required=True)
    p.add_argument("--output", required=True)
    r = commands.add_parser("results")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    try:
        if args.command == "calibrate":
            write_json(args.output, calibrate(args.device, args.environment, args.threads, args.gemm_size,
                                             args.buffer_mib, args.warmup, args.iterations))
            print(f"Calibration saved to {args.output}")
        elif args.command == "plot":
            plot_rooflines(args.runs, args.calibration, args.output)
        else:
            plot_results(args.inputs, args.output_dir)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"{'roofline-incomplete' if args.command == 'plot' else 'error'}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


