# CIFAR-10 性能实验

`models.py` 定义模型和复杂度估算，`benchmark.py` 运行训练并保存数据，`roofline.py` 校准和画图。`prepare.sh`、`run_gpu.sh`、`finish.sh` 分别处理准备、GPU 测量和收尾，`run.sh` 提供一键入口和各组实验命令。

## 环境与单次运行

在 Linux/WSL 中运行，Python 3.10+。本机已配置的 WSL 环境是 Ubuntu-26.04、Python 3.12、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124：

```bash
cd '/mnt/e/大学/研三第一学期/cloud/cloudproj1'
source /home/yuidesu/.local/share/cloudproj1/venv/bin/activate
python benchmark.py --model resnet18 --environment local-wsl --device cuda --no-download
```

在其他机器上根据 [PyTorch 安装指南](https://pytorch.org/get-started/locally/) 选择与驱动匹配的 PyTorch 和 torchvision，再执行 `python -m pip install -r requirements.txt`。文件列出了 Python 依赖，Nsight 采集另外需要系统中的 `ncu` 和 GPU 计数器权限。首次下载数据：

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10('data', train=True, download=True)"
```

三个模型均使用原始 3×32×32 CIFAR-10 训练图片，ToTensor 后按 mean/std=0.5 归一化，无增强。ResNet-18 保留 torchvision 默认卷积 stem 和池化，输出类别改为 10。

```bash
# 用参数预算选择最接近的有效宽度
python benchmark.py --model vit --target-params 10000000 --vit-depth 6 --no-download
python benchmark.py --model mlp --target-params 5000000 --mlp-depth 3 --no-download

# 或直接指定宽度；不能同时指定 target-params
python benchmark.py --model vit --vit-embed-dim 256 --vit-heads 8 --no-download
python benchmark.py --model mlp --mlp-width 1024 --no-download
```

ViT 默认 6 层、8 heads、patch size 4、MLP ratio 4，dropout 为零；MLP 默认 3 个等宽 ReLU 隐藏层。不指定宽度或预算时，ViT/MLP 匹配 ResNet-18 的 11,181,642 个参数。解析后的实际尺寸和参数差异写入结果。

默认 batch 32、预热 20 步、测量 100 步、独立重复 3 次、seed 42。使用 FP32、SGD(lr=0.01, momentum=0.9)，关闭 TF32、编译和 cuDNN autotuning。每次重复按 seed + 重复编号重建模型、优化器和打乱顺序；数据集只加载一次。ResNet-18 的 batch 必须至少为 2。

正式测量可明确指定更长的设置：

```bash
python benchmark.py --model resnet18 --environment local-wsl --device cuda \
  --batch-size 32 --warmup-steps 50 --steps 1000 --repetitions 5 --no-download
```

- `--timing-scope compute-only`：循环使用默认四个已经驻留设备的真实 batch；计时包含清梯度、前向、loss、反向和 SGD 更新。GPU 用复用的 CUDA events，CPU 用 wall clock。
- `--timing-scope end-to-end`：同步 wall clock 还包含取 batch 和传输到设备。初始化、下载、复杂度估算、预热、结果保存都不计时。
- 默认显示准备、预热和测量进度，常规刷新最多每秒一次，在计时区间外更新。正式比较可用 `--no-progress` 关闭，减少步骤间输出干扰。
- `--threads`、`--num-workers`、`--data-dir`、`--output-dir` 控制资源和路径。`python benchmark.py --help` 查看所有选项。

## 结果与批量实验

在目标 GPU 机器上克隆代码、激活 Python 环境后，一条命令运行正式实验：

```bash
bash run.sh google-cloud all      # Google Cloud GPU 实例
sbatch nyu_hpc.sbatch all         # NYU HPC，按站点要求指定 account/partition
bash run.sh nyu-hpc all           # 已在 HPC GPU 计算节点的交互作业中
```

`all` 依次调用 `prepare.sh`、`run_gpu.sh` 和 `finish.sh`。HPC 可以在登录节点准备数据，只提交中间的 GPU 阶段。三个命令使用同一个结果目录：

```bash
run_dir=results/nyu-hpc-01
bash prepare.sh nyu-hpc "$run_dir"
sbatch --wait nyu_hpc.sbatch gpu "$run_dir"
bash finish.sh nyu-hpc "$run_dir"
```

- `prepare.sh` 在 CPU 上检查或下载 CIFAR-10，并创建结果目录。下载需要联网。共享存储已有数据时，用 `DATA_DIR=/path/to/data` 指向包含 `cifar-10-batches-py/` 的目录，并在提交 GPU 作业时沿用该路径。
- `run_gpu.sh` 检查 CUDA，跑六组短跑和 18 组 baseline（预热 50 步、测量 1000 步、重复 5 次，默认 1 个 CPU 线程），校准并用 Nsight 采集三个模型。已在 GPU 计算节点时，可直接执行 `bash run_gpu.sh nyu-hpc "$run_dir"`。每次 GPU 测量使用新的结果目录。
- `finish.sh` 在 CPU 上读取已保存的数据，生成性能图、汇总 CSV 和自绘 roofline。GPU 作业结束后即可运行。

把整个结果目录带回本地，里面有 `smoke/`、`baseline/`、`figures/` 和 `nsight/`。Nsight 采集要求目标节点有 `ncu` 和计数器权限。如果采集失败，GPU 阶段会返回非零状态，`finish` 仍可根据已保存的 baseline 生成性能图。可选的参数量扫描用 `size` 单独运行。

每次直接保存到 `results/环境_显卡_模型_YYYY-MM-DD_HH-MM-SS/`，重名时增加微秒：

- `steps.csv`：重复编号、step、seed、耗时、吞吐量。
- `summary.json`：配置、模型尺寸、环境信息、复杂度估算、重复统计和总体统计。

OOM 会记录失败和已测数据，不自动缩小配置。退出码 0 表示成功，2 表示失败。显存峰值在预热后重置，包含存活的训练张量和驻留 batch，不代表 DRAM 流量。

```bash
bash run.sh local-wsl smoke      # 3 个模型 × 2 种计时范围，均为短跑
bash run.sh local-wsl baseline   # 3 个模型 × batch 8/32/64 × 2 种范围
bash run.sh local-wsl size       # ViT/MLP × 1M/5M/10M/20M 参数，batch 32

# 附加参数直接传给训练脚本
bash run.sh local-wsl baseline --warmup-steps 50 --steps 1000 --repetitions 5 --no-progress

python roofline.py results results --output-dir results/figures
```

绘图输出 `combined.csv`、吞吐量、延迟、显存和模型大小图。延迟误差条表示各次重复平均耗时的标准差。只读取新格式的成功、未被 profiler 包裹的运行；旧结果文件保留但不转换。

单独运行 `smoke`、`baseline` 或 `size` 时，脚本使用当前 `python`、设备 auto、一个 CPU 线程和已有 `data/`。可用环境变量覆盖：

```bash
DEVICE=cuda THREADS=4 DATA_DIR=/path/to/data OUTPUT_DIR=results/hpc \
  bash run.sh nyu-hpc baseline
sbatch nyu_hpc.sbatch baseline
```

HPC 先申请 GPU 资源并激活环境，Slurm 的 account/partition 按站点要求填写。Google Cloud 上同样运行脚本，将标签改为 `google-cloud`。不同硬件的结果反映完整环境差异，不能单独归因于虚拟化。

## Roofline 与 Nsight

独立校准上限不需要 Nsight：

```bash
python roofline.py calibrate --environment local-wsl --device cuda --threads 1 \
  --output results/local-calibration.json
```

校准使用预热后的 FP32 GEMM 和设备内复制，默认 GEMM 2048、每个复制缓冲区 128 MiB、5 次预热、10 次测量。缓冲区应大于相关末级缓存。得到的是持续性能参考上限，部分工作负载可能超过它。

现有自绘图的数据点仍使用 Nsight 的实测计数器；独立流量估算尚未实现。FLOPs 估算覆盖卷积、线性层、注意力矩阵乘法，乘加算 2 FLOPs；训练暂按 3×前向近似。参数、梯度、momentum 和激活存储估算均不是 DRAM 流量。

Linux `ncu` 可用且计数器权限开放后：

```bash
DEVICE=cuda bash run.sh local-wsl profile --warmup-steps 50
```

这个命令先校准，再给三个模型各采集预热后的一个训练 step，batch 32。`ncu` 直接启动同一个 `benchmark.py --capture`，没有额外 worker。每个采集目录保存：

- `summary.json`、`steps.csv`：标记为 capture，其耗时不会进入普通性能图。
- `ncu.csv`：匹配 kernel launch 的 FP32 操作、DRAM 读取/写入、耗时。
- `nsight-roofline.ncu-repz`（旧版本为 `.ncu-rep`）：用 Nsight Compute GUI 打开查看内置图。
- `roofline.png`、`roofline.csv`：我们计算和绘制的图及其数据点。

也可重新画图，`--runs` 和 `--calibration` 都接受多个路径：

```bash
python roofline.py plot --runs results/实际采集目录 \
  --calibration results/local-wsl_calibration.json --output results/roofline.png
```

校准与采集的环境、GPU、CPU/线程、PyTorch/CUDA、精度必须匹配。横坐标=实测 FLOPs/DRAM bytes，纵坐标=实测 FLOPs/kernel 秒数；计算 FADD + FMUL + 2×FFMA，仅适用于当前 FP32 scalar 口径。零运算/零流量的 kernel 排除，缺失或非法数据报 `roofline-incomplete`。Nsight 重放会影响测量，不要用采集耗时替代正常训练速度。

本机此前缺少 Linux `ncu`，且宿主机计数器权限受限。安装包在 `.tools/installers/`；WSL 的计数器权限需在 Windows NVIDIA 设置中开放。CPU 硬件计数器采集未实现。仅有训练耗时或显存占用不能生成完整的实测 roofline。
