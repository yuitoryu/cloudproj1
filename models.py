"""Exact parameter formulas avoid allocating candidate models during sizing."""

from functools import lru_cache

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.vision_transformer import VisionTransformer


def mlp_parameters(width, depth, input_dim=3072, classes=10):
    return (input_dim + 1) * width + (depth - 1) * (width + 1) * width + (width + 1) * classes


def vit_parameters(width, mlp_dim, depth, patch_size=4, image_size=32, classes=10):
    tokens = (image_size // patch_size) ** 2 + 1
    block = 4 * width**2 + 2 * width * mlp_dim + 9 * width + mlp_dim
    return depth * block + width * (3 * patch_size**2 + tokens + classes + 4) + classes


def nearest_width(target, count, multiple=1):
    """Find adjacent valid widths by binary search; ties choose the smaller model."""
    low, high = 1, 1
    while count(high * multiple) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if count(middle * multiple) < target:
            low = middle + 1
        else:
            high = middle
    candidates = {max(1, low - 1) * multiple, low * multiple}
    return min(candidates, key=lambda width: (abs(count(width) - target), width))


@lru_cache(maxsize=1)
def reference_parameters():
    with torch.device("meta"):
        return sum(p.numel() for p in resnet18(weights=None, num_classes=10).parameters())


def resolve_model_size(config):
    if config.model == "resnet18":
        return {"model": "resnet18", "stem": "torchvision-default", "num_classes": 10,
                "actual_params": reference_parameters(), "requested_params": None,
                "relative_difference": None, "sizing": "fixed"}
    explicit = config.vit_embed_dim if config.model == "vit" else config.mlp_width
    target = config.target_params if config.target_params is not None else (reference_parameters() if explicit is None else None)
    if config.model == "vit":
        def mlp_dim(width):
            return max(1, round(width * config.vit_mlp_ratio))
        def count(width):
            return vit_parameters(width, mlp_dim(width), config.vit_depth, config.patch_size)
        width = explicit if explicit is not None else nearest_width(target, count, config.vit_heads)
        architecture = {"embed_dim": width, "mlp_dim": mlp_dim(width), "depth": config.vit_depth,
                        "heads": config.vit_heads, "patch_size": config.patch_size, "dropout": 0.0}
    else:
        def count(width):
            return mlp_parameters(width, config.mlp_depth)
        width = explicit if explicit is not None else nearest_width(target, count)
        architecture = {"width": width, "depth": config.mlp_depth, "activation": "relu", "input_dim": 3072}
    actual = count(width)
    return {"model": config.model, "image_size": 32, "num_classes": 10, **architecture,
            "actual_params": actual, "requested_params": target,
            "relative_difference": (actual - target) / target if target is not None else None,
            "sizing": "explicit" if explicit is not None else "closest-valid-width"}


def build_model(config, resolved=None):
    resolved = resolved or resolve_model_size(config)
    if config.model == "resnet18":
        model = resnet18(weights=None, num_classes=10)
    elif config.model == "vit":
        model = VisionTransformer(image_size=32, patch_size=resolved["patch_size"],
                                  num_layers=resolved["depth"], num_heads=resolved["heads"],
                                  hidden_dim=resolved["embed_dim"], mlp_dim=resolved["mlp_dim"],
                                  dropout=0.0, attention_dropout=0.0, num_classes=10)
    else:
        layers, features = [nn.Flatten()], 3072
        for _ in range(resolved["depth"]):
            layers.extend([nn.Linear(features, resolved["width"]), nn.ReLU()])
            features = resolved["width"]
        model = nn.Sequential(*layers, nn.Linear(features, 10))
    actual = sum(p.numel() for p in model.parameters())
    if actual != resolved["actual_params"]:
        raise RuntimeError(f"Parameter formula mismatch: estimated {resolved['actual_params']}, constructed {actual}")
    return model, resolved


def estimate_complexity(model, batch_size, device):
    totals = {"forward_flops": 0, "layer_output_bytes": 0, "attention_intermediate_bytes": 0}
    excluded = {id(m.out_proj) for m in model.modules() if isinstance(m, nn.MultiheadAttention)}
    handles = []

    def hook(module, inputs, output):
        x = inputs[0]
        y = output[0] if isinstance(output, tuple) else output
        if isinstance(y, torch.Tensor):
            totals["layer_output_bytes"] += y.numel() * y.element_size()
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            totals["forward_flops"] += 2 * y.numel() * kernel
        elif isinstance(module, nn.Linear):
            totals["forward_flops"] += 2 * y.numel() * module.in_features
        elif isinstance(module, nn.MultiheadAttention):
            b, n, d = x.shape
            totals["forward_flops"] += 8 * b * n * d * d + 4 * b * n * n * d
            # Unfused Q/K/V and attention-score/probability storage proxy.
            totals["attention_intermediate_bytes"] += (3 * b * n * d + 2 * b * module.num_heads * n * n) * x.element_size()

    for module in model.modules():
        if id(module) in excluded:
            continue
        if isinstance(module, nn.MultiheadAttention) or not list(module.children()):
            handles.append(module.register_forward_hook(hook))
    previous = model.training
    try:
        model.eval()
        # Keep attention on its ordinary module path so hooks/formulas have consistent coverage.
        mha_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.no_grad():
                model(torch.zeros(batch_size, 3, 32, 32, device=device))
        finally:
            torch.backends.mha.set_fastpath_enabled(mha_fastpath)
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    gradient_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return {"parameters": parameters, "trainable_parameters": trainable,
            "parameter_bytes": parameter_bytes, "gradient_bytes_estimate": gradient_bytes,
            "optimizer_state_bytes_estimate": gradient_bytes,
            "activation_bytes_estimate": totals["layer_output_bytes"] + totals["attention_intermediate_bytes"],
            "forward_flops_estimate": totals["forward_flops"],
            "backward_flops_estimate": 2 * totals["forward_flops"],
            "training_flops_estimate": 3 * totals["forward_flops"],
            "flop_convention": "multiply-add = 2 FLOPs",
            "flop_coverage": "Conv2d, Linear, and dense self-attention matrix products; excludes bias, normalization, activations, softmax, loss, and optimizer",
            "training_approximation": "backward ~= 2 * forward; training ~= 3 * forward, not hardware-measured",
            "activation_method": "sum of leaf-layer outputs plus unfused attention intermediates; storage proxy, not peak allocation or DRAM traffic",
            "optimizer": "SGD momentum=0.9: one FP32 momentum tensor per trainable parameter"}
