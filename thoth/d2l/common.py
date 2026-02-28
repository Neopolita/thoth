from safetensors.numpy import save_file as np_save_file
import json
import os
import torch
import torch.nn as nn

# Mistral-7B-Instruct-v0.2 architecture constants
NUM_LAYERS = 32
LORA_RANK = 8


class MLPResidualBlock(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, pre_layer_norm, post_dropout
    ):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


def _zero_lora_param_dict(target_modules, n_layers, rank, in_features, out_features):
    return nn.ParameterDict(
        {
            "A": nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.zeros(n_layers, rank, in_features[m]), requires_grad=False
                    )
                    for m in target_modules
                }
            ),
            "B": nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.zeros(n_layers, out_features[m], rank),
                        requires_grad=False,
                    )
                    for m in target_modules
                }
            ),
        }
    )


def save_mlx_adapter(
    lora_weights, output_dir, target_modules, key_fn, num_layers, rank, scale
):
    """Save LoRA weights in mlx-lm adapter format.

    Args:
        lora_weights: dict mapping module name to (A, B) tuples.
            A: [num_layers, rank, in_features]
            B: [num_layers, rank, out_features]
        output_dir: Directory to save adapter files.
        target_modules: List of target module names.
        key_fn: Callable(layer_idx, module) -> key prefix string,
                e.g. "model.layers.0.self_attn.q_proj"
        num_layers: Number of transformer layers.
        rank: LoRA rank for adapter config.
        scale: LoRA scaling factor for adapter config.

    Returns:
        Path to the saved adapter directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    adapter_weights = {}
    for module in target_modules:
        A, B = lora_weights[module]  # A: [layers, r, in], B: [layers, r, out]
        for layer_idx in range(num_layers):
            prefix = key_fn(layer_idx, module)
            a = A[layer_idx].cpu().float().numpy()  # [r, in]
            b = B[layer_idx].cpu().float().numpy()  # [r, out]
            # mlx-lm format: lora_a = [in, r], lora_b = [r, out]
            adapter_weights[f"{prefix}.lora_a"] = a.T
            adapter_weights[f"{prefix}.lora_b"] = b

    np_save_file(adapter_weights, os.path.join(output_dir, "adapters.safetensors"))

    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": rank,
            "dropout": 0.0,
            "scale": round(scale, 4),
        },
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    return output_dir
