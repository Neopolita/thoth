from huggingface_hub import snapshot_download
from thoth.d2l.common import LORA_RANK, NUM_LAYERS, save_mlx_adapter
from thoth.logger import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import types

# Article: https://pub.sakana.ai/doc-to-lora/
# Doc-to-LoRa Paper: https://arxiv.org/abs/2602.15902
# Doc-to-LoRa repository: https://github.com/SakanaAI/Doc-to-LoRA
# Doc-to-LoRa networks: https://huggingface.co/SakanaAI/doc-to-lora/tree/main/mistral_7b_d2l

logger = get_logger()

# D2L-specific constants
TARGET_MODULES = ["down_proj"]
IN_FEATURES = {"down_proj": 14336}  # intermediate_size
OUT_FEATURES = {"down_proj": 4096}  # hidden_size
D_LORA = IN_FEATURES["down_proj"] + OUT_FEATURES["down_proj"]  # 18432

# Perceiver config (verified from checkpoint tensor shapes)
LATENT_SIZE = 512
N_LATENT_QUERIES = 8
NUM_ENCODER_BLOCKS = 9
NUM_DECODER_BLOCKS = 1
N_HEADS = 16  # q_proj: [2048, 512] → 2048 / 128 = 16
N_KV_HEADS = 4  # k_proj: [512, 512] → 512 / 128 = 4
HEAD_DIM = 128

EFFECTIVE_RANK = LORA_RANK * 2  # 16 (8 generated + 8 bias)

# D2L uses rslora: alpha / sqrt(r) = 16 / sqrt(8) ≈ 5.657
# (model_loading.py: use_rslora=True, lora_alpha = r * 2 = 16)
LORA_ALPHA = 16
D2L_SCALING = LORA_ALPHA / math.sqrt(LORA_RANK)
HIDDEN_SIZE = 4096
CTX_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


# Perceiver modules
# Matches Idefics2-based architecture in SakanaAI/Doc-to-LoRA
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps  # type: ignore

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class GatedMLP(nn.Module):
    def __init__(self, in_size, intermediate_size, out_size):
        super().__init__()
        self.gate_proj = nn.Linear(in_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, out_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PerceiverAttention(nn.Module):
    """Cross-attention with GQA. Q from latents, KV from concat(latents, context)."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(LATENT_SIZE, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(LATENT_SIZE, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(LATENT_SIZE, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, LATENT_SIZE, bias=False)

    def forward(self, latents, context):
        bs, n_q, _ = latents.shape
        kv_input = torch.cat([latents, context], dim=-2)
        n_kv = kv_input.shape[1]

        q = self.q_proj(latents).view(bs, n_q, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(kv_input).view(bs, n_kv, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(kv_input).view(bs, n_kv, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # Repeat KV heads for GQA
        repeats = N_HEADS // N_KV_HEADS
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(bs, n_q, N_HEADS * HEAD_DIM)
        return self.o_proj(attn)


class PerceiverLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_latents_layernorm = RMSNorm(LATENT_SIZE)
        self.input_context_layernorm = RMSNorm(LATENT_SIZE)
        self.self_attn = PerceiverAttention()
        self.post_attention_layernorm = RMSNorm(LATENT_SIZE)
        self.pre_ff_layernorm = RMSNorm(LATENT_SIZE)
        self.mlp = GatedMLP(LATENT_SIZE, LATENT_SIZE * 4, LATENT_SIZE)
        self.post_ff_layernorm = RMSNorm(LATENT_SIZE)

    def forward(self, latents, context):
        latents_normed = self.input_latents_layernorm(latents)
        context_normed = self.input_context_layernorm(context)
        latents = latents + self.post_attention_layernorm(
            self.self_attn(latents_normed, context_normed)
        )
        latents = latents + self.post_ff_layernorm(
            self.mlp(self.pre_ff_layernorm(latents))
        )
        return latents


class PerceiverResampler(nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        self.latents_q = nn.Parameter(torch.randn(N_LATENT_QUERIES, LATENT_SIZE))
        self.layers = nn.ModuleList([PerceiverLayer() for _ in range(n_blocks)])
        self.layernorm = RMSNorm(LATENT_SIZE)

    def forward(self, context):
        bs = context.shape[0]
        latents = self.latents_q.unsqueeze(0).expand(bs, -1, -1)
        for layer in self.layers:
            latents = layer(latents, context)
        return self.layernorm(latents)


class D2LPerceiver(nn.Module):
    """Modality projection + encoder (9 blocks) + decoder (1 block)."""

    def __init__(self):
        super().__init__()
        self.modality_projection = GatedMLP(HIDDEN_SIZE, HIDDEN_SIZE * 4, LATENT_SIZE)
        self.encoder = PerceiverResampler(NUM_ENCODER_BLOCKS)
        self.decoder = PerceiverResampler(NUM_DECODER_BLOCKS)

    def forward(self, hidden_states):
        """[32, seq_len, 4096] → [32, 8, 512]"""
        projected = self.modality_projection(hidden_states)
        encoded = self.encoder(projected)
        return self.decoder(encoded)


# EinMix layers
# Per-layer linear transformations (replaces einops.EinMix dependency)
class EinMixLinear(nn.Module):
    """Per-layer linear: y[l] = x[l] @ weight[l] + bias[l]"""

    def __init__(self, n_layers, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_layers, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(1, n_layers, 1, 1, d_out))

    def forward(self, x):
        # x: [batch, n_layers, n_samples, n_queries, d_in]
        return torch.einsum("blnqd,ldo->blnqo", x, self.weight) + self.bias


class ResMLPBlockPerLayer(nn.Module):
    """Per-layer residual MLP.

    State dict keys: layers.{0=LN, 1=EinMix_up, 2=SiLU, 3=EinMix_down, 4=LN}
    """

    def __init__(self, n_layers, d_model, d_hidden):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.LayerNorm(d_model),  # 0
                EinMixLinear(n_layers, d_model, d_hidden),  # 1
                nn.SiLU(),  # 2
                EinMixLinear(n_layers, d_hidden, d_model),  # 3
                nn.LayerNorm(d_model),  # 4
            ]
        )

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        return residual + x


class EinMixHead(nn.Module):
    """Per-layer projection head (no bias). weight: [n_layers, d_in, d_out]"""

    def __init__(self, n_layers, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_layers, d_in, d_out))

    def forward(self, x):
        return torch.einsum("blnqd,ldo->blnqo", x, self.weight)


# HyperLoRA
class HyperLoRA(nn.Module):
    """D2L hypernetwork: Perceiver aggregator + per-layer MLP + EinMix head.

    Generates LoRA A, B matrices from per-layer hidden state activations.
    ~574M params. Module structure matches checkpoint state_dict keys.
    """

    def __init__(self):
        super().__init__()
        self.aggregator = nn.ModuleDict({"perceiver": D2LPerceiver()})
        self.layers = nn.ModuleList(
            [
                ResMLPBlockPerLayer(NUM_LAYERS, LATENT_SIZE, LATENT_SIZE * 4),
            ]
        )
        self.head = EinMixHead(NUM_LAYERS, LATENT_SIZE, D_LORA)

        # Bias and scaler parameters (concatenated with generated LoRA for effective rank 16)
        self.bias_A = nn.ParameterDict(
            {
                "down_proj": nn.Parameter(
                    torch.zeros(NUM_LAYERS, LORA_RANK, IN_FEATURES["down_proj"])
                ),
            }
        )
        self.bias_B = nn.ParameterDict(
            {
                "down_proj": nn.Parameter(
                    torch.zeros(NUM_LAYERS, LORA_RANK, OUT_FEATURES["down_proj"])
                ),
            }
        )
        self.scaler_A = nn.ParameterDict(
            {
                "down_proj": nn.Parameter(torch.ones(1, NUM_LAYERS, LORA_RANK, 1)),
            }
        )
        self.scaler_B = nn.ParameterDict(
            {
                "down_proj": nn.Parameter(torch.ones(1, NUM_LAYERS, LORA_RANK, 1)),
            }
        )

    @torch.no_grad()
    def forward(self, hidden_states):
        """Generate LoRA from per-layer activations.

        Args:
            hidden_states: [32, seq_len, 4096] per-layer activations from Mistral-7B
        Returns:
            {"down_proj": (A, B)} where A=[32, 16, 14336], B=[32, 16, 4096]
        """
        # Perceiver: [32, seq_len, 4096] → [32, 8, 512]
        compressed = self.aggregator["perceiver"](hidden_states)

        # Reshape: [32, 8, 512] → [1, 32, 1, 8, 512]
        x = compressed.unsqueeze(0).unsqueeze(2)

        # Per-layer MLP
        for layer in self.layers:
            x = layer(x)

        # L2 normalize
        x = F.normalize(x, dim=-1)

        # Head: [1, 32, 1, 8, 512] → [1, 32, 1, 8, 18432]
        x = self.head(x)

        # Squeeze to [32, 8, 18432]
        x = x.squeeze(0).squeeze(1)

        # Split into A and B
        d_in = IN_FEATURES["down_proj"]
        A_flat = x[..., :d_in]  # [32, 8, 14336]
        B_flat = x[..., d_in:]  # [32, 8, 4096]

        # Scale
        scaler_A = self.scaler_A["down_proj"].squeeze(0)  # [32, 8, 1]
        scaler_B = self.scaler_B["down_proj"].squeeze(0)  # [32, 8, 1]
        A_scaled = A_flat * scaler_A
        B_scaled = B_flat * scaler_B

        # Concatenate with bias → effective rank 16
        A = torch.cat([A_scaled, self.bias_A["down_proj"]], dim=1)  # [32, 16, 14336]
        B = torch.cat([B_scaled, self.bias_B["down_proj"]], dim=1)  # [32, 16, 4096]

        return {"down_proj": (A, B)}


# Context encoding
def _encode_context(text, tokenizer, model, device, max_len=512):
    """Extract per-layer hidden states from frozen Mistral-7B.
    Reference uses chat template: system="", user=context, add_generation_prompt=True.
    Returns: [32, seq_len, 4096]
    """
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": text.strip()},
    ]
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    token_ids = torch.tensor([result.input_ids[:max_len]], dtype=torch.long)
    tokens = {
        "input_ids": token_ids.to(device),
        "attention_mask": torch.ones_like(token_ids).to(device),
    }

    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)

    # hidden_states: tuple of 33 tensors (embedding + 32 layers)
    # Reference removes last transformer block, keeps embedding.
    # Equivalent: take [:-1] = [embedding, layer_0, ..., layer_30] (32 tensors)
    hidden_states = torch.stack(
        outputs.hidden_states[:-1], dim=0
    )  # [32, 1, seq_len, 4096]
    # hidden_states = torch.stack(outputs.hidden_states[1:], dim=0)  # [32, 1, seq_len, 4096]
    return hidden_states.squeeze(1).float()  # [32, seq_len, 4096]


# Stub modules for checkpoint unpickling
class _StubModule(types.ModuleType):
    """Module that returns stub classes for any attribute access (for pickle)."""

    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []  # make it a package

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        # Return a stub class that accepts any constructor args
        return type(name, (), {"__init__": lambda self, *a, **kw: None})


def _register_stub_modules():
    """Pre-register ctx_to_lora.* stub modules for checkpoint unpickling."""
    submodules = [
        "ctx_to_lora",
        "ctx_to_lora.configs",
        "ctx_to_lora.modeling",
        "ctx_to_lora.modeling.hypernet",
        "ctx_to_lora.modeling.aggregator",
        "ctx_to_lora.modeling.aggregator.idefics2",
        "ctx_to_lora.modeling.aggregator.perceiver",
        "ctx_to_lora.modeling.einmix",
        "ctx_to_lora.modeling.layers",
    ]
    for mod_name in submodules:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _StubModule(mod_name)


# Main entry point
def process_doc_to_lora(
    text: str,
    output_dir: str = "adapters/d2l",
    d2l_dir: str | None = None,
) -> str:
    """Generate a LoRA adapter from document text using Doc-to-LoRA.

    Uses the pretrained SakanaAI D2L Perceiver hypernetwork for Mistral-7B
    to encode document knowledge into LoRA weights in a single forward pass.

    Args:
        text: Document text to encode.
        output_dir: Where to save the adapter files.
        d2l_dir: Path to D2L checkpoint directory. If None, downloads from HF.

    Returns:
        Path to the saved adapter directory.
    """
    device = torch.device("cpu")

    # 1. Download D2L checkpoint if needed
    if d2l_dir is None:
        logger.info("Downloading D2L checkpoint from SakanaAI/doc-to-lora...")
        cache_dir = snapshot_download(
            "SakanaAI/doc-to-lora",
            allow_patterns=["mistral_7b_d2l/*", "mistral_7b_d2l/checkpoint-20000/*"],
        )
        d2l_dir = os.path.join(cache_dir, "mistral_7b_d2l")
    logger.info(f"D2L checkpoint: {d2l_dir}")

    # 2. Build and load HyperLoRA
    logger.info("Loading HyperLoRA...")
    _register_stub_modules()
    hyper = HyperLoRA()

    ckpt_path = os.path.join(d2l_dir, "checkpoint-20000", "pytorch_model.bin")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(d2l_dir, "pytorch_model.bin")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Filter to tensor keys only (checkpoint may contain pickled config objects)
    state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}

    info = hyper.load_state_dict(state_dict, strict=False)
    if info.unexpected_keys:
        logger.debug(f"Unexpected keys: {info.unexpected_keys}")
    if info.missing_keys:
        logger.debug(f"Missing keys: {info.missing_keys}")
    hyper.eval().to(device)
    logger.info(
        f"HyperLoRA loaded ({sum(p.numel() for p in hyper.parameters()):,} params)"
    )

    # 3. Load context encoder (frozen Mistral-7B, float16 ~14GB)
    logger.info(f"Loading context encoder: {CTX_MODEL_NAME}")
    ctx_tokenizer = AutoTokenizer.from_pretrained(CTX_MODEL_NAME)
    ctx_model = AutoModelForCausalLM.from_pretrained(
        CTX_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
    ).eval()

    # 4. Encode document → per-layer activations
    logger.info(f"Encoding document ({len(text)} chars)...")
    hidden_states = _encode_context(text, ctx_tokenizer, ctx_model, device)
    logger.info(f"Hidden states: {hidden_states.shape}")

    # Free context encoder (~14GB)
    del ctx_model, ctx_tokenizer

    # 5. Generate LoRA
    logger.info("Generating LoRA from document...")
    lora_weights = hyper(hidden_states)

    # 6. Save as mlx-lm adapter (effective rank 16)
    def _d2l_key(layer_idx, module):
        return f"model.layers.{layer_idx}.mlp.{module}"

    save_mlx_adapter(
        lora_weights,
        output_dir,
        TARGET_MODULES,
        _d2l_key,
        NUM_LAYERS,
        EFFECTIVE_RANK,
        D2L_SCALING,
    )

    logger.info(f"Adapter saved to {output_dir}")
    logger.info(
        f"  Scale: {D2L_SCALING:.4f}, Rank: {EFFECTIVE_RANK}, Targets: {TARGET_MODULES}"
    )

    del hyper, checkpoint, lora_weights
    return output_dir
