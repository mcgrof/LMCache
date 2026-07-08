# SPDX-License-Identifier: Apache-2.0
"""KV-cache byte geometry for real model configs.

Python port of the size math in ``examples/kv_cache_calculator`` (the web
calculator's ``kv_cache_calculator.html``), sharing its ``modelconfig.json``.
Given a model config and a token count it returns the KV-cache byte size,
handling the four families the calculator supports:

  * default (MHA / GQA): ``2 * layers * tokens * kv_heads * head_size`` where
    ``head_size = hidden_size / num_attention_heads``;
  * GQA with an explicit ``head_dim`` (Qwen3, GLM-4.x, Hunyuan dense):
    ``2 * layers * tokens * kv_heads * head_dim``;
  * DeepSeek MLA (V3/V3.1/V3.2, R1): a single latent, no factor of two, no
    kv-head axis -- ``layers * tokens * (kv_lora_rank + qk_rope_head_dim)``;
  * Hunyuan-Large CLA: ``2 * (layers / cla_share_factor) * tokens * kv_heads *
    head_size`` (every ``cla_share_factor`` layers share one KV cache).

This is the geometry the IO workload generator uses to size fake KV blocks from
real model dimensions -- see ``run_kv_offload_io.py``.
"""
from __future__ import annotations

DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    # storage codecs sometimes offload V at 1 byte; accept the common spellings
    "fp8": 1,
    "float8": 1,
}


def model_family(model_name: str) -> str:
    """Classify a model name into a KV-geometry family (mirrors the calculator)."""
    m = model_name
    ml = model_name.lower()
    if m.startswith("deepseek-ai/DeepSeek-V3") or m == "deepseek-ai/DeepSeek-R1":
        return "mla"
    if ml == "tencent/hunyuan-large":
        return "cla"
    if (
        "qwen/qwen3-" in ml
        or m.startswith("zai-org/GLM-4.")
        or (ml.startswith("tencent/hunyuan-") and ml != "tencent/hunyuan-large")
    ):
        return "gqa_head_dim"
    return "default"


def kv_cache_bytes(model_name: str, config: dict, tokens: int, dtype: str):
    """Return ``(total_bytes, detail)`` for ``tokens`` tokens of KV cache.

    ``detail`` carries the family and the resolved dimensions so callers can
    print/record exactly how the size was derived (same fields the calculator
    shows).  Byte count is integer; head_size/effective_layers are exact for
    the shipped configs (integer divisions) but rounded defensively.
    """
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; know {sorted(DTYPE_BYTES)}")
    dbytes = DTYPE_BYTES[dtype]
    fam = model_family(model_name)
    nl = int(config["num_hidden_layers"])
    detail = {"family": fam, "dtype": dtype, "dtype_bytes": dbytes,
              "tokens": tokens, "num_hidden_layers": nl}

    if fam == "mla":
        kv_lora_rank = int(config["kv_lora_rank"])
        qk_rope_head_dim = int(config["qk_rope_head_dim"])
        elements = nl * tokens * (kv_lora_rank + qk_rope_head_dim)
        detail.update(kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim)
    elif fam == "cla":
        cla = int(config["cla_share_factor"])
        eff = nl / cla
        head_size = config["hidden_size"] / config["num_attention_heads"]
        kv_heads = int(config["num_key_value_heads"])
        elements = 2 * eff * tokens * kv_heads * head_size
        detail.update(cla_share_factor=cla, effective_layers=eff,
                      num_key_value_heads=kv_heads, head_size=head_size)
    elif fam == "gqa_head_dim":
        head_dim = int(config["head_dim"])
        kv_heads = int(config["num_key_value_heads"])
        elements = 2 * nl * tokens * kv_heads * head_dim
        detail.update(num_key_value_heads=kv_heads, head_dim=head_dim)
    else:  # default MHA / GQA
        head_size = config["hidden_size"] / config["num_attention_heads"]
        kv_heads = int(config["num_key_value_heads"])
        elements = 2 * nl * tokens * kv_heads * head_size
        detail.update(hidden_size=config["hidden_size"],
                      num_attention_heads=config["num_attention_heads"],
                      num_key_value_heads=kv_heads, head_size=head_size)

    total_bytes = int(round(elements * dbytes))
    detail["total_elements"] = int(round(elements))
    detail["total_bytes"] = total_bytes
    return total_bytes, detail


if __name__ == "__main__":
    # Self-test against known calculator outputs (float16).
    llama8b = {"hidden_size": 4096, "num_attention_heads": 32,
               "num_hidden_layers": 32, "num_key_value_heads": 8}
    b, _ = kv_cache_bytes("meta-llama/Llama-3.1-8B-Instruct", llama8b, 4096, "float16")
    assert b == 536870912, b  # 0.5 GiB
    qwen3 = {"hidden_size": 5120, "num_attention_heads": 64, "num_hidden_layers": 64,
             "num_key_value_heads": 8, "head_dim": 128}
    b2, d2 = kv_cache_bytes("Qwen/Qwen3-32B", qwen3, 4096, "float16")
    assert d2["family"] == "gqa_head_dim" and b2 == 1073741824, (d2, b2)  # 1.0 GiB
    hl = {"hidden_size": 6400, "num_attention_heads": 80, "num_hidden_layers": 64,
          "num_key_value_heads": 8, "cla_share_factor": 2}
    b3, d3 = kv_cache_bytes("tencent/Hunyuan-Large", hl, 4096, "float16")
    assert d3["family"] == "cla" and d3["effective_layers"] == 32, d3
    print("kv_geometry self-test OK:",
          f"Llama-3.1-8B@4096={b/1024**3:.3f}GiB, Qwen3-32B@4096={b2/1024**3:.3f}GiB")
