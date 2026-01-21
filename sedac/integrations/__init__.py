"""
SEDAC Integrations Module
=========================

vLLM patch integrations:
- vllm_v6_patch: V6.0 compatible patch (hard exit)
- vllm_v61_patch: V6.1 patch with Rust integration (soft exit)
"""

from sedac.integrations.vllm_patcher import (
    VLLMPatcher,
    PatchConfig,
    apply_sedac_patch,
    remove_sedac_patch,
)

__all__ = [
    "VLLMPatcher",
    "PatchConfig",
    "apply_sedac_patch",
    "remove_sedac_patch",
]
