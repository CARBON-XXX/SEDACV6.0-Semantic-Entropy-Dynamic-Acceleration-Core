"""
SEDAC vLLM Patcher
==================

Unified patcher for vLLM integration supporting both V6.0 and V6.1 modes.

Features:
- Auto-detection of vLLM installation path
- Support for Qwen2/Qwen2.5 architectures
- Version-aware patching (V6.0 hard exit, V6.1 soft exit)
- Safe patch/unpatch operations
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class PatchVersion(Enum):
    """SEDAC patch versions."""
    V60 = "6.0"
    V61 = "6.1"


@dataclass
class PatchConfig:
    """Configuration for vLLM patching."""
    version: PatchVersion = PatchVersion.V61
    target_path: Optional[str] = None
    probe_dir: str = "sedac_data"
    probe_layers: Tuple[int, ...] = (7, 14, 21)
    probe_rank: int = 64
    soft_exit: bool = True
    confidence_decay: float = 0.9
    
    @classmethod
    def from_env(cls) -> "PatchConfig":
        """Create config from environment variables."""
        version_str = os.environ.get("SEDAC_VERSION", "6.1")
        version = PatchVersion.V61 if "6.1" in version_str else PatchVersion.V60
        
        layers_str = os.environ.get("SEDAC_PROBE_LAYERS", "7,14,21")
        layers = tuple(int(x.strip()) for x in layers_str.split(",") if x.strip())
        
        return cls(
            version=version,
            target_path=os.environ.get("SEDAC_VLLM_QWEN2_PATH"),
            probe_dir=os.environ.get("SEDAC_PROBE_DIR", "sedac_data"),
            probe_layers=layers,
            probe_rank=int(os.environ.get("SEDAC_PROBE_RANK", "64")),
            soft_exit=os.environ.get("SEDAC_SOFT_EXIT", "1").lower() in ("1", "true"),
            confidence_decay=float(os.environ.get("SEDAC_CONFIDENCE_DECAY", "0.9")),
        )


# =============================================================================
# Shared Code Components
# =============================================================================

PROBE_DEFINITION = '''
import os
import torch
import numpy as np
SEDAC_PATCH_VERSION = {version}

class LREProbe(nn.Module):
    """Low-Rank Entropy Probe for semantic uncertainty prediction."""
    def __init__(self, input_dim, rank=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, rank, bias=False)
        self.norm = nn.LayerNorm(rank)
        self.head = nn.Linear(rank, 1)
        self.act = nn.Softplus()

    def forward(self, x):
        h = self.proj(x)
        h = self.norm(h)
        return self.act(self.head(h))

# Try to import modular SEDAC components
try:
    from sedac.core import CascadeController, CascadeConfig
    from sedac.calibration import DualMetricCalibration
    from sedac.metrics import SEDACMetricsCollector
    _SEDAC_MODULAR = True
except ImportError:
    _SEDAC_MODULAR = False

# Try Rust core
try:
    from sedac_core import CascadeController as RustCascadeController
    _SEDAC_RUST = True
except ImportError:
    _SEDAC_RUST = False
'''

INIT_CODE_SHARED = '''
        import os
        import torch
        import numpy as np
        from vllm.logger import init_logger

        self._sedac_patch_begin = {version}
        self._sedac_logger = init_logger("vllm.sedac")
        
        _sedac_enabled_env = os.environ.get("SEDAC_ENABLED", "0")
        self.sedac_enabled = _sedac_enabled_env.lower() in ("1", "true", "yes")
        
        _probe_layers_str = os.environ.get("SEDAC_PROBE_LAYERS", "7,14,21")
        self.sedac_probe_layers = tuple(int(x.strip()) for x in _probe_layers_str.split(",") if x.strip())
        
        self.sedac_confidence_decay = float(os.environ.get("SEDAC_CONFIDENCE_DECAY", "0.9"))
        self.sedac_soft_exit = os.environ.get("SEDAC_SOFT_EXIT", "{soft_exit}").lower() in ("1", "true", "yes")
        
        _layer_weights_str = os.environ.get("SEDAC_LAYER_WEIGHTS", "0.3,0.4,0.3")
        _layer_weights = [float(x.strip()) for x in _layer_weights_str.split(",") if x.strip()]
        while len(_layer_weights) < len(self.sedac_probe_layers):
            _layer_weights.append(1.0 / len(self.sedac_probe_layers))
        self.sedac_layer_weights = dict(zip(self.sedac_probe_layers, _layer_weights))
        
        self.sedac_adaptive = os.environ.get("SEDAC_ADAPTIVE", "1").lower() in ("1", "true", "yes")
        self.sedac_alpha = float(os.environ.get("SEDAC_ALPHA", "0.1"))
        self.sedac_calibrated = False
        self.sedac_calibration_steps = int(os.environ.get("SEDAC_CALIBRATION_STEPS", "50"))
        self.sedac_risk_history = {{l: [] for l in self.sedac_probe_layers}}
        
        _exit_rates_str = os.environ.get("SEDAC_EXIT_RATES", "0.2,0.5,0.8")
        _exit_rates_list = [float(x.strip()) for x in _exit_rates_str.split(",") if x.strip()]
        while len(_exit_rates_list) < len(self.sedac_probe_layers):
            _exit_rates_list.append(0.8)
        self.sedac_target_exit_rates = dict(zip(self.sedac_probe_layers, _exit_rates_list))
        
        _thresholds_str = os.environ.get("SEDAC_THRESHOLDS", "0.8,1.0,1.2")
        _thresholds_list = [float(x.strip()) for x in _thresholds_str.split(",") if x.strip()]
        while len(_thresholds_list) < len(self.sedac_probe_layers):
            _thresholds_list.append(_thresholds_list[-1] if _thresholds_list else 1.0)
        self.sedac_thresholds = dict(zip(self.sedac_probe_layers, _thresholds_list))
        
        self.__dict__["sedac_probes"] = {{}}
        self.sedac_threshold_tensors = {{}}
        self.sedac_log_every = int(os.environ.get("SEDAC_LOG_EVERY", "50"))
        self.sedac_calls = 0
        self.sedac_exit_calls = {{}}
        
        self._sedac_accumulated_confidence = None
        self._sedac_exited_mask = None
        self._sedac_soft_exit_ratios = None
        
        _probe_dir = os.environ.get("SEDAC_PROBE_DIR", "")
        if not _probe_dir:
            for _try_dir in ["/mnt/g/SEDACV5.0 FAST/sedac_data", "G:/SEDACV5.0 FAST/sedac_data", "./sedac_data"]:
                if os.path.isdir(_try_dir):
                    _probe_dir = _try_dir
                    break
            else:
                _probe_dir = "./sedac_data"
        
        self._sedac_logger.warning(
            "SEDAC patch v{version} enabled=%s layers=%s soft_exit=%s",
            self.sedac_enabled, self.sedac_probe_layers, self.sedac_soft_exit,
        )

        if self.sedac_enabled:
            try:
                _dev = "cuda" if torch.cuda.is_available() else "cpu"
                _dtype = config.dtype if hasattr(config, "dtype") else torch.float16
                probe_rank = int(os.environ.get("SEDAC_PROBE_RANK", "64"))
                pp_group = get_pp_group()
                _pp_ws = getattr(pp_group, "world_size", 1)
                pp_world_size = int(_pp_ws() if callable(_pp_ws) else _pp_ws)
                
                if pp_world_size != 1:
                    self.sedac_enabled = False
                    self._sedac_logger.warning("SEDAC disabled: pp_world_size=%d", pp_world_size)
                else:
                    _loaded_any = False
                    for _layer_idx in self.sedac_probe_layers:
                        _probe_path = os.path.join(_probe_dir, f"sedac_probe_layer{{_layer_idx}}.pth")
                        if not os.path.exists(_probe_path):
                            continue
                        
                        _probe = LREProbe(config.hidden_size, rank=probe_rank)
                        _probe.load_state_dict(torch.load(_probe_path, map_location="cpu"), strict=True)
                        _probe.to(device=_dev, dtype=_dtype).eval()
                        for _p in _probe.parameters():
                            _p.requires_grad_(False)
                        
                        self.__dict__["sedac_probes"][_layer_idx] = _probe
                        self.sedac_threshold_tensors[_layer_idx] = torch.tensor(
                            self.sedac_thresholds[_layer_idx], device=_dev, dtype=_dtype
                        )
                        self.sedac_exit_calls[_layer_idx] = 0
                        _loaded_any = True
                    
                    if not _loaded_any:
                        self.sedac_enabled = False
                    else:
                        self._sedac_logger.warning("SEDAC ready: %d probes", len(self.__dict__["sedac_probes"]))
            except Exception:
                self.sedac_enabled = False
                self._sedac_logger.exception("SEDAC probe load failed")
        self._sedac_patch_end = {version}
'''

FORWARD_CODE_V61 = '''
            _sedac_patch_forward_begin = 61
            if self.sedac_enabled:
                _probes = self.__dict__.get("sedac_probes", {})
                _abs_layer = idx + self.start_layer
                _batch_size = hidden_states.shape[0]
                
                if idx == 0:
                    self._sedac_accumulated_confidence = torch.zeros(_batch_size, device=hidden_states.device, dtype=hidden_states.dtype)
                    self._sedac_exited_mask = torch.zeros(_batch_size, device=hidden_states.device, dtype=torch.bool)
                    self._sedac_soft_exit_ratios = torch.zeros(_batch_size, device=hidden_states.device, dtype=hidden_states.dtype)
                    for _layer in self.layers:
                        _layer._sedac_skip_mlp = False
                        _layer._sedac_soft_ratio = None
                
                if _abs_layer in _probes:
                    _probe = _probes[_abs_layer]
                    
                    with torch.inference_mode():
                        current_h = hidden_states + residual if residual is not None else hidden_states
                        _risk = _probe(current_h).squeeze(-1)
                        
                        if _risk.dim() > 1:
                            _risk = _risk[:, -1]
                        
                        self.sedac_calls += 1
                        
                        _threshold = self.sedac_threshold_tensors[_abs_layer]
                        _layer_confidence = ((_threshold - _risk) / (_threshold + 1e-6)).clamp(0, 1)
                        
                        _weight = self.sedac_layer_weights[_abs_layer]
                        self._sedac_accumulated_confidence = (
                            self._sedac_accumulated_confidence * self.sedac_confidence_decay
                            + _layer_confidence * _weight
                        )
                        
                        _target_rate = self.sedac_target_exit_rates[_abs_layer]
                        _can_exit = (self._sedac_accumulated_confidence >= _target_rate) & (~self._sedac_exited_mask)
                        
                        if _can_exit.any():
                            self._sedac_exited_mask = self._sedac_exited_mask | _can_exit
                            _exit_count = _can_exit.sum().item()
                            self.sedac_exit_calls[_abs_layer] = self.sedac_exit_calls.get(_abs_layer, 0) + _exit_count
                            
                            if self.sedac_soft_exit:
                                _conf_for_exit = self._sedac_accumulated_confidence[_can_exit]
                                _soft_ratio = torch.tanh(_conf_for_exit * 2.0 - 1.0) * 0.5 + 0.5
                                self._sedac_soft_exit_ratios[_can_exit] = _soft_ratio
                            else:
                                self._sedac_soft_exit_ratios[_can_exit] = 1.0
                            
                            for _li, _layer in enumerate(self.layers):
                                if _li > idx:
                                    _layer._sedac_skip_mlp = True
                                    _layer._sedac_soft_ratio = self._sedac_soft_exit_ratios.clone()
                                    _layer._sedac_exited_mask = self._sedac_exited_mask.clone()
                        
                        if self.sedac_adaptive:
                            if not self.sedac_calibrated:
                                self.sedac_risk_history[_abs_layer].extend(_risk.cpu().tolist())
                                _all_ready = all(
                                    len(self.sedac_risk_history.get(l, [])) >= self.sedac_calibration_steps
                                    for l in self.sedac_probe_layers
                                )
                                if _all_ready:
                                    for _l in self.sedac_probe_layers:
                                        _hist = sorted(self.sedac_risk_history[_l])
                                        _target = self.sedac_target_exit_rates[_l]
                                        _q_idx = int(len(_hist) * _target)
                                        _q_idx = min(max(0, _q_idx), len(_hist) - 1)
                                        _new_thr = _hist[_q_idx]
                                        self.sedac_thresholds[_l] = _new_thr
                                        self.sedac_threshold_tensors[_l].fill_(_new_thr)
                                    self.sedac_calibrated = True
                            else:
                                _batch_risks = _risk.cpu().numpy()
                                _sorted = np.sort(_batch_risks)
                                _target = self.sedac_target_exit_rates[_abs_layer]
                                _k = int(len(_sorted) * _target)
                                _k = min(max(0, _k), len(_sorted) - 1)
                                _batch_thr = _sorted[_k]
                                _old_thr = self.sedac_thresholds[_abs_layer]
                                _new_thr = self.sedac_alpha * _batch_thr + (1 - self.sedac_alpha) * _old_thr
                                self.sedac_thresholds[_abs_layer] = _new_thr
                                self.sedac_threshold_tensors[_abs_layer].fill_(_new_thr)
            _sedac_patch_forward_end = 61
'''

FORWARD_CODE_V60 = '''
            _sedac_patch_forward_begin = 60
            if self.sedac_enabled:
                _probes = self.__dict__.get("sedac_probes", {})
                _abs_layer = idx + self.start_layer
                
                if _abs_layer in _probes:
                    _probe = _probes[_abs_layer]
                    
                    with torch.inference_mode():
                        current_h = hidden_states + residual if residual is not None else hidden_states
                        _risk = _probe(current_h).squeeze(-1)
                        
                        if _risk.dim() > 1:
                            _max_risk = _risk[:, -1].max()
                        else:
                            _max_risk = _risk.max()
                        
                        self.sedac_calls += 1
                        _threshold = self.sedac_threshold_tensors[_abs_layer]
                        _can_exit = _max_risk < _threshold
                        
                        if _can_exit:
                            self.sedac_exit_calls[_abs_layer] = self.sedac_exit_calls.get(_abs_layer, 0) + 1
                            for _li, _layer in enumerate(self.layers):
                                if _li > idx:
                                    _layer._sedac_skip_mlp = True
            _sedac_patch_forward_end = 60
'''

DECODER_PATCH_V61 = '''
# --- SEDAC DECODER PATCH v6.1 ---
_sedac_original_decoder_forward = Qwen2DecoderLayer.forward

def _sedac_decoder_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _skip_mlp = getattr(self, "_sedac_skip_mlp", False)
    _soft_ratio = getattr(self, "_sedac_soft_ratio", None)
    _exited_mask = getattr(self, "_sedac_exited_mask", None)
    
    if _skip_mlp and _soft_ratio is not None and _exited_mask is not None:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        _batch_size = hidden_states.shape[0]
        if _batch_size == _soft_ratio.shape[0]:
            mlp_out = self.mlp(hidden_states)
            _mask = _exited_mask.view(-1, 1, 1).float()
            _ratio = _soft_ratio.view(-1, 1, 1)
            hidden_states = mlp_out * (1.0 - _mask * _ratio)
        else:
            hidden_states = hidden_states.new_zeros(hidden_states.shape)
        
        return hidden_states, residual
    
    elif _skip_mlp:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = hidden_states.new_zeros(hidden_states.shape)
        return hidden_states, residual
    
    return _sedac_original_decoder_forward(self, positions, hidden_states, residual)

Qwen2DecoderLayer.forward = _sedac_decoder_forward
# ---------------------------
'''

DECODER_PATCH_V60 = '''
# --- SEDAC DECODER PATCH v6.0 ---
_sedac_original_decoder_forward = Qwen2DecoderLayer.forward

def _sedac_decoder_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _skip_mlp = getattr(self, "_sedac_skip_mlp", False)
    
    if _skip_mlp:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = hidden_states.new_zeros(hidden_states.shape)
        return hidden_states, residual
    
    return _sedac_original_decoder_forward(self, positions, hidden_states, residual)

Qwen2DecoderLayer.forward = _sedac_decoder_forward
# ---------------------------
'''


class VLLMPatcher:
    """
    Unified vLLM patcher for SEDAC integration.
    
    Supports V6.0 (hard exit) and V6.1 (soft exit with confidence accumulation).
    """
    
    def __init__(self, config: PatchConfig):
        self.config = config
        self.target_path = self._resolve_target_path()
    
    def _resolve_target_path(self) -> Path:
        """Resolve vLLM Qwen2 model source path."""
        if self.config.target_path:
            return Path(self.config.target_path)
        
        env_target = os.environ.get("SEDAC_VLLM_QWEN2_PATH")
        if env_target:
            return Path(env_target)
        
        try:
            import vllm.model_executor.models.qwen2 as qwen2
            return Path(qwen2.__file__)
        except ImportError:
            raise RuntimeError("Cannot find vLLM Qwen2 model. Install vLLM or set SEDAC_VLLM_QWEN2_PATH")
    
    def _strip_existing_patches(self, lines: List[str]) -> List[str]:
        """Remove existing SEDAC patches from source."""
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if stripped.startswith("SEDAC_PATCH_VERSION ="):
                i += 1
                continue
            
            if stripped.startswith("self._sedac_patch_begin ="):
                i += 1
                while i < len(lines):
                    if lines[i].strip().startswith("self._sedac_patch_end ="):
                        i += 1
                        break
                    i += 1
                continue
            
            if stripped.startswith("_sedac_patch_forward_begin ="):
                i += 1
                while i < len(lines):
                    if lines[i].strip().startswith("_sedac_patch_forward_end ="):
                        i += 1
                        break
                    i += 1
                continue
            
            if "SEDAC DECODER PATCH" in stripped:
                i += 1
                while i < len(lines):
                    if lines[i].strip() == "# ---------------------------":
                        i += 1
                        break
                    i += 1
                continue
            
            if stripped.startswith("_sedac_original_decoder_forward ="):
                i += 1
                continue
            
            if stripped.startswith("Qwen2DecoderLayer.forward = _sedac_decoder_forward"):
                i += 1
                continue
            
            out.append(line)
            i += 1
        
        return out
    
    def _generate_patch_code(self) -> Tuple[str, str, str, str]:
        """Generate patch code for the configured version."""
        version_num = 61 if self.config.version == PatchVersion.V61 else 60
        soft_exit_str = "1" if self.config.soft_exit else "0"
        
        probe_def = PROBE_DEFINITION.format(version=version_num)
        init_code = INIT_CODE_SHARED.format(
            version=version_num,
            soft_exit=soft_exit_str,
        )
        
        if self.config.version == PatchVersion.V61:
            forward_code = FORWARD_CODE_V61
            decoder_patch = DECODER_PATCH_V61
        else:
            forward_code = FORWARD_CODE_V60
            decoder_patch = DECODER_PATCH_V60
        
        return probe_def, init_code, forward_code, decoder_patch
    
    def apply(self, dry_run: bool = False) -> bool:
        """
        Apply SEDAC patch to vLLM source.
        
        Args:
            dry_run: If True, print patch without applying
        
        Returns:
            True if successful
        """
        if not self.target_path.exists():
            raise FileNotFoundError(f"Target file not found: {self.target_path}")
        
        with open(self.target_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        lines = self._strip_existing_patches(lines)
        probe_def, init_code, forward_code, decoder_patch = self._generate_patch_code()
        
        new_lines: List[str] = []
        inserted_probe = False
        inserted_init = False
        inserted_forward = False
        
        for line in lines:
            new_lines.append(line)
            
            if not inserted_probe and "class Qwen2MLP(nn.Module):" in line:
                new_lines.pop()
                new_lines.append(probe_def)
                new_lines.append(line)
                inserted_probe = True
            
            if not inserted_init and "self.aux_hidden_state_layers = tuple[int, ...]()" in line:
                new_lines.append(init_code)
                inserted_init = True
            
            if not inserted_forward and "hidden_states, residual = layer(positions, hidden_states, residual)" in line:
                new_lines.append(forward_code)
                inserted_forward = True
        
        new_lines.append(decoder_patch)
        
        if not (inserted_probe and inserted_init and inserted_forward):
            print(f"Error: Could not find all insertion points")
            print(f"  Probe: {inserted_probe}, Init: {inserted_init}, Forward: {inserted_forward}")
            return False
        
        if dry_run:
            print("--- DRY RUN: Patch preview ---")
            print("".join(new_lines[-80:]))
            print("--- End ---")
            return True
        
        with open(self.target_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        
        print(f"✅ SEDAC {self.config.version.value} patch applied to {self.target_path}")
        return True
    
    def remove(self) -> bool:
        """Remove SEDAC patch from vLLM source."""
        if not self.target_path.exists():
            return False
        
        with open(self.target_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        clean_lines = self._strip_existing_patches(lines)
        
        with open(self.target_path, "w", encoding="utf-8") as f:
            f.writelines(clean_lines)
        
        print(f"✅ SEDAC patch removed from {self.target_path}")
        return True


def apply_sedac_patch(
    version: str = "6.1",
    target_path: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """
    Convenience function to apply SEDAC patch.
    
    Args:
        version: "6.0" or "6.1"
        target_path: Path to vLLM qwen2.py (auto-detected if None)
        dry_run: Preview without applying
    
    Returns:
        True if successful
    """
    config = PatchConfig(
        version=PatchVersion.V61 if "6.1" in version else PatchVersion.V60,
        target_path=target_path,
    )
    patcher = VLLMPatcher(config)
    return patcher.apply(dry_run=dry_run)


def remove_sedac_patch(target_path: Optional[str] = None) -> bool:
    """
    Remove SEDAC patch from vLLM.
    
    Args:
        target_path: Path to vLLM qwen2.py (auto-detected if None)
    
    Returns:
        True if successful
    """
    config = PatchConfig(target_path=target_path)
    patcher = VLLMPatcher(config)
    return patcher.remove()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEDAC vLLM Patcher")
    parser.add_argument("--version", type=str, default="6.1", choices=["6.0", "6.1"])
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    
    if args.remove:
        remove_sedac_patch(args.target)
    else:
        apply_sedac_patch(args.version, args.target, args.dry_run)
