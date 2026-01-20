"""
SEDAC V6.0 - Semantic Entropy Dynamic Acceleration Core
========================================================

Multi-layer Cascade Early Exit for vLLM (Qwen2 Architecture)

This module patches vLLM's Qwen2 model to enable dynamic early exit
based on semantic entropy prediction at multiple checkpoint layers.

Architecture:
    - LREProbe: Low-Rank Entropy probe for risk prediction
    - Cascade Exit: Token-level early exit across layers 7, 14, 21
    - Adaptive Threshold: Runtime calibration based on target exit rates

Environment Variables:
    SEDAC_ENABLED         : Enable/disable SEDAC (0/1)
    SEDAC_PROBE_LAYERS    : Comma-separated checkpoint layers (default: 7,14,21)
    SEDAC_THRESHOLDS      : Initial thresholds per layer (default: 0.8,1.3,1.7)
    SEDAC_ADAPTIVE        : Enable adaptive threshold calibration (0/1)
    SEDAC_EXIT_RATES      : Target exit rates for calibration (default: 0.2,0.6,0.9)
    SEDAC_PROBE_DIR       : Directory containing trained probe weights
    SEDAC_PROBE_RANK      : Probe hidden dimension (default: 64)

Reference:
    SEDAC: Semantic Entropy Dynamic Acceleration Core
    https://github.com/CARBON-XXX/Semantic-Entropy-Dynamic-Acceleration-Core-SEDAC

License: MIT
"""

import argparse
import os
import sys


def _resolve_target_path(arg_target_path: str | None) -> str:
    """Resolve vLLM Qwen2 model source path for patching."""
    if arg_target_path:
        return arg_target_path
    env_target = os.environ.get("SEDAC_VLLM_QWEN2_PATH")
    if env_target:
        return env_target
    try:
        import vllm.model_executor.models.qwen2 as qwen2
        return str(qwen2.__file__)
    except Exception:
        return ""


# =============================================================================
# LREProbe Definition - Low-Rank Entropy Predictor
# =============================================================================
probe_def = """
import os
import torch
SEDAC_PATCH_VERSION = 6

class LREProbe(nn.Module):
    \"\"\"
    Low-Rank Entropy Probe for semantic uncertainty prediction.
    
    Architecture: Linear(d -> r) -> LayerNorm -> Linear(r -> 1) -> Softplus
    
    Args:
        input_dim: Model hidden dimension (e.g., 2048 for Qwen2.5-3B)
        rank: Probe hidden dimension (default: 64)
    
    Returns:
        Predicted risk score (non-negative, lower = more confident)
    \"\"\"
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
"""

init_code_v6 = """
        import os
        import torch
        from vllm.logger import init_logger

        self._sedac_patch_begin = 6
        self._sedac_logger = init_logger("vllm.sedac")
        self._sedac_calls_metric = None
        self._sedac_exits_metric = None
        try:
            from prometheus_client import Counter
            _cls = type(self)
            if not hasattr(_cls, "_sedac_calls_metric"):
                _cls._sedac_calls_metric = Counter("sedac_calls_total", "SEDAC decisions evaluated")
                _cls._sedac_exits_metric = Counter("sedac_exits_total", "SEDAC exits triggered")
            self._sedac_calls_metric = getattr(_cls, "_sedac_calls_metric", None)
            self._sedac_exits_metric = getattr(_cls, "_sedac_exits_metric", None)
        except Exception:
            pass
        
        _sedac_enabled_env = os.environ.get("SEDAC_ENABLED", "0")
        self.sedac_enabled = _sedac_enabled_env.lower() in ("1", "true", "yes")
        
        # V6: Multi-layer configuration
        _probe_layers_str = os.environ.get("SEDAC_PROBE_LAYERS", "7,14,21")
        self.sedac_probe_layers = tuple(int(x.strip()) for x in _probe_layers_str.split(",") if x.strip())
        
        # Adaptive threshold mode
        self.sedac_adaptive = os.environ.get("SEDAC_ADAPTIVE", "1").lower() in ("1", "true", "yes")
        self.sedac_calibrated = False
        self.sedac_calibration_steps = int(os.environ.get("SEDAC_CALIBRATION_STEPS", "50"))
        self.sedac_risk_history = {l: [] for l in self.sedac_probe_layers}
        
        # Target exit rates (adaptive mode)
        _exit_rates_str = os.environ.get("SEDAC_EXIT_RATES", "0.2,0.6,0.9")
        _exit_rates_list = [float(x.strip()) for x in _exit_rates_str.split(",") if x.strip()]
        while len(_exit_rates_list) < len(self.sedac_probe_layers):
            _exit_rates_list.append(0.9)
        self.sedac_target_exit_rates = dict(zip(self.sedac_probe_layers, _exit_rates_list))
        
        # Initial thresholds (static mode or pre-calibration)
        _thresholds_str = os.environ.get("SEDAC_THRESHOLDS", "0.8,1.3,1.7")
        _thresholds_list = [float(x.strip()) for x in _thresholds_str.split(",") if x.strip()]
        while len(_thresholds_list) < len(self.sedac_probe_layers):
            _thresholds_list.append(_thresholds_list[-1] if _thresholds_list else 1.0)
        self.sedac_thresholds = dict(zip(self.sedac_probe_layers, _thresholds_list))
        
        self.__dict__["sedac_probes"] = {}
        self.sedac_threshold_tensors = {}
        self.sedac_log_every = int(os.environ.get("SEDAC_LOG_EVERY", "50"))
        self.sedac_calls = 0
        self.sedac_exit_calls = {}
        self._sedac_exited = False
        
        # Support Windows and Linux paths
        _probe_dir = os.environ.get("SEDAC_PROBE_DIR", "")
        if not _probe_dir:
            for _try_dir in ["/mnt/g/SEDACV5.0 FAST/sedac_data", "G:/SEDACV5.0 FAST/sedac_data", "./sedac_data"]:
                if os.path.isdir(_try_dir):
                    _probe_dir = _try_dir
                    break
            else:
                _probe_dir = "./sedac_data"
        
        self._sedac_logger.warning(
            "SEDAC patch v6 (cascade) enabled=%s layers=%s thresholds=%s probe_dir=%s",
            self.sedac_enabled, self.sedac_probe_layers, self.sedac_thresholds, _probe_dir,
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
                        _probe_path = os.path.join(_probe_dir, f"sedac_probe_layer{_layer_idx}.pth")
                        if not os.path.exists(_probe_path):
                            self._sedac_logger.warning("SEDAC probe missing for layer %d: %s", _layer_idx, _probe_path)
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
                        self._sedac_logger.warning("SEDAC probe loaded: layer=%d threshold=%.4f", _layer_idx, self.sedac_thresholds[_layer_idx])
                        _loaded_any = True
                    
                    if not _loaded_any:
                        self.sedac_enabled = False
                        self._sedac_logger.warning("SEDAC disabled: no probes loaded")
                    else:
                        self._sedac_logger.warning("SEDAC ready: %d probes loaded", len(self.__dict__["sedac_probes"]))
            except Exception:
                self.sedac_enabled = False
                self._sedac_logger.exception("SEDAC probe load failed")
        self._sedac_patch_end = 6
"""

forward_code_v6 = """
            _sedac_patch_forward_begin = 6
            if self.sedac_enabled:
                _probes = self.__dict__.get("sedac_probes", {})
                _abs_layer = idx + self.start_layer
                
                # Reset state at first layer
                if idx == 0:
                    self._sedac_exited = False
                    for _layer in self.layers:
                        _layer._sedac_skip_mlp = False
                
                # If already exited, skip subsequent checks
                if not self._sedac_exited and _abs_layer in _probes:
                    _probe = _probes[_abs_layer]
                    
                    with torch.inference_mode():
                        current_h = hidden_states + residual if residual is not None else hidden_states
                        _risk = _probe(current_h)
                        _max_risk = _risk.max()
                        
                        self.sedac_calls += 1
                        if self._sedac_calls_metric is not None:
                            try:
                                self._sedac_calls_metric.inc()
                            except Exception:
                                pass
                        
                        # Adaptive threshold calibration
                        if self.sedac_adaptive and not self.sedac_calibrated:
                            _risk_val = _max_risk.item()
                            self.sedac_risk_history[_abs_layer].append(_risk_val)
                            
                            # Check if all layers have collected enough samples
                            _all_ready = all(
                                len(self.sedac_risk_history.get(l, [])) >= self.sedac_calibration_steps
                                for l in self.sedac_probe_layers
                            )
                            if _all_ready:
                                # Calculate threshold based on target exit rate
                                for _l in self.sedac_probe_layers:
                                    _hist = sorted(self.sedac_risk_history[_l])
                                    _target_rate = self.sedac_target_exit_rates[_l]
                                    _q_idx = int(len(_hist) * _target_rate)
                                    _q_idx = min(max(0, _q_idx), len(_hist) - 1)
                                    _new_thr = _hist[_q_idx]
                                    self.sedac_thresholds[_l] = _new_thr
                                    self.sedac_threshold_tensors[_l].fill_(_new_thr)
                                self.sedac_calibrated = True
                                self._sedac_logger.warning("SEDAC calibration done: thresholds=%s", self.sedac_thresholds)
                            _can_exit = False  # Do not exit during calibration
                        else:
                            # Normal exit decision
                            _threshold = self.sedac_threshold_tensors[_abs_layer]
                            _can_exit = _max_risk < _threshold
                        
                        if _can_exit:
                            self._sedac_exited = True
                            self.sedac_exit_calls[_abs_layer] = self.sedac_exit_calls.get(_abs_layer, 0) + 1
                            if self._sedac_exits_metric is not None:
                                try:
                                    self._sedac_exits_metric.inc()
                                except Exception:
                                    pass
                            # Mark subsequent layers to skip MLP
                            for _li, _layer in enumerate(self.layers):
                                if _li > idx:
                                    _layer._sedac_skip_mlp = True
                        
                        if self.sedac_log_every > 0 and (self.sedac_calls % self.sedac_log_every) == 0:
                            self._sedac_logger.warning(
                                "SEDAC L%d risk=%.4f thr=%.4f exit=%s calls=%d exits=%s cal=%s",
                                _abs_layer, _max_risk.item(), self.sedac_thresholds[_abs_layer],
                                bool(_can_exit), self.sedac_calls, self.sedac_exit_calls, self.sedac_calibrated,
                            )
            _sedac_patch_forward_end = 6
"""

decoder_patch_v6 = """
# --- SEDAC DECODER PATCH v6 ---
_sedac_original_decoder_forward = Qwen2DecoderLayer.forward

def _sedac_decoder_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _skip_mlp = getattr(self, "_sedac_skip_mlp", False)
    
    if _skip_mlp:
        # SEDAC: Only run attention + layernorm, skip MLP
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # Zero out MLP output
        hidden_states = hidden_states.new_zeros(hidden_states.shape)
        return hidden_states, residual
    
    return _sedac_original_decoder_forward(self, positions, hidden_states, residual)

Qwen2DecoderLayer.forward = _sedac_decoder_forward
# ---------------------------
"""


def _strip_sedac_blocks(lines: list[str]) -> list[str]:
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

        if stripped in ("# --- SEDAC DECODER PATCH ---", "# --- SEDAC DECODER PATCH v2 ---", "# --- SEDAC DECODER PATCH v6 ---"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEDAC V6 vLLM Patcher")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print patch without applying")
    args = parser.parse_args()

    target_path = _resolve_target_path(args.target)
    print(f"SEDAC V6 Patcher")
    print(f"Target: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 清理旧 patch
    lines = _strip_sedac_blocks(lines)

    new_lines: list[str] = []
    inserted_probe = False
    inserted_init = False
    inserted_forward = False

    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Insert Probe Definition BEFORE Qwen2MLP class definition
        if not inserted_probe and "class Qwen2MLP(nn.Module):" in line:
             new_lines.pop()
             new_lines.append(probe_def)
             new_lines.append(line)
             inserted_probe = True
             
        # Insert Init Code
        if not inserted_init and "self.aux_hidden_state_layers = tuple[int, ...]()" in line:
            new_lines.append(init_code_v6)
            inserted_init = True
            
        # Insert Forward Code (AFTER the layer execution line)
        if not inserted_forward and "hidden_states, residual = layer(positions, hidden_states, residual)" in line:
            new_lines.append(forward_code_v6)
            inserted_forward = True

    # Append Decoder Patch at the end
    new_lines.append(decoder_patch_v6)

    # Safety check
    if not (inserted_probe and inserted_init and inserted_forward):
        print("Error: Could not find all insertion points.")
        print(f"  Probe: {inserted_probe}")
        print(f"  Init: {inserted_init}")
        print(f"  Forward: {inserted_forward}")
        sys.exit(1)

    if args.dry_run:
        print("\n--- DRY RUN: Patch content ---")
        print("".join(new_lines[-100:]))
        print("--- End ---")
    else:
        with open(target_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print("✅ Successfully patched with SEDAC V6!")
        print("\nUsage:")
        print("  export SEDAC_ENABLED=1")
        print("  export SEDAC_PROBE_LAYERS=7,14,21")
        print("  export SEDAC_THRESHOLDS=0.15,0.25,0.40")
        print("  export SEDAC_PROBE_DIR=/path/to/sedac_data")
