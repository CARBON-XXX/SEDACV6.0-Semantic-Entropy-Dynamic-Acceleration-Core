"""
SEDAC Probe Inference Module
============================

Low-Rank Entropy Probe (LREProbe) for semantic uncertainty prediction.

Architecture:
    Linear(d -> r) -> LayerNorm -> Linear(r -> 1) -> Softplus

Features:
    - JIT compilation for accelerated inference
    - Batch processing with CUDA graph support
    - LRU caching for calibration
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict


class LREProbe(nn.Module):
    """
    Low-Rank Entropy Probe for semantic uncertainty prediction.
    
    Args:
        input_dim: Model hidden dimension (e.g., 2048 for Qwen2.5-3B)
        rank: Probe hidden dimension (default: 64)
    
    Returns:
        Predicted risk score (non-negative, lower = more confident)
    """
    
    def __init__(self, input_dim: int, rank: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        
        self.proj = nn.Linear(input_dim, rank, bias=False)
        self.norm = nn.LayerNorm(rank)
        self.head = nn.Linear(rank, 1)
        self.act = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states [batch, seq_len, hidden_dim] or [batch, hidden_dim]
        
        Returns:
            Risk scores [batch, seq_len, 1] or [batch, 1]
        """
        h = self.proj(x)
        h = self.norm(h)
        return self.act(self.head(h))
    
    def get_confidence(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Compute confidence score (how much below threshold).
        
        Args:
            x: Hidden states
            threshold: Risk threshold for this layer
        
        Returns:
            Confidence scores in [0, 1]
        """
        risk = self.forward(x).squeeze(-1)
        confidence = ((threshold - risk) / (threshold + 1e-6)).clamp(0, 1)
        return confidence


@dataclass
class ProbeConfig:
    """Configuration for a single probe."""
    layer_idx: int
    rank: int = 64
    weight_path: Optional[str] = None
    jit_compile: bool = True


class ProbeManager:
    """
    Manages multiple LREProbe instances with optimizations.
    
    Features:
        - Lazy loading of probe weights
        - JIT compilation for accelerated inference
        - Batch inference across multiple layers
        - Feature caching for calibration
    """
    
    def __init__(
        self,
        hidden_dim: int,
        layer_indices: Tuple[int, ...] = (7, 14, 21),
        probe_dir: str = "sedac_data",
        rank: int = 64,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_jit: bool = True,
        cache_size: int = 10000,
    ):
        self.hidden_dim = hidden_dim
        self.layer_indices = layer_indices
        self.probe_dir = Path(probe_dir)
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.use_jit = use_jit
        self.cache_size = cache_size
        
        self.probes: Dict[int, nn.Module] = {}
        self.compiled_probes: Dict[int, nn.Module] = {}
        self._feature_cache: OrderedDict = OrderedDict()
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_inputs: Dict[int, torch.Tensor] = {}
        self._graph_outputs: Dict[int, torch.Tensor] = {}
    
    def load_probes(self) -> Dict[int, bool]:
        """
        Load all probe weights from disk.
        
        Returns:
            Dict mapping layer_idx to load success status
        """
        results = {}
        for layer_idx in self.layer_indices:
            probe_path = self.probe_dir / f"sedac_probe_layer{layer_idx}.pth"
            try:
                if not probe_path.exists():
                    results[layer_idx] = False
                    continue
                
                probe = LREProbe(self.hidden_dim, self.rank)
                state_dict = torch.load(probe_path, map_location="cpu")
                probe.load_state_dict(state_dict, strict=True)
                probe = probe.to(device=self.device, dtype=self.dtype)
                probe.eval()
                
                for p in probe.parameters():
                    p.requires_grad_(False)
                
                self.probes[layer_idx] = probe
                
                if self.use_jit:
                    try:
                        self.compiled_probes[layer_idx] = torch.jit.script(probe)
                    except Exception:
                        self.compiled_probes[layer_idx] = probe
                else:
                    self.compiled_probes[layer_idx] = probe
                
                results[layer_idx] = True
            except Exception as e:
                results[layer_idx] = False
        
        return results
    
    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run inference for a single layer's probe.
        
        Args:
            layer_idx: Layer index
            hidden_states: Hidden states [batch, seq_len, hidden_dim]
        
        Returns:
            Risk scores [batch, seq_len] or [batch]
        """
        probe = self.compiled_probes.get(layer_idx)
        if probe is None:
            raise ValueError(f"Probe for layer {layer_idx} not loaded")
        
        with torch.inference_mode():
            risk = probe(hidden_states).squeeze(-1)
        
        return risk
    
    def forward_batch(
        self,
        hidden_states_dict: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Batch inference for multiple layers' probes.
        
        Args:
            hidden_states_dict: Dict mapping layer_idx to hidden states
        
        Returns:
            Dict mapping layer_idx to risk scores
        """
        results = {}
        with torch.inference_mode():
            for layer_idx, hidden_states in hidden_states_dict.items():
                if layer_idx in self.compiled_probes:
                    results[layer_idx] = self.compiled_probes[layer_idx](
                        hidden_states
                    ).squeeze(-1)
        return results
    
    def cache_features(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        risk_scores: torch.Tensor,
    ) -> None:
        """Cache features for calibration."""
        key = (layer_idx, len(self._feature_cache))
        self._feature_cache[key] = (
            hidden_states.detach().cpu(),
            risk_scores.detach().cpu(),
        )
        
        while len(self._feature_cache) > self.cache_size:
            self._feature_cache.popitem(last=False)
    
    def get_cached_risks(self, layer_idx: int) -> torch.Tensor:
        """Get all cached risk scores for a layer."""
        risks = []
        for (idx, _), (_, risk) in self._feature_cache.items():
            if idx == layer_idx:
                risks.append(risk)
        if risks:
            return torch.cat(risks, dim=0)
        return torch.tensor([])
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._feature_cache.clear()
    
    def setup_cuda_graph(
        self,
        batch_size: int,
        seq_len: int = 1,
    ) -> bool:
        """
        Setup CUDA graph for accelerated multi-probe inference.
        
        Args:
            batch_size: Fixed batch size for graph capture
            seq_len: Fixed sequence length
        
        Returns:
            True if CUDA graph was successfully captured
        """
        if not torch.cuda.is_available():
            return False
        
        try:
            for layer_idx in self.layer_indices:
                if layer_idx not in self.probes:
                    continue
                self._graph_inputs[layer_idx] = torch.zeros(
                    batch_size, seq_len, self.hidden_dim,
                    device=self.device, dtype=self.dtype
                )
                self._graph_outputs[layer_idx] = torch.zeros(
                    batch_size, seq_len,
                    device=self.device, dtype=self.dtype
                )
            
            for _ in range(3):
                for layer_idx, inp in self._graph_inputs.items():
                    self._graph_outputs[layer_idx] = self.probes[layer_idx](inp).squeeze(-1)
            
            self._cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._cuda_graph):
                for layer_idx, inp in self._graph_inputs.items():
                    self._graph_outputs[layer_idx] = self.probes[layer_idx](inp).squeeze(-1)
            
            return True
        except Exception:
            self._cuda_graph = None
            return False
    
    def forward_with_graph(
        self,
        hidden_states_dict: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Run inference using captured CUDA graph.
        
        Falls back to regular forward if graph not available or shapes mismatch.
        """
        if self._cuda_graph is None:
            return self.forward_batch(hidden_states_dict)
        
        try:
            for layer_idx, hidden_states in hidden_states_dict.items():
                if layer_idx in self._graph_inputs:
                    self._graph_inputs[layer_idx].copy_(hidden_states)
            
            self._cuda_graph.replay()
            
            return {
                layer_idx: output.clone()
                for layer_idx, output in self._graph_outputs.items()
            }
        except Exception:
            return self.forward_batch(hidden_states_dict)
    
    def __contains__(self, layer_idx: int) -> bool:
        return layer_idx in self.probes
    
    def __getitem__(self, layer_idx: int) -> nn.Module:
        return self.probes[layer_idx]
    
    def __len__(self) -> int:
        return len(self.probes)
