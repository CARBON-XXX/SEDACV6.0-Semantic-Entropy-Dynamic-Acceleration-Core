"""
SEDAC Cascade Controller
========================

Main orchestrator for multi-layer cascade early exit decisions.

Features:
    - Confidence accumulation across layers (Bayesian-inspired)
    - Configurable exit strategies (Hard/Soft/Adaptive)
    - Adaptive threshold calibration with EMA smoothing
    - Per-layer metrics tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import threading

import numpy as np
import torch

from sedac.core.exit_strategy import (
    ExitDecision,
    ExitStrategy,
    ExitType,
    HardExit,
    SoftExit,
    create_exit_strategy,
)


@dataclass
class LayerConfig:
    """Configuration for a single checkpoint layer."""
    layer_idx: int
    target_exit_rate: float = 0.5
    initial_threshold: float = 1.0
    confidence_weight: float = 0.33
    
    def __post_init__(self):
        if not 0 <= self.target_exit_rate <= 1:
            raise ValueError(f"target_exit_rate must be in [0, 1], got {self.target_exit_rate}")
        if self.initial_threshold <= 0:
            raise ValueError(f"initial_threshold must be positive, got {self.initial_threshold}")


@dataclass
class CascadeConfig:
    """Configuration for the cascade controller."""
    layer_configs: List[LayerConfig]
    confidence_decay: float = 0.9
    alpha: float = 0.1
    calibration_steps: int = 50
    exit_strategy: str = "soft"
    exit_strategy_params: dict = field(default_factory=dict)
    total_layers: int = 36
    
    @classmethod
    def from_env(cls) -> "CascadeConfig":
        """Create config from environment variables."""
        import os
        
        layers_str = os.environ.get("SEDAC_PROBE_LAYERS", "7,14,21")
        layers = [int(x.strip()) for x in layers_str.split(",") if x.strip()]
        
        thresholds_str = os.environ.get("SEDAC_THRESHOLDS", "0.8,1.0,1.2")
        thresholds = [float(x.strip()) for x in thresholds_str.split(",") if x.strip()]
        
        exit_rates_str = os.environ.get("SEDAC_EXIT_RATES", "0.2,0.5,0.8")
        exit_rates = [float(x.strip()) for x in exit_rates_str.split(",") if x.strip()]
        
        weights_str = os.environ.get("SEDAC_LAYER_WEIGHTS", "0.3,0.4,0.3")
        weights = [float(x.strip()) for x in weights_str.split(",") if x.strip()]
        
        while len(thresholds) < len(layers):
            thresholds.append(thresholds[-1] if thresholds else 1.0)
        while len(exit_rates) < len(layers):
            exit_rates.append(exit_rates[-1] if exit_rates else 0.5)
        while len(weights) < len(layers):
            weights.append(1.0 / len(layers))
        
        layer_configs = [
            LayerConfig(
                layer_idx=layer,
                target_exit_rate=rate,
                initial_threshold=thr,
                confidence_weight=weight,
            )
            for layer, rate, thr, weight in zip(layers, exit_rates, thresholds, weights)
        ]
        
        strategy = "soft" if os.environ.get("SEDAC_SOFT_EXIT", "1").lower() in ("1", "true") else "hard"
        
        return cls(
            layer_configs=layer_configs,
            confidence_decay=float(os.environ.get("SEDAC_CONFIDENCE_DECAY", "0.9")),
            alpha=float(os.environ.get("SEDAC_ALPHA", "0.1")),
            calibration_steps=int(os.environ.get("SEDAC_CALIBRATION_STEPS", "50")),
            exit_strategy=strategy,
            total_layers=int(os.environ.get("SEDAC_TOTAL_LAYERS", "36")),
        )


class CascadeController:
    """
    Main controller for cascade early exit decisions.
    
    Implements confidence accumulation across multiple checkpoint layers
    with configurable exit strategies.
    
    Example:
        ```python
        config = CascadeConfig(
            layer_configs=[
                LayerConfig(layer_idx=7, target_exit_rate=0.2, initial_threshold=0.8),
                LayerConfig(layer_idx=14, target_exit_rate=0.5, initial_threshold=1.0),
                LayerConfig(layer_idx=21, target_exit_rate=0.8, initial_threshold=1.2),
            ],
            confidence_decay=0.9,
            exit_strategy="soft",
        )
        controller = CascadeController(config)
        
        # During inference
        for layer_idx in [7, 14, 21]:
            decisions = controller.evaluate(layer_idx, risk_scores, batch_size)
        ```
    """
    
    def __init__(self, config: CascadeConfig):
        self.config = config
        self.layer_configs = {cfg.layer_idx: cfg for cfg in config.layer_configs}
        self.layer_indices = tuple(sorted(self.layer_configs.keys()))
        
        self.thresholds: Dict[int, float] = {
            cfg.layer_idx: cfg.initial_threshold
            for cfg in config.layer_configs
        }
        self.threshold_tensors: Dict[int, torch.Tensor] = {}
        
        self.exit_strategy = create_exit_strategy(
            config.exit_strategy,
            **config.exit_strategy_params,
        )
        
        self._lock = threading.RLock()
        self._calibrated = False
        self._calibration_samples: Dict[int, List[float]] = {
            layer: [] for layer in self.layer_indices
        }
        
        self._accumulated_confidence: Optional[torch.Tensor] = None
        self._exited_mask: Optional[torch.Tensor] = None
        self._soft_exit_ratios: Optional[torch.Tensor] = None
        self._current_batch_size: int = 0
        
        self._total_calls = 0
        self._exit_counts: Dict[int, int] = {layer: 0 for layer in self.layer_indices}
    
    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """Reset per-forward state for a new batch."""
        self._accumulated_confidence = torch.zeros(batch_size, device=device, dtype=dtype)
        self._exited_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        self._soft_exit_ratios = torch.zeros(batch_size, device=device, dtype=dtype)
        self._current_batch_size = batch_size
        
        for layer_idx in self.layer_indices:
            if layer_idx not in self.threshold_tensors:
                self.threshold_tensors[layer_idx] = torch.tensor(
                    self.thresholds[layer_idx], device=device, dtype=dtype
                )
    
    def evaluate(
        self,
        layer_idx: int,
        risk_scores: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate exit decision at a checkpoint layer.
        
        Args:
            layer_idx: Current layer index
            risk_scores: Risk scores from probe [batch] or [batch, seq]
            hidden_states: Optional hidden states for caching
        
        Returns:
            Tuple of:
                - exited_mask: Boolean mask of tokens that should exit [batch]
                - soft_exit_ratios: MLP skip ratios for each token [batch]
                - accumulated_confidence: Current confidence scores [batch]
        """
        if layer_idx not in self.layer_configs:
            return self._exited_mask, self._soft_exit_ratios, self._accumulated_confidence
        
        cfg = self.layer_configs[layer_idx]
        
        if risk_scores.dim() > 1:
            risk_scores = risk_scores[:, -1]
        
        threshold = self.threshold_tensors.get(layer_idx)
        if threshold is None:
            threshold = torch.tensor(
                self.thresholds[layer_idx],
                device=risk_scores.device,
                dtype=risk_scores.dtype,
            )
            self.threshold_tensors[layer_idx] = threshold
        
        layer_confidence = ((threshold - risk_scores) / (threshold + 1e-6)).clamp(0, 1)
        
        self._accumulated_confidence = (
            self._accumulated_confidence * self.config.confidence_decay
            + layer_confidence * cfg.confidence_weight
        )
        
        can_exit = (self._accumulated_confidence >= cfg.target_exit_rate) & (~self._exited_mask)
        
        if can_exit.any():
            self._exited_mask = self._exited_mask | can_exit
            exit_count = can_exit.sum().item()
            self._exit_counts[layer_idx] = self._exit_counts.get(layer_idx, 0) + exit_count
            
            if self.exit_strategy.exit_type != ExitType.HARD:
                for i in range(can_exit.shape[0]):
                    if can_exit[i]:
                        ratio = self.exit_strategy.compute_exit_ratio(
                            self._accumulated_confidence[i].item(),
                            cfg.target_exit_rate,
                            layer_idx,
                            self.config.total_layers,
                        )
                        self._soft_exit_ratios[i] = ratio
            else:
                self._soft_exit_ratios[can_exit] = 1.0
        
        self._total_calls += 1
        
        self._update_threshold_online(layer_idx, risk_scores)
        
        return self._exited_mask.clone(), self._soft_exit_ratios.clone(), self._accumulated_confidence.clone()
    
    def _update_threshold_online(
        self,
        layer_idx: int,
        risk_scores: torch.Tensor,
    ) -> None:
        """Update threshold using online EMA calibration."""
        with self._lock:
            if not self._calibrated:
                self._calibration_samples[layer_idx].extend(
                    risk_scores.detach().cpu().tolist()
                )
                
                all_ready = all(
                    len(samples) >= self.config.calibration_steps
                    for samples in self._calibration_samples.values()
                )
                
                if all_ready:
                    for layer, samples in self._calibration_samples.items():
                        cfg = self.layer_configs[layer]
                        sorted_samples = sorted(samples)
                        q_idx = int(len(sorted_samples) * cfg.target_exit_rate)
                        q_idx = min(max(0, q_idx), len(sorted_samples) - 1)
                        new_threshold = sorted_samples[q_idx]
                        
                        self.thresholds[layer] = new_threshold
                        if layer in self.threshold_tensors:
                            self.threshold_tensors[layer].fill_(new_threshold)
                    
                    self._calibrated = True
            else:
                cfg = self.layer_configs[layer_idx]
                sorted_risks = risk_scores.detach().cpu().numpy()
                sorted_risks.sort()
                
                k = int(len(sorted_risks) * cfg.target_exit_rate)
                k = min(max(0, k), len(sorted_risks) - 1)
                batch_threshold = sorted_risks[k]
                
                old_threshold = self.thresholds[layer_idx]
                new_threshold = (
                    self.config.alpha * batch_threshold
                    + (1 - self.config.alpha) * old_threshold
                )
                
                self.thresholds[layer_idx] = new_threshold
                if layer_idx in self.threshold_tensors:
                    self.threshold_tensors[layer_idx].fill_(new_threshold)
    
    def get_stats(self) -> Dict[str, Union[int, Dict[int, int], Dict[int, float]]]:
        """Get controller statistics."""
        return {
            "total_calls": self._total_calls,
            "exit_counts": dict(self._exit_counts),
            "thresholds": dict(self.thresholds),
            "calibrated": self._calibrated,
        }
    
    def reset_calibration(self) -> None:
        """Reset calibration state."""
        with self._lock:
            self._calibrated = False
            self._calibration_samples = {layer: [] for layer in self.layer_indices}
            for layer_idx, cfg in self.layer_configs.items():
                self.thresholds[layer_idx] = cfg.initial_threshold
                if layer_idx in self.threshold_tensors:
                    self.threshold_tensors[layer_idx].fill_(cfg.initial_threshold)
    
    def is_calibrated(self) -> bool:
        """Check if calibration is complete."""
        return self._calibrated
    
    def get_threshold(self, layer_idx: int) -> float:
        """Get current threshold for a layer."""
        return self.thresholds.get(layer_idx, 1.0)
    
    def set_threshold(self, layer_idx: int, value: float) -> None:
        """Manually set threshold for a layer."""
        with self._lock:
            self.thresholds[layer_idx] = value
            if layer_idx in self.threshold_tensors:
                self.threshold_tensors[layer_idx].fill_(value)
    
    @property
    def exited_mask(self) -> Optional[torch.Tensor]:
        """Current exit mask."""
        return self._exited_mask
    
    @property
    def soft_exit_ratios(self) -> Optional[torch.Tensor]:
        """Current soft exit ratios."""
        return self._soft_exit_ratios
    
    @property
    def accumulated_confidence(self) -> Optional[torch.Tensor]:
        """Current accumulated confidence."""
        return self._accumulated_confidence
