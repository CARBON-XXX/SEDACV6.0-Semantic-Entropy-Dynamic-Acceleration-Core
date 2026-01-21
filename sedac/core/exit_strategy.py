"""
SEDAC Exit Strategy Module
==========================

Defines different exit strategies for cascade early exit:
- HardExit: Binary skip (0% or 100% MLP computation)
- SoftExit: Gradual MLP reduction based on confidence
- AdaptiveSoftExit: Dynamic scaling based on layer position
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch


class ExitType(Enum):
    """Exit strategy types."""
    HARD = "hard"
    SOFT = "soft"
    ADAPTIVE_SOFT = "adaptive_soft"


@dataclass
class ExitDecision:
    """Result of cascade exit evaluation for a single token."""
    should_exit: bool
    exit_layer: int
    accumulated_confidence: float
    soft_exit_ratio: float
    exit_type: ExitType = ExitType.HARD
    
    def __post_init__(self):
        if self.exit_layer < 0:
            self.exit_layer = -1


class ExitStrategy(ABC):
    """Abstract base class for exit strategies."""
    
    @abstractmethod
    def compute_exit_ratio(
        self,
        accumulated_confidence: float,
        threshold: float,
        layer_idx: int,
        total_layers: int,
    ) -> float:
        """
        Compute the MLP skip ratio for a given confidence level.
        
        Args:
            accumulated_confidence: Accumulated confidence score
            threshold: Exit threshold for this layer
            layer_idx: Current layer index
            total_layers: Total number of layers in model
        
        Returns:
            Skip ratio in [0, 1] where 1 = full skip, 0 = no skip
        """
        pass
    
    @abstractmethod
    def should_exit(
        self,
        accumulated_confidence: float,
        threshold: float,
    ) -> bool:
        """Determine if exit condition is met."""
        pass
    
    @property
    @abstractmethod
    def exit_type(self) -> ExitType:
        """Return the exit type."""
        pass


class HardExit(ExitStrategy):
    """
    Hard exit strategy: Binary decision (0% or 100% MLP skip).
    
    Once confidence exceeds threshold, all subsequent MLPs are skipped entirely.
    This is the V6.0 behavior.
    """
    
    def compute_exit_ratio(
        self,
        accumulated_confidence: float,
        threshold: float,
        layer_idx: int,
        total_layers: int,
    ) -> float:
        if accumulated_confidence >= threshold:
            return 1.0
        return 0.0
    
    def should_exit(
        self,
        accumulated_confidence: float,
        threshold: float,
    ) -> bool:
        return accumulated_confidence >= threshold
    
    @property
    def exit_type(self) -> ExitType:
        return ExitType.HARD


class SoftExit(ExitStrategy):
    """
    Soft exit strategy: Gradual MLP reduction based on confidence.
    
    Uses a smooth S-curve (tanh) to map confidence to skip ratio:
        ratio = tanh(k * (confidence - 0.5)) * 0.5 + 0.5
    
    This avoids sharp quality cliffs at decision boundaries.
    """
    
    def __init__(
        self,
        steepness: float = 2.0,
        min_ratio: float = 0.0,
        max_ratio: float = 0.95,
    ):
        """
        Args:
            steepness: Controls how sharp the transition is (higher = sharper)
            min_ratio: Minimum skip ratio (even at low confidence)
            max_ratio: Maximum skip ratio (even at high confidence)
        """
        self.steepness = steepness
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def compute_exit_ratio(
        self,
        accumulated_confidence: float,
        threshold: float,
        layer_idx: int,
        total_layers: int,
    ) -> float:
        if accumulated_confidence < threshold * 0.5:
            return self.min_ratio
        
        confidence_ratio = accumulated_confidence / (threshold + 1e-6)
        raw_ratio = (
            torch.tanh(torch.tensor(self.steepness * (confidence_ratio - 0.5))).item()
            * 0.5 + 0.5
        )
        
        return min(max(raw_ratio, self.min_ratio), self.max_ratio)
    
    def should_exit(
        self,
        accumulated_confidence: float,
        threshold: float,
    ) -> bool:
        return accumulated_confidence >= threshold * 0.5
    
    @property
    def exit_type(self) -> ExitType:
        return ExitType.SOFT


class AdaptiveSoftExit(ExitStrategy):
    """
    Adaptive soft exit: Skip ratio depends on layer position.
    
    Earlier exits get lower skip ratios (more conservative),
    later exits can skip more aggressively.
    
    Formula:
        base_ratio = sigmoid(k * (confidence / threshold - 0.5))
        position_factor = layer_idx / total_layers
        final_ratio = base_ratio * (0.5 + 0.5 * position_factor)
    """
    
    def __init__(
        self,
        steepness: float = 4.0,
        position_weight: float = 0.5,
        min_ratio: float = 0.1,
        max_ratio: float = 0.9,
    ):
        """
        Args:
            steepness: Controls sigmoid steepness
            position_weight: How much layer position affects ratio
            min_ratio: Minimum skip ratio
            max_ratio: Maximum skip ratio
        """
        self.steepness = steepness
        self.position_weight = position_weight
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def compute_exit_ratio(
        self,
        accumulated_confidence: float,
        threshold: float,
        layer_idx: int,
        total_layers: int,
    ) -> float:
        confidence_ratio = accumulated_confidence / (threshold + 1e-6)
        base_ratio = torch.sigmoid(
            torch.tensor(self.steepness * (confidence_ratio - 0.5))
        ).item()
        
        position_factor = layer_idx / max(total_layers, 1)
        position_multiplier = 1.0 - self.position_weight + self.position_weight * position_factor
        
        final_ratio = base_ratio * position_multiplier
        
        return min(max(final_ratio, self.min_ratio), self.max_ratio)
    
    def should_exit(
        self,
        accumulated_confidence: float,
        threshold: float,
    ) -> bool:
        return accumulated_confidence >= threshold * 0.4
    
    @property
    def exit_type(self) -> ExitType:
        return ExitType.ADAPTIVE_SOFT


def create_exit_strategy(
    strategy_type: str = "soft",
    **kwargs,
) -> ExitStrategy:
    """
    Factory function to create exit strategy instances.
    
    Args:
        strategy_type: "hard", "soft", or "adaptive_soft"
        **kwargs: Strategy-specific parameters
    
    Returns:
        ExitStrategy instance
    """
    strategies = {
        "hard": HardExit,
        "soft": SoftExit,
        "adaptive_soft": AdaptiveSoftExit,
    }
    
    strategy_class = strategies.get(strategy_type.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    if strategy_type == "hard":
        return strategy_class()
    return strategy_class(**kwargs)


class MLPScaler:
    """
    Applies soft exit scaling to MLP outputs.
    
    Supports multiple scaling modes:
    - channel_mask: Zero out a fraction of channels
    - output_scale: Scale entire output by ratio
    - early_truncate: Only compute first N% of channels
    """
    
    def __init__(self, mode: str = "output_scale"):
        """
        Args:
            mode: "channel_mask", "output_scale", or "early_truncate"
        """
        self.mode = mode
        self._channel_mask_cache: dict = {}
    
    def apply(
        self,
        mlp_output: torch.Tensor,
        skip_ratio: float,
        exited_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply soft exit scaling to MLP output.
        
        Args:
            mlp_output: MLP output tensor [batch, seq, hidden]
            skip_ratio: Fraction to skip (0 = full, 1 = zero output)
            exited_mask: Boolean mask [batch] indicating which tokens exited
        
        Returns:
            Scaled MLP output
        """
        if skip_ratio <= 0:
            return mlp_output
        if skip_ratio >= 1:
            return torch.zeros_like(mlp_output)
        
        scale = 1.0 - skip_ratio
        
        if self.mode == "output_scale":
            if exited_mask is not None:
                mask = exited_mask.view(-1, 1, 1).float()
                return mlp_output * (1.0 - mask * skip_ratio)
            return mlp_output * scale
        
        elif self.mode == "channel_mask":
            hidden_dim = mlp_output.shape[-1]
            keep_channels = int(hidden_dim * scale)
            
            cache_key = (hidden_dim, keep_channels, mlp_output.device)
            if cache_key not in self._channel_mask_cache:
                mask = torch.zeros(hidden_dim, device=mlp_output.device)
                mask[:keep_channels] = 1.0
                self._channel_mask_cache[cache_key] = mask
            
            channel_mask = self._channel_mask_cache[cache_key]
            return mlp_output * channel_mask
        
        elif self.mode == "early_truncate":
            hidden_dim = mlp_output.shape[-1]
            keep_channels = int(hidden_dim * scale)
            
            result = torch.zeros_like(mlp_output)
            result[..., :keep_channels] = mlp_output[..., :keep_channels]
            return result
        
        return mlp_output * scale
    
    def clear_cache(self) -> None:
        """Clear the channel mask cache."""
        self._channel_mask_cache.clear()
