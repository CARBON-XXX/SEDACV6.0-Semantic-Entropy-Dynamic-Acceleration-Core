"""
SEDAC Multi-Modal Confidence Fusion
====================================

Enhanced confidence accumulation using multiple signal sources:
- Entropy-based risk (primary)
- Perplexity estimation
- Attention entropy analysis
- Token consistency tracking

Supports:
- Weighted fusion with learned/configurable weights
- Adaptive decay based on historical patterns
- Gated fusion for dynamic modality selection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfidenceSignal:
    """A single confidence signal from one modality."""
    name: str
    value: float
    weight: float = 1.0
    uncertainty: float = 0.0


class ConfidenceEstimator(ABC):
    """Abstract base class for confidence estimators."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the modality name."""
        pass
    
    @abstractmethod
    def estimate(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Estimate confidence from hidden states.
        
        Args:
            hidden_states: [batch, seq, hidden] or [batch, hidden]
            layer_idx: Current layer index
        
        Returns:
            Confidence scores [batch] in [0, 1]
        """
        pass


class EntropyProbeEstimator(ConfidenceEstimator):
    """
    Primary estimator using trained LREProbe.
    
    Converts risk scores to confidence: conf = (threshold - risk) / threshold
    """
    
    def __init__(self, probe: nn.Module, threshold: float = 1.0):
        self.probe = probe
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "entropy"
    
    def estimate(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        with torch.inference_mode():
            risk = self.probe(hidden_states).squeeze(-1)
            if risk.dim() > 1:
                risk = risk[:, -1]
            
            confidence = ((self.threshold - risk) / (self.threshold + 1e-6)).clamp(0, 1)
            return confidence
    
    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold


class PerplexityEstimator(ConfidenceEstimator):
    """
    Estimates confidence based on local perplexity.
    
    Lower perplexity = higher confidence (more predictable next token).
    Uses a small projection head to estimate log-probability.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = 32000,
        max_ppl: float = 100.0,
    ):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_ppl = max_ppl
        
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "perplexity"
    
    def initialize_from_lm_head(self, lm_head: nn.Module) -> None:
        """Initialize projection from model's LM head for efficiency."""
        if hasattr(lm_head, 'weight'):
            with torch.no_grad():
                self.proj.weight.copy_(lm_head.weight)
            self._initialized = True
    
    def estimate(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        if not self._initialized:
            return torch.ones(hidden_states.shape[0], device=hidden_states.device) * 0.5
        
        with torch.inference_mode():
            if hidden_states.dim() == 3:
                h = hidden_states[:, -1, :]
            else:
                h = hidden_states
            
            logits = self.proj(h)
            probs = F.softmax(logits, dim=-1)
            
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            ppl = torch.exp(entropy)
            
            confidence = 1.0 - (ppl / self.max_ppl).clamp(0, 1)
            return confidence


class AttentionEntropyEstimator(ConfidenceEstimator):
    """
    Estimates confidence from attention pattern entropy.
    
    Focused attention (low entropy) = higher confidence.
    Diffuse attention (high entropy) = lower confidence.
    """
    
    def __init__(self, max_entropy: float = 5.0):
        self.max_entropy = max_entropy
    
    @property
    def name(self) -> str:
        return "attention_entropy"
    
    def estimate(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_weights is None:
            return torch.ones(hidden_states.shape[0], device=hidden_states.device) * 0.5
        
        with torch.inference_mode():
            if attention_weights.dim() == 4:
                attn = attention_weights.mean(dim=1)[:, -1, :]
            else:
                attn = attention_weights
            
            attn = attn.clamp(min=1e-10)
            entropy = -(attn * torch.log(attn)).sum(dim=-1)
            
            confidence = 1.0 - (entropy / self.max_entropy).clamp(0, 1)
            return confidence


class TokenConsistencyEstimator(ConfidenceEstimator):
    """
    Estimates confidence based on hidden state consistency across layers.
    
    If hidden states change rapidly between layers, confidence is lower.
    Stable representations indicate confident predictions.
    """
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self._history: Dict[int, List[torch.Tensor]] = {}
    
    @property
    def name(self) -> str:
        return "token_consistency"
    
    def estimate(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        if hidden_states.dim() == 3:
            h = hidden_states[:, -1, :].detach()
        else:
            h = hidden_states.detach()
        
        if layer_idx not in self._history:
            self._history[layer_idx] = []
        
        history = self._history[layer_idx]
        
        if len(history) == 0:
            history.append(h.clone())
            return torch.ones(batch_size, device=device) * 0.5
        
        prev_h = history[-1]
        if prev_h.shape[0] != batch_size:
            history.clear()
            history.append(h.clone())
            return torch.ones(batch_size, device=device) * 0.5
        
        cosine_sim = F.cosine_similarity(h, prev_h, dim=-1)
        
        confidence = (cosine_sim + 1) / 2
        
        history.append(h.clone())
        if len(history) > self.window_size:
            history.pop(0)
        
        return confidence
    
    def reset(self) -> None:
        """Reset history for new sequence."""
        self._history.clear()


class GatedFusion(nn.Module):
    """
    Learnable gated fusion for combining multiple confidence signals.
    
    Uses a small MLP to predict fusion weights based on input features.
    """
    
    def __init__(
        self,
        num_modalities: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        
        self.gate = nn.Sequential(
            nn.Linear(num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        confidences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            confidences: [batch, num_modalities] confidence scores
        
        Returns:
            fused: [batch] fused confidence
            weights: [batch, num_modalities] fusion weights
        """
        weights = self.gate(confidences)
        fused = (confidences * weights).sum(dim=-1)
        return fused, weights


class EnhancedConfidenceAccumulator:
    """
    Multi-modal confidence accumulator with fusion.
    
    Combines multiple confidence estimators:
    - Entropy probe (primary)
    - Perplexity
    - Attention entropy
    - Token consistency
    
    Supports:
    - Static weighted averaging
    - Learned gated fusion
    - Adaptive decay based on uncertainty
    
    Example:
        ```python
        accumulator = EnhancedConfidenceAccumulator(
            hidden_dim=2048,
            layer_indices=(7, 14, 21),
        )
        
        # Add entropy probe
        accumulator.add_estimator(EntropyProbeEstimator(probe, threshold=1.0))
        
        # During inference
        confidence = accumulator.compute_confidence(
            hidden_states,
            layer_idx=14,
            attention_weights=attn,
        )
        ```
    """
    
    def __init__(
        self,
        hidden_dim: int,
        layer_indices: Tuple[int, ...] = (7, 14, 21),
        base_decay: float = 0.9,
        use_gated_fusion: bool = False,
        fusion_weights: Optional[Dict[str, float]] = None,
    ):
        self.hidden_dim = hidden_dim
        self.layer_indices = layer_indices
        self.base_decay = base_decay
        self.use_gated_fusion = use_gated_fusion
        
        self.estimators: Dict[str, ConfidenceEstimator] = {}
        self.fusion_weights = fusion_weights or {}
        
        self.gated_fusion: Optional[GatedFusion] = None
        
        self._accumulated: Optional[torch.Tensor] = None
        self._layer_confidences: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def add_estimator(
        self,
        estimator: ConfidenceEstimator,
        weight: float = 1.0,
    ) -> None:
        """Add a confidence estimator."""
        self.estimators[estimator.name] = estimator
        self.fusion_weights[estimator.name] = weight
        
        if self.use_gated_fusion and self.gated_fusion is None:
            self._init_gated_fusion()
    
    def _init_gated_fusion(self) -> None:
        """Initialize gated fusion module."""
        if len(self.estimators) > 0:
            self.gated_fusion = GatedFusion(len(self.estimators))
    
    def compute_confidence(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        layer_weight: float = 0.33,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute fused confidence for current layer.
        
        Args:
            hidden_states: [batch, seq, hidden] or [batch, hidden]
            layer_idx: Current layer index
            layer_weight: Weight for this layer in accumulation
            **kwargs: Additional inputs (attention_weights, etc.)
        
        Returns:
            Accumulated confidence [batch]
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        if self._accumulated is None or self._accumulated.shape[0] != batch_size:
            self._accumulated = torch.zeros(batch_size, device=device, dtype=dtype)
        
        if not self.estimators:
            return self._accumulated
        
        confidences = {}
        for name, estimator in self.estimators.items():
            try:
                conf = estimator.estimate(hidden_states, layer_idx, **kwargs)
                confidences[name] = conf
            except Exception:
                confidences[name] = torch.ones(batch_size, device=device) * 0.5
        
        self._layer_confidences[layer_idx] = confidences
        
        if self.use_gated_fusion and self.gated_fusion is not None:
            conf_tensor = torch.stack(list(confidences.values()), dim=-1)
            fused, _ = self.gated_fusion(conf_tensor)
        else:
            total_weight = sum(self.fusion_weights.get(name, 1.0) for name in confidences)
            fused = torch.zeros(batch_size, device=device, dtype=dtype)
            for name, conf in confidences.items():
                weight = self.fusion_weights.get(name, 1.0) / total_weight
                fused = fused + conf * weight
        
        decay = self._compute_adaptive_decay(confidences)
        
        self._accumulated = self._accumulated * decay + fused * layer_weight
        
        return self._accumulated
    
    def _compute_adaptive_decay(
        self,
        confidences: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute adaptive decay based on confidence variance.
        
        High variance = more uncertainty = lower decay (forget faster)
        Low variance = consistent signals = higher decay (remember more)
        """
        if len(confidences) < 2:
            if isinstance(self.base_decay, torch.Tensor):
                return self.base_decay
            return torch.tensor(self.base_decay)
        
        conf_stack = torch.stack(list(confidences.values()), dim=-1)
        variance = conf_stack.var(dim=-1)
        
        decay = self.base_decay * (1.0 - variance * 0.5)
        decay = decay.clamp(0.5, 0.99)
        
        return decay
    
    def reset(self) -> None:
        """Reset accumulated confidence for new sequence."""
        self._accumulated = None
        self._layer_confidences.clear()
        
        for estimator in self.estimators.values():
            if hasattr(estimator, 'reset'):
                estimator.reset()
    
    def get_layer_breakdown(self, layer_idx: int) -> Dict[str, float]:
        """Get per-modality confidence breakdown for a layer."""
        if layer_idx not in self._layer_confidences:
            return {}
        
        return {
            name: conf.mean().item()
            for name, conf in self._layer_confidences[layer_idx].items()
        }
    
    @property
    def accumulated(self) -> Optional[torch.Tensor]:
        return self._accumulated


def adaptive_confidence_decay(
    layer_idx: int,
    historical_exit_rate: float,
    uncertainty: float,
    base_decay: float = 0.9,
) -> float:
    """
    Compute adaptive decay factor based on historical patterns.
    
    Args:
        layer_idx: Current layer index
        historical_exit_rate: Recent exit rate at this layer
        uncertainty: Current uncertainty estimate
        base_decay: Base decay factor
    
    Returns:
        Adjusted decay factor
    
    Formula:
        γ_t = γ_base × (1 - uncertainty) × (1 + historical_success_bonus)
    
    When a layer rarely triggers exits, reduce its influence on final decision.
    """
    historical_bonus = min(0.2, historical_exit_rate * 0.5)
    
    uncertainty_penalty = uncertainty * 0.3
    
    decay = base_decay * (1.0 - uncertainty_penalty) * (1.0 + historical_bonus)
    
    return max(0.5, min(0.99, decay))


def soft_exit_mlp_scaling(
    layer_idx: int,
    accumulated_confidence: float,
    confidence_threshold: float,
    total_layers: int = 36,
    steepness: float = 10.0,
) -> float:
    """
    Compute gradual MLP scaling factor for soft exit.
    
    Args:
        layer_idx: Current layer index
        accumulated_confidence: Current accumulated confidence
        confidence_threshold: Threshold for full exit
        total_layers: Total layers in model
        steepness: Steepness of sigmoid curve
    
    Returns:
        MLP scale factor: 0.0 (skip MLP) to 1.0 (full MLP)
    
    Uses sigmoid curve for smooth transition:
        scale = 1 / (1 + exp(-k * (ratio - 0.5)))
    """
    if accumulated_confidence < confidence_threshold * 0.3:
        return 1.0
    
    confidence_ratio = accumulated_confidence / (confidence_threshold + 1e-6)
    
    position_factor = 1.0 - (layer_idx / total_layers) * 0.3
    
    x = steepness * (confidence_ratio - 0.5)
    sigmoid_scale = 1.0 / (1.0 + math.exp(-x))
    
    mlp_scale = 1.0 - sigmoid_scale * position_factor
    
    return max(0.0, min(1.0, mlp_scale))
