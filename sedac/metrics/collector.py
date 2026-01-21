"""
SEDAC Metrics Collector
=======================

Fine-grained metrics collection and analysis for SEDAC inference.

Features:
- Per-layer exit distribution tracking
- Token-level decision logging
- Quality impact estimation
- Prometheus metrics export
- Real-time dashboard support
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import threading
import time

import numpy as np


@dataclass
class TokenDecisionLog:
    """Log entry for a single token's exit decision."""
    token_id: int
    layer_idx: int
    risk_score: float
    threshold: float
    accumulated_confidence: float
    should_exit: bool
    soft_exit_ratio: float
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LayerMetrics:
    """Aggregated metrics for a single checkpoint layer."""
    layer_idx: int
    exit_count: int = 0
    total_count: int = 0
    exit_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_risk: float = 0.0
    avg_threshold: float = 0.0
    avg_latency_ms: float = 0.0
    ppl_when_exit: Optional[float] = None
    ppl_when_full: Optional[float] = None
    
    def update(
        self,
        exited: bool,
        confidence: float,
        risk: float,
        threshold: float,
        latency_ms: float,
    ) -> None:
        """Update metrics with a new sample."""
        self.total_count += 1
        if exited:
            self.exit_count += 1
        
        self.exit_rate = self.exit_count / max(1, self.total_count)
        
        n = self.total_count
        self.avg_confidence = (self.avg_confidence * (n - 1) + confidence) / n
        self.avg_risk = (self.avg_risk * (n - 1) + risk) / n
        self.avg_threshold = (self.avg_threshold * (n - 1) + threshold) / n
        self.avg_latency_ms = (self.avg_latency_ms * (n - 1) + latency_ms) / n
    
    def to_dict(self) -> Dict:
        return {
            'layer_idx': self.layer_idx,
            'exit_count': self.exit_count,
            'total_count': self.total_count,
            'exit_rate': self.exit_rate,
            'avg_confidence': self.avg_confidence,
            'avg_risk': self.avg_risk,
            'avg_threshold': self.avg_threshold,
            'avg_latency_ms': self.avg_latency_ms,
            'ppl_when_exit': self.ppl_when_exit,
            'ppl_when_full': self.ppl_when_full,
        }


class SEDACMetricsCollector:
    """
    Comprehensive metrics collector for SEDAC inference.
    
    Tracks:
    - Per-layer exit statistics
    - Token-level decision logs
    - Confidence distributions
    - Latency profiles
    - Quality metrics
    
    Example:
        ```python
        collector = SEDACMetricsCollector(layer_indices=(7, 14, 21))
        
        # During inference
        collector.record_token_decision(
            token_id=42,
            layer_idx=14,
            exit_decision=decision,
            latency=5.2,
        )
        
        # Analysis
        report = collector.get_layer_analysis()
        collector.export_prometheus_metrics()
        ```
    """
    
    def __init__(
        self,
        layer_indices: Tuple[int, ...] = (7, 14, 21),
        log_capacity: int = 10000,
        enable_detailed_logging: bool = True,
    ):
        self.layer_indices = layer_indices
        self.log_capacity = log_capacity
        self.enable_detailed_logging = enable_detailed_logging
        
        self.layer_metrics: Dict[int, LayerMetrics] = {
            layer: LayerMetrics(layer_idx=layer)
            for layer in layer_indices
        }
        
        self.request_log: Deque[TokenDecisionLog] = deque(maxlen=log_capacity)
        
        self._risk_histograms: Dict[int, List[float]] = {
            layer: [] for layer in layer_indices
        }
        self._confidence_histograms: Dict[int, List[float]] = {
            layer: [] for layer in layer_indices
        }
        self._latency_histograms: Dict[int, List[float]] = {
            layer: [] for layer in layer_indices
        }
        
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._total_tokens = 0
        self._total_exits = 0
        
        self._prometheus_metrics: Optional[Dict] = None
        self._init_prometheus()
    
    def _init_prometheus(self) -> None:
        """Initialize Prometheus metrics if available."""
        try:
            from prometheus_client import Counter, Histogram, Gauge
            
            self._prometheus_metrics = {
                'calls': Counter(
                    'sedac_calls_total',
                    'Total SEDAC decisions evaluated',
                    ['layer']
                ),
                'exits': Counter(
                    'sedac_exits_total',
                    'Total SEDAC exits triggered',
                    ['layer']
                ),
                'confidence': Histogram(
                    'sedac_confidence',
                    'Accumulated confidence distribution',
                    ['layer'],
                    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ),
                'latency': Histogram(
                    'sedac_latency_ms',
                    'Decision latency in milliseconds',
                    ['layer'],
                    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
                ),
                'threshold': Gauge(
                    'sedac_threshold',
                    'Current threshold value',
                    ['layer']
                ),
                'exit_rate': Gauge(
                    'sedac_exit_rate',
                    'Current exit rate',
                    ['layer']
                ),
            }
        except ImportError:
            self._prometheus_metrics = None
    
    def record_token_decision(
        self,
        token_id: int,
        layer_idx: int,
        risk_score: float,
        threshold: float,
        accumulated_confidence: float,
        should_exit: bool,
        soft_exit_ratio: float,
        latency_ms: float,
    ) -> None:
        """
        Record a single token's exit decision.
        
        Args:
            token_id: Token identifier
            layer_idx: Layer where decision was made
            risk_score: Risk score from probe
            threshold: Threshold used for decision
            accumulated_confidence: Accumulated confidence at this layer
            should_exit: Whether token exited
            soft_exit_ratio: Soft exit ratio (0-1)
            latency_ms: Decision latency in milliseconds
        """
        with self._lock:
            self._total_tokens += 1
            if should_exit:
                self._total_exits += 1
            
            if layer_idx in self.layer_metrics:
                self.layer_metrics[layer_idx].update(
                    exited=should_exit,
                    confidence=accumulated_confidence,
                    risk=risk_score,
                    threshold=threshold,
                    latency_ms=latency_ms,
                )
            
            if layer_idx in self._risk_histograms:
                self._risk_histograms[layer_idx].append(risk_score)
                self._confidence_histograms[layer_idx].append(accumulated_confidence)
                self._latency_histograms[layer_idx].append(latency_ms)
                
                max_hist = 10000
                if len(self._risk_histograms[layer_idx]) > max_hist:
                    self._risk_histograms[layer_idx] = self._risk_histograms[layer_idx][-max_hist:]
                    self._confidence_histograms[layer_idx] = self._confidence_histograms[layer_idx][-max_hist:]
                    self._latency_histograms[layer_idx] = self._latency_histograms[layer_idx][-max_hist:]
            
            if self.enable_detailed_logging:
                self.request_log.append(TokenDecisionLog(
                    token_id=token_id,
                    layer_idx=layer_idx,
                    risk_score=risk_score,
                    threshold=threshold,
                    accumulated_confidence=accumulated_confidence,
                    should_exit=should_exit,
                    soft_exit_ratio=soft_exit_ratio,
                    latency_ms=latency_ms,
                ))
            
            if self._prometheus_metrics:
                layer_str = str(layer_idx)
                self._prometheus_metrics['calls'].labels(layer=layer_str).inc()
                if should_exit:
                    self._prometheus_metrics['exits'].labels(layer=layer_str).inc()
                self._prometheus_metrics['confidence'].labels(layer=layer_str).observe(accumulated_confidence)
                self._prometheus_metrics['latency'].labels(layer=layer_str).observe(latency_ms)
                self._prometheus_metrics['threshold'].labels(layer=layer_str).set(threshold)
                self._prometheus_metrics['exit_rate'].labels(layer=layer_str).set(
                    self.layer_metrics[layer_idx].exit_rate
                )
    
    def record_batch(
        self,
        layer_idx: int,
        risk_scores: np.ndarray,
        thresholds: np.ndarray,
        confidences: np.ndarray,
        exit_mask: np.ndarray,
        soft_ratios: np.ndarray,
        latency_ms: float,
    ) -> None:
        """Record a batch of token decisions efficiently."""
        batch_size = len(risk_scores)
        per_token_latency = latency_ms / max(1, batch_size)
        
        for i in range(batch_size):
            self.record_token_decision(
                token_id=i,
                layer_idx=layer_idx,
                risk_score=float(risk_scores[i]),
                threshold=float(thresholds[i]) if len(thresholds.shape) > 0 else float(thresholds),
                accumulated_confidence=float(confidences[i]),
                should_exit=bool(exit_mask[i]),
                soft_exit_ratio=float(soft_ratios[i]),
                latency_ms=per_token_latency,
            )
    
    def get_layer_analysis(self) -> Dict[int, Dict]:
        """
        Generate per-layer analysis report.
        
        Returns:
            Dict mapping layer_idx to analysis dict containing:
            - exit_distribution: Histogram of exit confidences
            - confidence_calibration: Calibration curve data
            - quality_impact: Estimated quality impact
        """
        with self._lock:
            analysis = {}
            
            for layer_idx in self.layer_indices:
                risks = np.array(self._risk_histograms.get(layer_idx, []))
                confs = np.array(self._confidence_histograms.get(layer_idx, []))
                lats = np.array(self._latency_histograms.get(layer_idx, []))
                metrics = self.layer_metrics[layer_idx]
                
                analysis[layer_idx] = {
                    'metrics': metrics.to_dict(),
                    'exit_distribution': {
                        'confidence_mean': float(confs.mean()) if len(confs) > 0 else 0,
                        'confidence_std': float(confs.std()) if len(confs) > 0 else 0,
                        'confidence_p50': float(np.percentile(confs, 50)) if len(confs) > 0 else 0,
                        'confidence_p90': float(np.percentile(confs, 90)) if len(confs) > 0 else 0,
                    },
                    'risk_distribution': {
                        'risk_mean': float(risks.mean()) if len(risks) > 0 else 0,
                        'risk_std': float(risks.std()) if len(risks) > 0 else 0,
                        'risk_p50': float(np.percentile(risks, 50)) if len(risks) > 0 else 0,
                        'risk_p90': float(np.percentile(risks, 90)) if len(risks) > 0 else 0,
                    },
                    'latency_distribution': {
                        'latency_mean': float(lats.mean()) if len(lats) > 0 else 0,
                        'latency_p50': float(np.percentile(lats, 50)) if len(lats) > 0 else 0,
                        'latency_p99': float(np.percentile(lats, 99)) if len(lats) > 0 else 0,
                    },
                    'quality_impact': self._estimate_quality_impact(layer_idx),
                }
            
            return analysis
    
    def _estimate_quality_impact(self, layer_idx: int) -> Dict:
        """Estimate quality impact of exits at this layer."""
        metrics = self.layer_metrics[layer_idx]
        
        layer_position = layer_idx / 36
        exit_rate = metrics.exit_rate
        estimated_ppl_increase = exit_rate * (1 - layer_position) * 0.05
        
        return {
            'estimated_ppl_increase_pct': estimated_ppl_increase * 100,
            'exit_rate': exit_rate,
            'layer_depth_ratio': layer_position,
            'risk_level': 'low' if estimated_ppl_increase < 0.01 else (
                'medium' if estimated_ppl_increase < 0.02 else 'high'
            ),
        }
    
    def get_summary(self) -> Dict:
        """Get overall metrics summary."""
        elapsed = time.time() - self._start_time
        
        return {
            'total_tokens': self._total_tokens,
            'total_exits': self._total_exits,
            'overall_exit_rate': self._total_exits / max(1, self._total_tokens),
            'elapsed_seconds': elapsed,
            'tokens_per_second': self._total_tokens / max(1, elapsed),
            'layer_metrics': {
                layer: metrics.to_dict()
                for layer, metrics in self.layer_metrics.items()
            },
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus text format."""
        try:
            from prometheus_client import generate_latest
            return generate_latest().decode('utf-8')
        except ImportError:
            lines = []
            for layer, metrics in self.layer_metrics.items():
                lines.append(f'sedac_exit_rate{{layer="{layer}"}} {metrics.exit_rate}')
                lines.append(f'sedac_exit_count{{layer="{layer}"}} {metrics.exit_count}')
                lines.append(f'sedac_total_count{{layer="{layer}"}} {metrics.total_count}')
                lines.append(f'sedac_avg_confidence{{layer="{layer}"}} {metrics.avg_confidence}')
                lines.append(f'sedac_avg_latency_ms{{layer="{layer}"}} {metrics.avg_latency_ms}')
            return '\n'.join(lines)
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for layer in self.layer_indices:
                self.layer_metrics[layer] = LayerMetrics(layer_idx=layer)
                self._risk_histograms[layer] = []
                self._confidence_histograms[layer] = []
                self._latency_histograms[layer] = []
            
            self.request_log.clear()
            self._total_tokens = 0
            self._total_exits = 0
            self._start_time = time.time()
    
    def get_recent_logs(self, n: int = 100) -> List[TokenDecisionLog]:
        """Get recent token decision logs."""
        with self._lock:
            return list(self.request_log)[-n:]
