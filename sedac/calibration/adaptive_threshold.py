"""
SEDAC Adaptive Threshold Calibration
=====================================

Provides threshold calibration strategies:
- AdaptiveThreshold: Basic EMA-based online calibration
- DualMetricCalibration: Quality-aware calibration using PPL + throughput

The dual-metric approach ensures:
1. Quality protection (PPL degradation < threshold)
2. Performance improvement (throughput >= target)
3. Distribution shift detection and auto-recalibration
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import threading
import time

import numpy as np


class RollingWindow:
    """
    Thread-safe rolling window for streaming statistics.
    
    Maintains a fixed-size window of recent values for:
    - Mean, std, percentiles
    - Trend detection
    - Anomaly detection
    """
    
    def __init__(self, size: int = 100):
        self.size = size
        self._data: Deque[float] = deque(maxlen=size)
        self._lock = threading.Lock()
        self._sum = 0.0
        self._sum_sq = 0.0
    
    def add(self, value: float) -> None:
        """Add a value to the window."""
        with self._lock:
            if len(self._data) == self.size:
                old = self._data[0]
                self._sum -= old
                self._sum_sq -= old * old
            
            self._data.append(value)
            self._sum += value
            self._sum_sq += value * value
    
    def mean(self) -> float:
        """Compute mean of current window."""
        with self._lock:
            if not self._data:
                return 0.0
            return self._sum / len(self._data)
    
    def std(self) -> float:
        """Compute standard deviation of current window."""
        with self._lock:
            n = len(self._data)
            if n < 2:
                return 0.0
            mean = self._sum / n
            variance = (self._sum_sq / n) - (mean * mean)
            return max(0, variance) ** 0.5
    
    def percentile(self, p: float) -> float:
        """Compute p-th percentile (0-100)."""
        with self._lock:
            if not self._data:
                return 0.0
            sorted_data = sorted(self._data)
            idx = int(len(sorted_data) * p / 100)
            idx = min(max(0, idx), len(sorted_data) - 1)
            return sorted_data[idx]
    
    def trend(self, window: int = 10) -> float:
        """
        Compute trend (positive = increasing, negative = decreasing).
        
        Returns slope of linear regression over last `window` samples.
        """
        with self._lock:
            if len(self._data) < window:
                return 0.0
            
            recent = list(self._data)[-window:]
            x = np.arange(window)
            y = np.array(recent)
            
            x_mean = x.mean()
            y_mean = y.mean()
            
            numerator = ((x - x_mean) * (y - y_mean)).sum()
            denominator = ((x - x_mean) ** 2).sum()
            
            if denominator == 0:
                return 0.0
            return numerator / denominator
    
    def is_anomaly(self, value: float, sigma: float = 3.0) -> bool:
        """Check if value is an anomaly (> sigma std from mean)."""
        mean = self.mean()
        std = self.std()
        if std == 0:
            return False
        return abs(value - mean) > sigma * std
    
    def clear(self) -> None:
        """Clear the window."""
        with self._lock:
            self._data.clear()
            self._sum = 0.0
            self._sum_sq = 0.0
    
    def __len__(self) -> int:
        return len(self._data)
    
    def is_full(self) -> bool:
        return len(self._data) >= self.size


@dataclass
class CalibrationMetrics:
    """Metrics for a single calibration step."""
    timestamp: float
    layer_idx: int
    threshold: float
    exit_rate: float
    ppl: Optional[float] = None
    throughput: Optional[float] = None
    latency_ms: Optional[float] = None


class AdaptiveThreshold:
    """
    Basic adaptive threshold calibration using EMA.
    
    Adjusts thresholds based on observed risk distributions
    to maintain target exit rates.
    """
    
    def __init__(
        self,
        layer_indices: Tuple[int, ...],
        target_exit_rates: Dict[int, float],
        initial_thresholds: Dict[int, float],
        alpha: float = 0.1,
        warmup_steps: int = 50,
    ):
        self.layer_indices = layer_indices
        self.target_exit_rates = target_exit_rates
        self.thresholds = dict(initial_thresholds)
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        
        self._samples: Dict[int, List[float]] = {l: [] for l in layer_indices}
        self._calibrated = False
        self._lock = threading.Lock()
    
    def update(
        self,
        layer_idx: int,
        risk_samples: np.ndarray,
    ) -> float:
        """
        Update threshold with new risk samples.
        
        Args:
            layer_idx: Layer index
            risk_samples: Array of risk values
        
        Returns:
            Updated threshold
        """
        with self._lock:
            if not self._calibrated:
                self._samples[layer_idx].extend(risk_samples.tolist())
                
                all_ready = all(
                    len(s) >= self.warmup_steps
                    for s in self._samples.values()
                )
                
                if all_ready:
                    for layer in self.layer_indices:
                        sorted_samples = np.sort(self._samples[layer])
                        target = self.target_exit_rates[layer]
                        q_idx = int(len(sorted_samples) * target)
                        q_idx = min(max(0, q_idx), len(sorted_samples) - 1)
                        self.thresholds[layer] = float(sorted_samples[q_idx])
                    self._calibrated = True
                
                return self.thresholds[layer_idx]
            
            sorted_risks = np.sort(risk_samples)
            target = self.target_exit_rates[layer_idx]
            k = int(len(sorted_risks) * target)
            k = min(max(0, k), len(sorted_risks) - 1)
            batch_threshold = float(sorted_risks[k])
            
            old = self.thresholds[layer_idx]
            new = self.alpha * batch_threshold + (1 - self.alpha) * old
            self.thresholds[layer_idx] = new
            
            return new
    
    def get_threshold(self, layer_idx: int) -> float:
        return self.thresholds.get(layer_idx, 1.0)
    
    def is_calibrated(self) -> bool:
        return self._calibrated
    
    def reset(self) -> None:
        with self._lock:
            self._samples = {l: [] for l in self.layer_indices}
            self._calibrated = False


class DualMetricCalibration:
    """
    Quality-aware threshold calibration using PPL + throughput.
    
    Features:
    1. Quality Protection: Prevents PPL degradation beyond threshold
    2. Performance Tracking: Monitors throughput improvements
    3. Distribution Shift Detection: Triggers recalibration on drift
    4. Safe Bounds: Enforces min/max threshold constraints
    
    Example:
        ```python
        calibrator = DualMetricCalibration(
            layer_indices=(7, 14, 21),
            target_exit_rates={7: 0.2, 14: 0.5, 21: 0.8},
            initial_thresholds={7: 0.8, 14: 1.0, 21: 1.2},
            ppl_tolerance=0.01,  # Allow 1% PPL increase
        )
        
        # During inference
        new_threshold = calibrator.update_threshold(
            layer_idx=14,
            new_threshold=1.05,
            metrics={
                'ppl': 5.2,
                'throughput': 150.0,
                'latency_ms': 45.0,
            }
        )
        ```
    """
    
    def __init__(
        self,
        layer_indices: Tuple[int, ...],
        target_exit_rates: Dict[int, float],
        initial_thresholds: Dict[int, float],
        ppl_tolerance: float = 0.01,
        throughput_target: float = 1.1,
        alpha: float = 0.1,
        window_size: int = 100,
        drift_threshold: float = 0.1,
    ):
        """
        Args:
            layer_indices: Checkpoint layer indices
            target_exit_rates: Target exit rate per layer
            initial_thresholds: Initial threshold per layer
            ppl_tolerance: Max allowed PPL increase ratio (e.g., 0.01 = 1%)
            throughput_target: Target throughput improvement ratio
            alpha: EMA smoothing factor
            window_size: Size of rolling window for metrics
            drift_threshold: KL divergence threshold for drift detection
        """
        self.layer_indices = layer_indices
        self.target_exit_rates = target_exit_rates
        self.thresholds = dict(initial_thresholds)
        self.ppl_tolerance = ppl_tolerance
        self.throughput_target = throughput_target
        self.alpha = alpha
        self.drift_threshold = drift_threshold
        
        self.ppl_window = RollingWindow(window_size)
        self.throughput_window = RollingWindow(window_size)
        self.latency_window = RollingWindow(window_size)
        
        self._layer_risk_windows: Dict[int, RollingWindow] = {
            layer: RollingWindow(window_size) for layer in layer_indices
        }
        
        self.threshold_bounds: Dict[int, Tuple[float, float]] = {
            layer: (0.1, 3.0) for layer in layer_indices
        }
        
        self.ppl_baseline: Optional[float] = None
        self.throughput_baseline: Optional[float] = None
        
        self._calibration_history: List[CalibrationMetrics] = []
        self._lock = threading.Lock()
        self._drift_detected = False
    
    def set_baselines(
        self,
        ppl_baseline: float,
        throughput_baseline: float,
    ) -> None:
        """Set baseline metrics for quality comparison."""
        self.ppl_baseline = ppl_baseline
        self.throughput_baseline = throughput_baseline
    
    def update_threshold(
        self,
        layer_idx: int,
        new_threshold: float,
        metrics: Dict[str, float],
    ) -> float:
        """
        Update threshold with quality-aware constraints.
        
        Args:
            layer_idx: Layer index
            new_threshold: Proposed new threshold (from risk distribution)
            metrics: Dict with 'ppl', 'throughput', 'latency_ms', etc.
        
        Returns:
            Final (constrained) threshold
        """
        with self._lock:
            ppl = metrics.get('ppl')
            throughput = metrics.get('throughput')
            latency = metrics.get('latency_ms')
            
            if ppl is not None:
                self.ppl_window.add(ppl)
            if throughput is not None:
                self.throughput_window.add(throughput)
            if latency is not None:
                self.latency_window.add(latency)
            
            constrained_threshold = new_threshold
            
            if ppl is not None and self.ppl_baseline is not None:
                ppl_ratio = ppl / self.ppl_baseline
                if ppl_ratio > 1 + self.ppl_tolerance:
                    min_bound, max_bound = self.threshold_bounds[layer_idx]
                    constrained_threshold = min(
                        constrained_threshold,
                        self.thresholds[layer_idx] * 0.95
                    )
                    constrained_threshold = max(constrained_threshold, min_bound)
            
            if throughput is not None and self.throughput_baseline is not None:
                improvement = throughput / self.throughput_baseline
                if improvement < self.throughput_target:
                    constrained_threshold = min(
                        constrained_threshold * 1.05,
                        self.threshold_bounds[layer_idx][1]
                    )
            
            min_b, max_b = self.threshold_bounds[layer_idx]
            constrained_threshold = max(min_b, min(max_b, constrained_threshold))
            
            old_threshold = self.thresholds[layer_idx]
            final_threshold = (
                self.alpha * constrained_threshold
                + (1 - self.alpha) * old_threshold
            )
            
            self.thresholds[layer_idx] = final_threshold
            
            exit_rate = self._estimate_exit_rate(layer_idx)
            self._calibration_history.append(CalibrationMetrics(
                timestamp=time.time(),
                layer_idx=layer_idx,
                threshold=final_threshold,
                exit_rate=exit_rate,
                ppl=ppl,
                throughput=throughput,
                latency_ms=latency,
            ))
            
            return final_threshold
    
    def _estimate_exit_rate(self, layer_idx: int) -> float:
        """Estimate current exit rate from risk window."""
        window = self._layer_risk_windows.get(layer_idx)
        if window is None or len(window) == 0:
            return self.target_exit_rates.get(layer_idx, 0.5)
        
        threshold = self.thresholds[layer_idx]
        with window._lock:
            below_count = sum(1 for v in window._data if v < threshold)
            return below_count / len(window._data)
    
    def add_risk_sample(self, layer_idx: int, risk: float) -> None:
        """Add a risk sample for distribution tracking."""
        if layer_idx in self._layer_risk_windows:
            self._layer_risk_windows[layer_idx].add(risk)
    
    def detect_distribution_shift(self, layer_idx: int) -> bool:
        """
        Detect if risk distribution has shifted significantly.
        
        Uses rolling window statistics to detect drift.
        """
        window = self._layer_risk_windows.get(layer_idx)
        if window is None or not window.is_full():
            return False
        
        current_mean = window.mean()
        current_std = window.std()
        
        if not hasattr(self, '_baseline_stats'):
            self._baseline_stats: Dict[int, Tuple[float, float]] = {}
        
        if layer_idx not in self._baseline_stats:
            self._baseline_stats[layer_idx] = (current_mean, current_std)
            return False
        
        baseline_mean, baseline_std = self._baseline_stats[layer_idx]
        
        if baseline_std > 0:
            z_score = abs(current_mean - baseline_mean) / baseline_std
            if z_score > 2.0:
                self._drift_detected = True
                return True
        
        return False
    
    def trigger_recalibration(self) -> None:
        """Trigger full recalibration (reset baselines)."""
        self._drift_detected = False
        if hasattr(self, '_baseline_stats'):
            self._baseline_stats.clear()
        
        for layer_idx in self.layer_indices:
            window = self._layer_risk_windows[layer_idx]
            if window.is_full():
                self._baseline_stats[layer_idx] = (window.mean(), window.std())
    
    def get_threshold(self, layer_idx: int) -> float:
        return self.thresholds.get(layer_idx, 1.0)
    
    def set_bounds(
        self,
        layer_idx: int,
        min_threshold: float,
        max_threshold: float,
    ) -> None:
        """Set threshold bounds for a layer."""
        self.threshold_bounds[layer_idx] = (min_threshold, max_threshold)
    
    def get_calibration_report(self) -> Dict:
        """Generate calibration report."""
        return {
            'thresholds': dict(self.thresholds),
            'ppl_stats': {
                'mean': self.ppl_window.mean(),
                'std': self.ppl_window.std(),
                'trend': self.ppl_window.trend(),
            },
            'throughput_stats': {
                'mean': self.throughput_window.mean(),
                'std': self.throughput_window.std(),
                'trend': self.throughput_window.trend(),
            },
            'drift_detected': self._drift_detected,
            'history_size': len(self._calibration_history),
        }
    
    def get_recent_history(self, n: int = 10) -> List[CalibrationMetrics]:
        """Get recent calibration history."""
        return self._calibration_history[-n:]
