"""
SEDAC - Semantic Entropy Dynamic Acceleration Core
===================================================

A modular framework for LLM inference acceleration using cascade early exit.

Modules:
    - core: Cascade controller, probe inference, exit strategies
    - calibration: Adaptive threshold calibration
    - integrations: vLLM patches (V6.0, V6.1)
    - metrics: Monitoring and observability

Usage:
    from sedac.core import CascadeController, ExitStrategy
    from sedac.calibration import DualMetricCalibration
"""

__version__ = "6.1.0"
__author__ = "CARBON-XXX"

from sedac.core.cascade_controller import CascadeController, LayerConfig, ExitDecision
from sedac.core.exit_strategy import ExitStrategy, HardExit, SoftExit
from sedac.core.probe_inference import LREProbe, ProbeManager

__all__ = [
    "CascadeController",
    "LayerConfig", 
    "ExitDecision",
    "ExitStrategy",
    "HardExit",
    "SoftExit",
    "LREProbe",
    "ProbeManager",
    "__version__",
]
