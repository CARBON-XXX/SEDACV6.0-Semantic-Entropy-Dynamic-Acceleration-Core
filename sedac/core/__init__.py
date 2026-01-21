"""
SEDAC Core Module
=================

Contains the fundamental components for cascade early exit:
- CascadeController: Main orchestrator for multi-layer exit decisions
- ProbeInference: LREProbe model and batch inference
- ExitStrategy: Hard/Soft exit implementations
"""

from sedac.core.cascade_controller import (
    CascadeController,
    LayerConfig,
    ExitDecision,
)
from sedac.core.probe_inference import LREProbe, ProbeManager
from sedac.core.exit_strategy import ExitStrategy, HardExit, SoftExit

__all__ = [
    "CascadeController",
    "LayerConfig",
    "ExitDecision",
    "LREProbe",
    "ProbeManager",
    "ExitStrategy",
    "HardExit",
    "SoftExit",
]
