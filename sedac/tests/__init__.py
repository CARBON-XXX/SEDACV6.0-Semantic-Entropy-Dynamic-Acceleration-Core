"""
SEDAC Test Suite
================

Comprehensive testing framework for SEDAC:
- Quality preservation tests (PPL tolerance)
- Noise robustness tests
- Distribution shift tests
- Latency SLO verification
"""

from sedac.tests.test_suite import (
    ComprehensiveTestSuite,
    QualityTest,
    RobustnessTest,
    PerformanceTest,
)

__all__ = [
    "ComprehensiveTestSuite",
    "QualityTest",
    "RobustnessTest", 
    "PerformanceTest",
]
