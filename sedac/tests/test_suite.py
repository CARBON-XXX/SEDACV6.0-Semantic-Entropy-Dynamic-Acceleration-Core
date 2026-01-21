"""
SEDAC Comprehensive Test Suite
==============================

Multi-dimensional testing framework covering:
- Quality preservation (PPL tolerance)
- Noise robustness
- Distribution shift handling
- Latency SLO verification
- Regression testing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import time

import numpy as np
import torch


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    metric_value: float
    threshold: float
    message: str = ""
    duration_ms: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class TestConfig:
    """Configuration for test execution."""
    ppl_tolerance: float = 0.01
    latency_slo_ms: float = 100.0
    noise_sigma: float = 0.1
    min_samples: int = 100
    batch_sizes: Tuple[int, ...] = (1, 8, 32)
    domains: Tuple[str, ...] = ("general", "code", "math")


class BaseTest(ABC):
    """Abstract base class for tests."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def run(self, **kwargs) -> TestResult:
        pass


class QualityTest(BaseTest):
    """
    Quality preservation test.
    
    Verifies that SEDAC does not degrade model quality beyond tolerance.
    Compares perplexity between baseline and SEDAC-enabled inference.
    """
    
    def __init__(
        self,
        tolerance: float = 0.01,
        datasets: Optional[List[str]] = None,
    ):
        self.tolerance = tolerance
        self.datasets = datasets or ["wikitext"]
    
    @property
    def name(self) -> str:
        return "quality_preservation"
    
    def run(
        self,
        baseline_model: Callable,
        sedac_model: Callable,
        test_data: List[str],
        **kwargs,
    ) -> TestResult:
        """
        Run quality preservation test.
        
        Args:
            baseline_model: Function that returns PPL for baseline
            sedac_model: Function that returns PPL for SEDAC model
            test_data: List of test texts
        """
        start = time.time()
        
        try:
            baseline_ppl = baseline_model(test_data)
            sedac_ppl = sedac_model(test_data)
            
            ppl_increase = (sedac_ppl - baseline_ppl) / baseline_ppl
            passed = ppl_increase <= self.tolerance
            
            return TestResult(
                name=self.name,
                passed=passed,
                metric_value=ppl_increase,
                threshold=self.tolerance,
                message=f"PPL increase: {ppl_increase*100:.2f}% (limit: {self.tolerance*100:.1f}%)",
                duration_ms=(time.time() - start) * 1000,
                details={
                    "baseline_ppl": baseline_ppl,
                    "sedac_ppl": sedac_ppl,
                    "ppl_increase_pct": ppl_increase * 100,
                },
            )
        except Exception as e:
            return TestResult(
                name=self.name,
                passed=False,
                metric_value=float('inf'),
                threshold=self.tolerance,
                message=f"Test failed with error: {e}",
                duration_ms=(time.time() - start) * 1000,
            )


class RobustnessTest(BaseTest):
    """
    Noise robustness test.
    
    Tests SEDAC stability under perturbed inputs:
    - Hidden state noise injection
    - Exit decision consistency
    - Threshold stability
    """
    
    def __init__(
        self,
        noise_levels: Tuple[float, ...] = (0.01, 0.05, 0.1),
        consistency_threshold: float = 0.9,
    ):
        self.noise_levels = noise_levels
        self.consistency_threshold = consistency_threshold
    
    @property
    def name(self) -> str:
        return "noise_robustness"
    
    def run(
        self,
        cascade_controller,
        hidden_states: torch.Tensor,
        risk_scores: torch.Tensor,
        num_trials: int = 10,
        **kwargs,
    ) -> TestResult:
        """
        Run robustness test with noise injection.
        
        Args:
            cascade_controller: SEDAC CascadeController instance
            hidden_states: Sample hidden states [batch, hidden]
            risk_scores: Sample risk scores [batch]
            num_trials: Number of trials per noise level
        """
        start = time.time()
        results_by_noise = {}
        
        try:
            baseline_decisions = []
            for _ in range(num_trials):
                cascade_controller.reset_state(
                    hidden_states.shape[0],
                    hidden_states.device,
                    hidden_states.dtype,
                )
                exited, _, _ = cascade_controller.evaluate(7, risk_scores)
                baseline_decisions.append(exited.cpu().numpy())
            
            baseline_consistency = self._compute_consistency(baseline_decisions)
            
            for noise_level in self.noise_levels:
                noisy_decisions = []
                for _ in range(num_trials):
                    noise = torch.randn_like(risk_scores) * noise_level
                    noisy_risk = risk_scores + noise
                    
                    cascade_controller.reset_state(
                        hidden_states.shape[0],
                        hidden_states.device,
                        hidden_states.dtype,
                    )
                    exited, _, _ = cascade_controller.evaluate(7, noisy_risk)
                    noisy_decisions.append(exited.cpu().numpy())
                
                consistency = self._compute_consistency(noisy_decisions)
                agreement = self._compute_agreement(baseline_decisions, noisy_decisions)
                
                results_by_noise[noise_level] = {
                    "consistency": consistency,
                    "agreement_with_baseline": agreement,
                }
            
            min_consistency = min(r["consistency"] for r in results_by_noise.values())
            passed = min_consistency >= self.consistency_threshold
            
            return TestResult(
                name=self.name,
                passed=passed,
                metric_value=min_consistency,
                threshold=self.consistency_threshold,
                message=f"Min consistency: {min_consistency:.2%} across noise levels",
                duration_ms=(time.time() - start) * 1000,
                details={
                    "baseline_consistency": baseline_consistency,
                    "results_by_noise": results_by_noise,
                },
            )
        except Exception as e:
            return TestResult(
                name=self.name,
                passed=False,
                metric_value=0.0,
                threshold=self.consistency_threshold,
                message=f"Test failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )
    
    def _compute_consistency(self, decisions: List[np.ndarray]) -> float:
        """Compute decision consistency across trials."""
        if len(decisions) < 2:
            return 1.0
        
        stacked = np.stack(decisions, axis=0)
        agreement = (stacked == stacked[0]).mean()
        return float(agreement)
    
    def _compute_agreement(
        self,
        baseline: List[np.ndarray],
        noisy: List[np.ndarray],
    ) -> float:
        """Compute agreement between baseline and noisy decisions."""
        baseline_majority = (np.stack(baseline).mean(axis=0) > 0.5).astype(int)
        noisy_majority = (np.stack(noisy).mean(axis=0) > 0.5).astype(int)
        return float((baseline_majority == noisy_majority).mean())


class PerformanceTest(BaseTest):
    """
    Performance and latency test.
    
    Verifies:
    - Latency SLO compliance
    - Throughput improvements
    - Memory overhead
    """
    
    def __init__(
        self,
        latency_slo_ms: float = 100.0,
        batch_sizes: Tuple[int, ...] = (1, 8, 32, 64),
        warmup_iterations: int = 10,
        test_iterations: int = 100,
    ):
        self.latency_slo_ms = latency_slo_ms
        self.batch_sizes = batch_sizes
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
    
    @property
    def name(self) -> str:
        return "performance_latency"
    
    def run(
        self,
        inference_fn: Callable,
        input_generator: Callable,
        **kwargs,
    ) -> TestResult:
        """
        Run latency SLO test.
        
        Args:
            inference_fn: Function to run inference
            input_generator: Function to generate inputs for given batch size
        """
        start = time.time()
        results_by_batch = {}
        
        try:
            for batch_size in self.batch_sizes:
                inputs = input_generator(batch_size)
                
                for _ in range(self.warmup_iterations):
                    inference_fn(inputs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latencies = []
                for _ in range(self.test_iterations):
                    iter_start = time.perf_counter()
                    inference_fn(inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - iter_start) * 1000)
                
                latencies = np.array(latencies)
                results_by_batch[batch_size] = {
                    "mean_ms": float(latencies.mean()),
                    "p50_ms": float(np.percentile(latencies, 50)),
                    "p99_ms": float(np.percentile(latencies, 99)),
                    "std_ms": float(latencies.std()),
                }
            
            max_p99 = max(r["p99_ms"] for r in results_by_batch.values())
            passed = max_p99 <= self.latency_slo_ms
            
            return TestResult(
                name=self.name,
                passed=passed,
                metric_value=max_p99,
                threshold=self.latency_slo_ms,
                message=f"Max P99 latency: {max_p99:.2f}ms (SLO: {self.latency_slo_ms}ms)",
                duration_ms=(time.time() - start) * 1000,
                details={"results_by_batch": results_by_batch},
            )
        except Exception as e:
            return TestResult(
                name=self.name,
                passed=False,
                metric_value=float('inf'),
                threshold=self.latency_slo_ms,
                message=f"Test failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )


class DistributionShiftTest(BaseTest):
    """
    Distribution shift robustness test.
    
    Tests SEDAC behavior across different domains/distributions.
    """
    
    def __init__(
        self,
        domains: Tuple[str, ...] = ("news", "code", "medical", "social"),
        max_metric_variance: float = 0.2,
    ):
        self.domains = domains
        self.max_metric_variance = max_metric_variance
    
    @property
    def name(self) -> str:
        return "distribution_shift"
    
    def run(
        self,
        evaluate_fn: Callable,
        domain_data: Dict[str, List[str]],
        **kwargs,
    ) -> TestResult:
        """
        Run distribution shift test.
        
        Args:
            evaluate_fn: Function that returns metrics dict for given data
            domain_data: Dict mapping domain names to test data
        """
        start = time.time()
        
        try:
            domain_results = {}
            for domain in self.domains:
                if domain not in domain_data:
                    continue
                metrics = evaluate_fn(domain_data[domain])
                domain_results[domain] = metrics
            
            if len(domain_results) < 2:
                return TestResult(
                    name=self.name,
                    passed=True,
                    metric_value=0.0,
                    threshold=self.max_metric_variance,
                    message="Insufficient domains for comparison",
                    duration_ms=(time.time() - start) * 1000,
                )
            
            exit_rates = [r.get("exit_rate", 0.5) for r in domain_results.values()]
            variance = np.var(exit_rates)
            passed = variance <= self.max_metric_variance
            
            return TestResult(
                name=self.name,
                passed=passed,
                metric_value=float(variance),
                threshold=self.max_metric_variance,
                message=f"Exit rate variance across domains: {variance:.4f}",
                duration_ms=(time.time() - start) * 1000,
                details={"domain_results": domain_results},
            )
        except Exception as e:
            return TestResult(
                name=self.name,
                passed=False,
                metric_value=float('inf'),
                threshold=self.max_metric_variance,
                message=f"Test failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )


class ComprehensiveTestSuite:
    """
    Comprehensive test suite orchestrator.
    
    Runs all configured tests and generates a unified report.
    
    Example:
        ```python
        suite = ComprehensiveTestSuite(config=TestConfig(
            ppl_tolerance=0.01,
            latency_slo_ms=50.0,
        ))
        
        suite.add_test(QualityTest(tolerance=0.01))
        suite.add_test(RobustnessTest())
        suite.add_test(PerformanceTest(latency_slo_ms=50.0))
        
        results = suite.run_all(
            baseline_model=baseline_fn,
            sedac_model=sedac_fn,
            test_data=data,
        )
        
        suite.print_report(results)
        ```
    """
    
    def __init__(self, config: Optional[TestConfig] = None):
        self.config = config or TestConfig()
        self.tests: List[BaseTest] = []
    
    def add_test(self, test: BaseTest) -> None:
        """Add a test to the suite."""
        self.tests.append(test)
    
    def add_default_tests(self) -> None:
        """Add default test suite."""
        self.tests = [
            QualityTest(tolerance=self.config.ppl_tolerance),
            RobustnessTest(),
            PerformanceTest(
                latency_slo_ms=self.config.latency_slo_ms,
                batch_sizes=self.config.batch_sizes,
            ),
            DistributionShiftTest(domains=self.config.domains),
        ]
    
    def run_all(self, **kwargs) -> List[TestResult]:
        """Run all tests and return results."""
        results = []
        for test in self.tests:
            try:
                result = test.run(**kwargs)
                results.append(result)
            except Exception as e:
                results.append(TestResult(
                    name=test.name,
                    passed=False,
                    metric_value=float('inf'),
                    threshold=0.0,
                    message=f"Test execution failed: {e}",
                ))
        return results
    
    def run_test(self, test_name: str, **kwargs) -> Optional[TestResult]:
        """Run a specific test by name."""
        for test in self.tests:
            if test.name == test_name:
                return test.run(**kwargs)
        return None
    
    def print_report(self, results: List[TestResult]) -> None:
        """Print formatted test report."""
        print("\n" + "=" * 60)
        print("SEDAC TEST REPORT")
        print("=" * 60)
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} | {result.name}")
            print(f"  Metric: {result.metric_value:.4f} (threshold: {result.threshold:.4f})")
            print(f"  {result.message}")
            print(f"  Duration: {result.duration_ms:.1f}ms")
        
        print("\n" + "-" * 60)
        print(f"SUMMARY: {passed}/{total} tests passed")
        print("=" * 60 + "\n")
    
    def export_json(self, results: List[TestResult]) -> Dict:
        """Export results as JSON-serializable dict."""
        return {
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "metric_value": r.metric_value,
                    "threshold": r.threshold,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "details": r.details,
                }
                for r in results
            ],
        }
