"""
SEDAC V6.0 - Local Validation Suite
====================================

Validates trained probes and simulates cascade early-exit behavior.

Tests Include:
    - Probe inference throughput benchmark
    - Per-layer exit rate analysis
    - Cascade exit simulation with speedup estimation
    - Output quality risk assessment

Usage:
    python test_v6_local.py

Requirements:
    - Trained probes in sedac_data/sedac_probe_layer{N}.pth
    - Test data in sedac_data/hidden_states_layer{N}.pt
"""

import os
import time

import torch
import torch.nn as nn


class LREProbe(nn.Module):
    def __init__(self, input_dim: int, rank: int = 64):
        super().__init__()
        self.proj = nn.Linear(input_dim, rank, bias=False)
        self.norm = nn.LayerNorm(rank)
        self.head = nn.Linear(rank, 1)
        self.act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.norm(h)
        return self.act(self.head(h))


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 50)
    print("SEDAC V6 Multi-Layer Cascade Exit Test")
    print("=" * 50)

    hidden_dim = 2048
    layers = [7, 14, 21]  # Checkpoint layers for Qwen2.5-3B
    
    # Set thresholds based on probe prediction discriminability
    # Low entropy tokens pred ≈ 0.8-1.0, high entropy ≈ 1.8
    # Use fixed thresholds instead of percentiles, only exit truly low-entropy tokens
    use_fixed_thresholds = True
    # Balanced config: ~5% high risk, ~15% exit rate
    fixed_thresholds = {7: 0.8, 14: 0.9, 21: 0.8}
    target_exit_rates = {7: 0.05, 14: 0.15, 21: 0.35}  # Fallback only
    thresholds = {}  # Will be auto-set via calibration
    
    # Load multi-layer probes
    probes = {}
    for layer_idx in layers:
        probe_path = f"sedac_data/sedac_probe_layer{layer_idx}.pth"
        if not os.path.exists(probe_path):
            print(f"⚠ Probe not found: {probe_path}")
            continue
        probe = LREProbe(hidden_dim, rank=64)
        probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
        probe.to(device, dtype=torch.float16).eval()
        probes[layer_idx] = probe
        print(f"✓ Loaded probe layer {layer_idx}")
    
    if not probes:
        print("Error: No probes loaded")
        return 1

    # Load test data
    num_samples = 1000
    test_data = {}
    for layer_idx in layers:
        h_path = f"sedac_data/hidden_states_layer{layer_idx}.pt"
        e_path = f"sedac_data/entropies_layer{layer_idx}.pt"
        if os.path.exists(h_path):
            h = torch.load(h_path, map_location="cpu")[:num_samples].float()
            h = torch.nan_to_num(h, nan=0.0).half().to(device)
            e = torch.load(e_path, map_location="cpu")[:num_samples].float().to(device)
            test_data[layer_idx] = (h, e)
            print(f"✓ Loaded test data layer {layer_idx}: {h.shape}")

    # Threshold configuration
    print("\n--- Threshold Configuration ---")
    with torch.inference_mode():
        for layer_idx in layers:
            if layer_idx not in probes or layer_idx not in test_data:
                continue
            probe = probes[layer_idx]
            h, e = test_data[layer_idx]
            risk = probe(h).squeeze(-1).float()
            
            if use_fixed_thresholds:
                thr = fixed_thresholds[layer_idx]
            else:
                target_rate = target_exit_rates[layer_idx]
                thr = risk.quantile(target_rate).item()
            thresholds[layer_idx] = thr
            
            exit_rate = (risk < thr).float().mean().item()
            # Calculate average entropy of exited tokens
            exit_mask = risk < thr
            avg_exit_entropy = e[exit_mask].mean().item() if exit_mask.sum() > 0 else 0
            print(f"  Layer {layer_idx}: thr={thr:.3f}, exit_rate={exit_rate*100:.1f}%, avg_exit_entropy={avg_exit_entropy:.3f}")

    # Benchmark Probe Inference Speed
    print("\n--- Probe Inference Speed Benchmark ---")
    first_layer = layers[0]
    h_test = test_data[first_layer][0]
    probe = probes[first_layer]
    
    with torch.inference_mode():
        for _ in range(10):
            _ = probe(h_test[:100])
        torch.cuda.synchronize()

        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = probe(h_test)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tps = (h_test.shape[0] * iterations) / elapsed
        print(f"  Single probe throughput: {tps:,.0f} tokens/sec")
        print(f"  Latency per 1000 tokens: {elapsed / iterations * 1000:.3f} ms")

    # Evaluate Independent Exit Capability
    print("\n--- Independent Layer Exit Capability ---")
    exit_rates = {}
    high_risk_rates = {}
    with torch.inference_mode():
        for layer_idx in layers:
            if layer_idx not in probes or layer_idx not in test_data:
                continue
            
            probe = probes[layer_idx]
            h, e = test_data[layer_idx]
            thr = thresholds[layer_idx]
            
            risk = probe(h).squeeze(-1).float()
            can_exit = risk < thr
            
            exit_rate = can_exit.float().mean().item()
            exit_rates[layer_idx] = exit_rate
            
            # Calculate high risk exit ratio
            if can_exit.sum() > 0:
                high_risk = (e[can_exit] > 1.5).float().mean().item()
            else:
                high_risk = 0
            high_risk_rates[layer_idx] = high_risk
            
            print(f"  Layer {layer_idx}: exit={exit_rate*100:.1f}%, high_risk={high_risk*100:.1f}%")
    
    # Simulate Cascade Exit
    print("\n--- Cascade Exit Simulation ---")
    # Assume independent exit probability for distribution estimation
    remaining = 1.0
    exit_dist = {}
    for layer_idx in layers:
        exit_at_layer = remaining * exit_rates.get(layer_idx, 0)
        exit_dist[layer_idx] = exit_at_layer
        remaining -= exit_at_layer
        print(f"  Exit @ Layer {layer_idx}: {exit_at_layer*100:.1f}%")
    exit_dist['full'] = remaining
    print(f"  Full execution: {remaining*100:.1f}%")

    # Calculate Theoretical Speedup
    print("\n--- Theoretical Speedup Estimation ---")
    total_layers = 36  # Qwen2.5-3B has 36 layers
    baseline_cost = total_layers
    
    # Calculate actual cost based on exit distribution
    actual_cost = 0
    for layer_idx in layers:
        actual_cost += exit_dist.get(layer_idx, 0) * (layer_idx + 1)
    actual_cost += exit_dist.get('full', 0) * total_layers
    
    speedup = baseline_cost / actual_cost if actual_cost > 0 else 1.0
    savings = (1 - actual_cost / baseline_cost) * 100
    print(f"  Baseline cost: {baseline_cost} layers/token")
    print(f"  Actual cost: {actual_cost:.1f} layers/token")
    print(f"  Theoretical speedup: {speedup:.2f}x")
    print(f"  Compute savings: {savings:.1f}%")

    # Evaluate Output Quality Impact
    print("\n--- Output Quality Risk Assessment ---")
    total_high_risk = 0
    total_exits = 0
    for layer_idx in layers:
        hr = high_risk_rates.get(layer_idx, 0)
        er = exit_rates.get(layer_idx, 0)
        total_high_risk += hr * er
        total_exits += er
    
    if total_exits > 0:
        weighted_high_risk = total_high_risk / total_exits * 100
        print(f"  Weighted High-Risk Exit Rate: {weighted_high_risk:.1f}%")
        if weighted_high_risk < 5:
            print("  ✓ Low Risk: Output quality should be preserved")
        elif weighted_high_risk < 15:
            print("  ⚠ Medium Risk: Slight quality degradation possible")
        else:
            print("  ✗ High Risk: Consider reducing exit rates or increasing thresholds")

    print("\n✅ V6 Validation Complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
