# SEDAC V6.0 Proposal: Dynamic Multi-Layer Adaptive Exit

## 1. Limitations of Current V5.x Architecture

### Problem Analysis

```
Current Architecture: Fixed decision at Layer 21 → Either skip next 7 layers or execute all
```

| Issue | Impact |
|-------|--------|
| **Fixed Exit Layer** | Simple tokens still process 21 layers before exit check |
| **Binary Decision** | Only "Exit/No Exit" options, too coarse-grained |
| **Task Agnostic** | Math reasoning and simple QA use the same exit point |
| **Single Probe** | Probe only at one layer, wasting early exit opportunities |

### Theoretical Limit

Assuming a 28-layer model:
- V5.x Best Case: Skip 7 layers → **25% Compute Savings**
- Theoretical Optimal (Dynamic): Average exit at Layer 14 → **50% Compute Savings**

---

## 2. V6.0 Core Design: Cascade Early Exit

### 2.1 Architecture Diagram

```
Input Token
    ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 0-7   │ Probe₁ │ risk < θ₁ ? → Exit @ L8        │
├─────────────────────────────────────────────────────────┤
│  Layer 8-14  │ Probe₂ │ risk < θ₂ ? → Exit @ L15       │
├─────────────────────────────────────────────────────────┤
│  Layer 15-21 │ Probe₃ │ risk < θ₃ ? → Exit @ L22       │
├─────────────────────────────────────────────────────────┤
│  Layer 22-27 │ (full) │ Complete execution             │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Multi-Layer Probe Deployment Strategy

```python
# Recommended layers (for 28-layer model)
PROBE_LAYERS = [7, 14, 21]  # Checkpoint every 7 layers

# Threshold Gradient: Stricter at shallow layers, looser at deep layers
THRESHOLDS = {
    7:  0.15,   # Shallow: Exit only if extremely confident
    14: 0.25,   # Middle: Moderate confidence
    21: 0.40,   # Deep: More lenient
}
```

### 2.3 Token-Level Adaptation

Key Insight: **Different tokens have different "difficulty"**

```
"The capital of France is" → [Paris]     # Simple, exit at Layer 8
"∫x²dx = "                 → [x³/3+C]    # Complex, full 28 layers
```

V6.0 makes exit decisions for each token independently:

```python
def forward_with_cascade_exit(self, hidden_states, ...):
    exit_layer = torch.full((batch_size,), 9999, device=device)  # Exit layer per token
    
    for layer_idx, layer in enumerate(self.layers):
        hidden_states, residual = layer(positions, hidden_states, residual)
        
        if layer_idx in self.probe_layers:
            probe = self.probes[layer_idx]
            risk = probe(hidden_states)  # [batch, seq, 1]
            
            # Per-token decision
            can_exit = (risk.squeeze(-1) < self.thresholds[layer_idx])
            
            # Record first exit layer
            first_exit = (exit_layer == 9999) & can_exit
            exit_layer = torch.where(first_exit, layer_idx, exit_layer)
            
            # Skip subsequent MLPs for exited tokens
            # ...
    
    return hidden_states, exit_layer  # Return exit info for analysis
```

---

## 3. Training Scheme

### 3.1 Multi-Layer Probe Data Collection

```python
# collect_multi_layer_data.py
COLLECT_LAYERS = [7, 14, 21]

for layer_idx in COLLECT_LAYERS:
    hidden_states = extract_hidden_states(model, data, layer=layer_idx)
    final_entropy = compute_final_output_entropy(model, data)
    
    save(f"hidden_layer{layer_idx}.pt", hidden_states)
    save(f"entropy_layer{layer_idx}.pt", final_entropy)  # Use final layer entropy as label
```

### 3.2 Joint Training Objective

```python
# Multi-Probe Joint Loss
total_loss = 0
for layer_idx, probe in probes.items():
    pred_risk = probe(hidden_states[layer_idx])
    layer_loss = huber_loss(pred_risk, target_entropy)
    
    # Shallow probes need higher precision (higher cost of error)
    weight = 1.0 + (max_layer - layer_idx) / max_layer  # Higher weight for shallow layers
    total_loss += weight * layer_loss
```

### 3.3 Exit Head (Optional)

If a probe decides to exit, a lightweight Exit Head is needed to generate logits:

```python
class SharedExitHead(nn.Module):
    """Shared exit head to reduce parameters"""
    def __init__(self, hidden_size, vocab_size, num_layers=3):
        super().__init__()
        self.layer_adapters = nn.ModuleDict({
            str(l): nn.Linear(hidden_size, hidden_size) 
            for l in [7, 14, 21]
        })
        self.shared_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, hidden, layer_idx):
        adapted = self.layer_adapters[str(layer_idx)](hidden)
        return self.shared_head(adapted)
```

---

## 4. Inference Optimization

### 4.1 Batch Heterogeneity Handling

Problem: Tokens in the same batch may exit at different layers.

```python
# Scheme A: Mask-based skip (Recommended)
def forward_with_mask(self, hidden_states, active_mask):
    """
    active_mask: [batch, seq] bool tensor
    True = Token still needs computation
    """
    for layer_idx, layer in enumerate(self.layers):
        if layer_idx in self.probe_layers:
            # Run probe only for active tokens
            if active_mask.any():
                risk = self.probes[layer_idx](hidden_states[active_mask])
                can_exit = risk < self.thresholds[layer_idx]
                active_mask[active_mask.clone()] &= ~can_exit
        
        if not active_mask.any():
            break  # All tokens exited
        
        # MLP only for active tokens
        hidden_states = layer.attention(hidden_states)
        hidden_states[active_mask] = layer.mlp(hidden_states[active_mask])
    
    return hidden_states
```

### 4.2 KV Cache Compatibility

```python
# Exited tokens still need KV cache maintenance for future attention
# But MLP computation can be skipped

# Implementation in vLLM:
# 1. Attention executes normally (all tokens)
# 2. MLP executes only for active tokens
# 3. Hidden states of exited tokens remain unchanged or zeroed
```

---

## 5. Configuration Example

```bash
# sedac_v6_config.yaml
version: "6.0"

probes:
  layers: [7, 14, 21]
  rank: 64
  paths:
    7:  "sedac_data/probe_layer7.pth"
    14: "sedac_data/probe_layer14.pth"
    21: "sedac_data/probe_layer21.pth"

thresholds:
  mode: "adaptive"  # static | adaptive | learned
  initial:
    7:  0.15
    14: 0.25
    21: 0.40
  calibration:
    enabled: true
    steps: 100
    quantile: 0.9

exit_head:
  enabled: true
  type: "shared"  # shared | per_layer
  path: "sedac_data/exit_head_shared.pth"

# Task Adaptation (Optional)
task_adaptation:
  enabled: false
  classifier_path: "sedac_data/task_classifier.pth"
  profiles:
    math:      {layers: [14, 21],    thresholds: {14: 0.10, 21: 0.20}}
    qa:        {layers: [7, 14, 21], thresholds: {7: 0.30, 14: 0.40, 21: 0.50}}
    coding:    {layers: [14, 21],    thresholds: {14: 0.15, 21: 0.30}}
```

---

## 6. Implementation Roadmap

### Phase 1: Multi-Layer Probe Infrastructure (1 Week)
- [ ] Modify data collection script for multi-layer support
- [ ] Train probes for Layer 7, 14, 21
- [ ] Validate probe accuracy per layer

### Phase 2: Inference Engine Adaptation (1 Week)
- [ ] Modify `patch_vllm_surgical.py` to support multiple checkpoints
- [ ] Implement Mask-based MLP skip
- [ ] Validate KV Cache compatibility

### Phase 3: Exit Head Training (1 Week)
- [ ] Train Shared Exit Head
- [ ] Optimization via Distillation (KD from full model)
- [ ] Quality evaluation (PPL, downstream tasks)

### Phase 4: Adaptive Thresholds (Optional, 1 Week)
- [ ] Online calibration mechanism
- [ ] Train task classifier
- [ ] A/B testing framework

---

## 7. Expected Benefits

| Metric | V5.x | V6.0 Expected |
|--------|------|---------------|
| Avg Exit Layer | 21 (Fixed) | ~14 (Dynamic) |
| Compute Savings | 0-25% | **30-50%** |
| Throughput Gain | ~1.0x | **1.3-1.5x** |
| Quality Loss (PPL) | <0.1% | <0.5% |

---

## 8. Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Poor shallow exit quality | Strict thresholds + Exit Head distillation |
| Batch heterogeneity overhead | Mask-based skip optimization |
| Increased training data need | Reuse V5 data + incremental collection |
| Implementation complexity | Staged delivery, implement static multi-layer first |

---

## 9. References

1. **CALM** (Confident Adaptive Language Modeling) - Google, 2022
2. **DeeBERT** - Early Exit for BERT
3. **FastBERT** - Speed-Tunable BERT
4. **SkipDecode** - Autoregressive Skip Decoding

---

*SEDAC V6.0 - Semantic Entropy Dynamic Acceleration Core: Cascade Edition*
