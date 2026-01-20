# Changelog

## V6.0 (2024-01)

### Major Changes

- **Multi-layer Cascade Exit**: Replaced single-layer exit with cascade architecture
  - Probes at layers 7, 14, 21 for progressive confidence assessment
  - Token-level early exit decisions at each checkpoint

- **Adaptive Threshold Calibration**: Runtime threshold optimization
  - Collects risk distribution during warmup phase
  - Automatically calibrates thresholds based on target exit rates
  - Configurable via `SEDAC_ADAPTIVE` and `SEDAC_EXIT_RATES`

- **Per-layer Entropy Training**: Fixed data collection methodology
  - Each layer now trained on its own exit entropy (not shared final-layer entropy)
  - Improved probe prediction accuracy and correlation

### Performance

- Probe inference: ~5-8M tokens/sec
- Theoretical speedup: 1.08x-1.25x (task-dependent)
- Memory overhead: <1MB per probe

### Breaking Changes

- Environment variables renamed for V6 compatibility
- Probe file format: `sedac_probe_layer{N}.pth` (layer-specific)
- Removed single-layer mode (`SEDAC_LAYER` deprecated)

### Files

- `patch_vllm_surgical.py` - V6 vLLM patch
- `collect_multilayer_data.py` - Multi-layer data collection
- `train_multilayer_probes.py` - Batch probe training
- `test_v6_local.py` - Local validation suite

---

## V5.0 (2023)

- Initial release with single-layer early exit
- Fixed exit layer configuration
- Basic threshold calibration
