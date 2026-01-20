# Contributing

Thanks for helping improve SEDAC.

## Development Setup

1. Create a Python environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Patch vLLM (required for runtime experiments):

   ```bash
   python patch_vllm_surgical.py
   python patch_vllm_openai_metrics.py
   ```

## Running Benchmarks

- Speed benchmarks:

  ```bash
  python sedac_test_suite.py --config configs/test_matrix_speed.json --out-dir artifacts/reports/speed_gptq --verbose
  ```

- fp16 variant (more VRAM required):

  ```bash
  python sedac_test_suite.py --config configs/test_matrix_speed_fp16.json --out-dir artifacts/reports/speed_fp16 --verbose
  ```

## Pull Requests

- Keep PRs focused and small when possible.
- Include a clear description of what changed and why.
- Provide reproduction steps or benchmark outputs when relevant.
- Avoid committing model weights or large datasets.

## Reporting Bugs

Please use the bug report template and include:

- GPU model and VRAM
- CUDA/driver version
- vLLM version
- Model identifier
- Config JSON used
- Full logs (attach or link)

