# SEDAC: Safe Early-Exit & Speculative Decoding Toolkit

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#performance">Performance</a> •
  <a href="#faq">FAQ</a>
</p>
my email:jasonuzi12@gmail.com
## Introduction

**SEDAC** (Speculative Early-Exit Decoding with Adaptive Calibration) is a research toolkit designed to accelerate Large Language Model (LLM) inference without sacrificing generation quality.

It addresses a critical flaw in traditional Early-Exit mechanisms on Decoder-only architectures (like Llama, Qwen): **KV Cache Corruption**. By introducing a novel **MLP-Skipping** architecture (SEDAC v5), this toolkit achieves real speedups while maintaining **bit-level accuracy** (PPL ratio ~1.00) compared to the baseline.

## Key Features

### 🚀 SEDAC v5: Safe MLP-Skipping
Traditional Early-Exit methods skip entire layers, which leaves the KV Cache uninitialized for skipped layers. This causes catastrophic quality degradation (PPL > 10^5) for subsequent tokens.

**SEDAC v5 solves this by:**
1.  **Always computing Attention**: Ensuring KV Cache is perfectly maintained for every layer, every token.
2.  **Skipping MLPs**: If the model is confident (Entropy < Threshold), we skip the Feed-Forward Network (MLP) block, which accounts for ~65% of the parameters.
3.  **Result**: **Zero PPL degradation** with measurable speedup.

### 🛡️ Adaptive Safety
- **Max-Entropy Decision**: In batched inference, SEDAC exits only if **all** tokens in the batch are confident. This prevents "easy" tokens from forcing "hard" tokens to skip computation.
- **Dynamic Thresholding**: (Optional) Calibrates the exit threshold based on the prompt difficulty at runtime.

### 📊 Comprehensive Benchmarking
- **End-to-End Suite**: Tests TPS (Tokens/sec), Speedup, PPL (Perplexity), and Acceptance Rate.
- **Multi-Mode**: Supports **Online** (OpenAI-compatible HTTP server) and **Offline** (vLLM Engine) benchmarking.
- **Metrics**: Automatically exports Acceptance Rate (AR) and Token Recovery Rate (TRR) for speculative decoding analysis.

## Installation

1. **Clone and Install Dependencies**:
   ```bash
   git clone https://github.com/your-org/SEDAC.git
   cd SEDAC
   pip install -r requirements.txt
   ```

2. **(Optional) Suffix Decoding Backend**:
   If you plan to use Suffix Decoding:
   ```bash
   pip install arctic-inference==0.1.1
   ```

## Quickstart

### 1. Patch vLLM
Apply the SEDAC hooks to your vLLM installation. This script patches the `Qwen2` model definition to support MLP skipping and probe injection.

```bash
python3 patch_vllm_surgical.py
```

### 2. Start the Server
Launch an OpenAI-compatible server with SEDAC enabled.
- `--sedac-layer 24`: Start checking for exits at Layer 24 (of 36).
- `--sedac-threshold 0.45`: Conservative threshold for safe acceleration.

```bash
python3 sedac_start_server.py \
    --model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 \
    --sedac-layer 24 \
    --sedac-threshold 0.45 \
    --port 8000
```

### 3. Run Benchmark
Run the test suite to evaluate speed and quality.

```bash
# Run a quick speed test
python3 sedac_test_suite.py --config configs/test_matrix_speed.json --verbose
```

## Performance

### Qwen2.5-3B-Instruct (Int4)
*Tested on single GPU. Baseline TPS: ~36 tokens/s.*

| Configuration | Speedup | PPL Ratio | Quality |
| :--- | :---: | :---: | :--- |
| **Vanilla Baseline** | 1.00x | 1.00 | Reference |
| **SEDAC v5 (Safe MLP-Skip)** | **~1.05x - 1.15x** | **1.00** | **Lossless** |
| SEDAC (Aggressive Latch) | ~1.43x | >1000 | ❌ Broken (Model Collapse) |

> **Note**: The speedup on 3B models is limited because they are often memory-bound or latency-bound. The overhead of the Python control plane competes with the small compute savings.

### Does Size Matter?
**Yes.** SEDAC's speedup potential increases with model size.

*   **Small Models (3B)**: Hard to accelerate. The compute time saved by skipping an MLP is small (e.g., 0.5ms), which is comparable to the overhead of the decision logic.
*   **Large Models (14B/32B/70B)**: **High Potential**. The compute time for a single MLP block is significant (e.g., 5-10ms). Skipping it yields a net positive even with overhead.
*   **Recommendation**: For production acceleration, target models **>7B parameters**.

## FAQ

**Q: Why did previous Early-Exit methods fail on vLLM?**
A: They skipped Attention layers. In vLLM's PagedAttention, if you don't write to the KV Cache for a token at Layer N, the next token will read garbage memory when it tries to attend to Layer N. SEDAC v5 fixes this by **never skipping Attention**.

**Q: Can I use this with Speculative Decoding (Draft Models)?**
A: Yes! SEDAC is orthogonal to Speculative Decoding. You can combine them (e.g., `ngram_sedac_adaptive` config) to get speedups from both draft matching AND MLP skipping in the verification phase.

**Q: How do I train a Probe?**
A: See `train_probe.py`. You need to collect hidden states from the target layer and train a small linear classifier to predict the entropy of the final output.

## License

MIT
