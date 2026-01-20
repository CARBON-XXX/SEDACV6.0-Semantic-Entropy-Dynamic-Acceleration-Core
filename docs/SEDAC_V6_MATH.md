# SEDAC V6.0 Mathematical Formulation

## 1. Notation & Definitions

| Symbol | Description | Code Variable |
|--------|-------------|---------------|
| $h_l$ | Hidden state at layer $l$ | `hidden_states` |
| $D$ | Model hidden dimension | `input_dim` (e.g., 2048) |
| $R$ | Probe rank (subspace dimension) | `rank` (default: 64) |
| $r_l(h)$ | Predicted risk (semantic entropy) | `_risk` |
| $\tau_l$ | Exit threshold for layer $l$ | `sedac_thresholds` |
| $\alpha$ | Calibration smoothing factor | `sedac_calibration_alpha` |
| $\rho$ | Target exit rate | `sedac_target_exit_rates` |

## 2. Low-Rank Entropy Probe (LREProbe)

The probe maps a hidden state vector $x \in \mathbb{R}^D$ to a scalar risk score $r \in \mathbb{R}^+$.
Before processing, inputs are sanitized: $x \leftarrow \text{NanToNum}(x)$.

$$
r(x) = \text{Softplus}(W_2 \cdot \text{LayerNorm}(W_1 x))
$$

**Strict Implementation Details:**
- **Projection**: $W_1 \in \mathbb{R}^{D \times R}$ (Code: `nn.Linear(input_dim, rank, bias=False)`). The linear transformation is $x W_1$.
- **Normalization**: $\text{LayerNorm}(z) = \frac{z - \mu}{\sigma} \odot \gamma + \beta$, where $\gamma, \beta \in \mathbb{R}^R$ are learnable affine parameters.
- **Head**: $W_2 \in \mathbb{R}^{R \times 1}$ (Code: `nn.Linear(rank, 1, bias=True)`).
- **Activation**: $\text{Softplus}(z) = \log(1 + \exp(z))$.

**Code Reference:** [patch_vllm_surgical.py](patch_vllm_surgical.py) `LREProbe` class

## 3. Training Objective

The probes are trained to predict the **log-transformed semantic entropy** of the generation.

### 3.1 Target Transformation
Let $S(x)$ be the semantic entropy of the sequence given hidden state $x$. The regression target $y$ is clamped to remove outliers:

$$
y_{raw} = \log(1 + S(x))
$$
$$
y = \text{Clamp}(y_{raw}, q_{0.01}, q_{0.99})
$$

Where $q_{0.01}, q_{0.99}$ are the 1st and 99th percentiles of the log-entropy distribution in the training batch.

### 3.2 Loss Function
We minimize the Huber Loss (Smooth L1) with $\beta=0.5$ to be robust against noise:

$$
\mathcal{L}(r, y) = \begin{cases}
0.5 (r - y)^2 / \beta, & \text{if } |r - y| < \beta \\
|r - y| - 0.5 \beta, & \text{otherwise}
\end{cases}
$$

**Code Reference:** [train_multilayer_probes.py](train_multilayer_probes.py) `train_single_probe` function

## 4. Inference Logic: Sequence-Level Latch

SEDAC V6 uses a **Sequence-Level Latch** to ensure KV-cache consistency.

### 4.1 Exit Condition
At each checkpoint layer $l \in \{7, 14, 21\}$, for a sequence of tokens with hidden states $H_l = \{h_{l,1}, \dots, h_{l,S}\}$:

$$
\text{Risk}_{seq} = \max_{i=1}^S r_l(h_{l,i})
$$

The exit decision $E_l \in \{0, 1\}$ is:

$$
E_l = \mathbb{I}(\text{Risk}_{seq} < \tau_l)
$$

### 4.2 Cascade Execution Flow
If $E_l = 1$ (Exit Triggered):
1.  **Latch State**: Global flag `self._sedac_exited` is set to `True`.
2.  **Forward Pass ($k > l$)**:
    -   **Attention**: Executed normally to update KV cache.
    -   **MLP Skip**: The MLP block is bypassed.
        $$
        h_{k, \text{out}} = h_{k, \text{attn}} + \mathbf{0} \quad (\text{Residual connection carries the signal})
        $$
    -   **LayerNorms**: Both Pre-Attn and Post-Attn LayerNorms are executed to maintain statistical stability.

**Code Reference:** [patch_vllm_surgical.py](patch_vllm_surgical.py) `forward_code_v6` & `decoder_patch_v6`

## 5. Adaptive Threshold Calibration

When `SEDAC_ADAPTIVE=1`, thresholds $\tau_l$ are updated online.

### 5.1 Quantile Estimation
Let $\mathcal{R}_l$ be a rolling buffer of history risk scores. The raw target threshold $\hat{\tau}_l$ for a target exit rate $\rho$ is:

$$
\hat{\tau}_l = \text{Quantile}(\mathcal{R}_l, \rho)
$$

### 5.2 EMA Smoothing
To prevent oscillation, the operating threshold is updated via Exponential Moving Average (EMA):

$$
\tau_{l}^{(t+1)} = (1 - \alpha) \cdot \tau_{l}^{(t)} + \alpha \cdot \hat{\tau}_l
$$

- $\alpha \in [0, 1]$ (`SEDAC_CALIBRATION_ALPHA`).
- Logic ensures $\tau_l$ only updates after a calibration warmup period (`SEDAC_CALIBRATION_STEPS`).

**Code Reference:** [patch_vllm_surgical.py](patch_vllm_surgical.py) (Calibration Block)

## 6. Complexity Analysis

- **Probe Cost**: $O(S \cdot D \cdot R)$ per checkpoint.
- **MLP Savings**: $O(S \cdot D \cdot 4D)$ per skipped layer (SwiGLU gate adds params).
- **Overhead Ratio**: Since $D \approx 32 R$, the probe is $\sim \frac{1}{128}$ the cost of an MLP block.
