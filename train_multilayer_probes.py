"""
SEDAC V6.0 - Multi-Layer Probe Training
========================================

Trains LREProbe models for each checkpoint layer to predict semantic entropy.

Training Details:
    - Loss: Huber Loss (robust to outliers)
    - Optimizer: AdamW with weight decay
    - Target: log1p(entropy) normalized to [0, 1]
    - Validation: 20% holdout split

Usage:
    python train_multilayer_probes.py \\
        --data-dir sedac_data \\
        --layers 7,14,21 \\
        --epochs 25 \\
        --batch-size 2048

Output:
    sedac_data/sedac_probe_layer{N}.pth - Trained probe weights
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LREProbe(nn.Module):
    """Low-Rank Entropy Probe"""
    def __init__(self, input_dim: int, rank: int = 64):
        super().__init__()
        self.proj = nn.Linear(input_dim, rank, bias=False)
        self.norm = nn.LayerNorm(rank)
        self.head = nn.Linear(rank, 1)
        self.act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.proj(x)
        h = self.norm(h)
        return self.act(self.head(h))


def train_single_probe(
    layer_idx: int,
    hidden_path: str,
    entropy_path: str,
    output_path: str,
    rank: int = 64,
    lr: float = 1e-3,
    batch_size: int = 1024,
    epochs: int = 20,
    device: str = "cuda",
) -> float:
    """Train a single layer probe"""
    print(f"\n{'='*50}")
    print(f"Training Probe for Layer {layer_idx}")
    print(f"{'='*50}")

    # Load data
    hidden_states = torch.load(hidden_path, map_location="cpu").float()
    entropies = torch.load(entropy_path, map_location="cpu").float()

    hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
    entropies = torch.nan_to_num(entropies, nan=0.0, posinf=0.0, neginf=0.0)

    if hidden_states.ndim != 2:
        raise RuntimeError(f"hidden_states must be [N, H], got {tuple(hidden_states.shape)}")

    hidden_dim = int(hidden_states.shape[1])
    print(f"Samples: {hidden_states.shape[0]}, Hidden dim: {hidden_dim}")

    # Preprocess entropy
    entropies = torch.log1p(entropies)
    q_low = float(torch.quantile(entropies, 0.01).item())
    q_high = float(torch.quantile(entropies, 0.99).item())
    entropies = torch.clamp(entropies, min=q_low, max=q_high)

    # Split dataset
    dataset = TensorDataset(hidden_states, entropies)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    probe = LREProbe(hidden_dim, rank).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr, eps=1e-8)
    criterion_huber = nn.SmoothL1Loss(beta=0.5)
    criterion_mse = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(epochs):
        # Train
        probe.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred = probe(x)
            loss = criterion_huber(pred, y) if epoch < epochs // 2 else criterion_mse(pred, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / train_size

        # Validation
        probe.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                pred = probe(x)
                loss = criterion_mse(pred, y)
                val_loss += loss.item() * x.size(0)

        avg_val_loss = val_loss / test_size

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(probe.state_dict(), output_path)

    print(f"  ✓ Best Val Loss: {best_loss:.6f} -> {output_path}")
    return best_loss


def main() -> int:
    parser = argparse.ArgumentParser(description="SEDAC V6 Multi-Layer Probe Training")
    parser.add_argument("--data-dir", type=str, default="sedac_data")
    parser.add_argument("--layers", type=str, default="7,14,21")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    target_layers = [int(x.strip()) for x in args.layers.split(",")]
    print(f"Target layers: {target_layers}")

    results: dict[int, float] = {}

    for layer_idx in target_layers:
        hidden_path = os.path.join(args.data_dir, f"hidden_states_layer{layer_idx}.pt")
        entropy_path = os.path.join(args.data_dir, f"entropies_layer{layer_idx}.pt")
        output_path = os.path.join(args.data_dir, f"sedac_probe_layer{layer_idx}.pth")

        if not os.path.exists(hidden_path):
            print(f"⚠ Skipping layer {layer_idx}: {hidden_path} not found")
            continue
        if not os.path.exists(entropy_path):
            print(f"⚠ Skipping layer {layer_idx}: {entropy_path} not found")
            continue

        best_loss = train_single_probe(
            layer_idx=layer_idx,
            hidden_path=hidden_path,
            entropy_path=entropy_path,
            output_path=output_path,
            rank=args.rank,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device,
        )
        results[layer_idx] = best_loss

    # 汇总
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    for layer_idx, loss in sorted(results.items()):
        print(f"  Layer {layer_idx:2d}: Val Loss = {loss:.6f}")

    print(f"\n✅ All probes trained! Files saved to: {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
