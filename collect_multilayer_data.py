"""
SEDAC V6.0 - Multi-Layer Data Collection
=========================================

Collects hidden states and per-layer entropy from multiple checkpoint layers
for training cascade early-exit probes.

Key Features:
    - Hook-based activation collection at specified layers
    - Per-layer entropy calculation (not shared final-layer entropy)
    - Support for 4-bit quantization and Flash Attention 2
    - Memory-efficient batch processing with periodic cleanup

Usage:
    python collect_multilayer_data.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --layers 7,14,21 \\
        --samples 500 \\
        --save-dir sedac_data

Output:
    sedac_data/hidden_states_layer{N}.pt  - Hidden states [num_tokens, hidden_dim]
    sedac_data/entropies_layer{N}.pt      - Per-layer entropy [num_tokens]
"""

from __future__ import annotations

import argparse
import gc
import os
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SEDAC V6 Multi-Layer Data Collection")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--save-dir", type=str, default="sedac_data")
    parser.add_argument(
        "--layers",
        type=str,
        default="7,14,21",
        help="Comma-separated layer indices to collect (0-indexed)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true", help="Use BnB 4-bit quantization")
    return parser.parse_args()


class MultiLayerCollector:
    """Multi-layer hidden states collector"""

    def __init__(self, model: Any, target_layers: list[int]):
        self.model = model
        self.target_layers = target_layers
        self.activations: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}
        self.hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks to target layers"""
        for layer_idx in self.target_layers:
            hook = self.model.model.layers[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        """Create hook function"""
        def hook(module: Any, input: Any, output: Any) -> None:
            # output[0]: [batch, seq, hidden]
            self.activations[layer_idx].append(output[0].detach().cpu())
        return hook

    def clear(self) -> None:
        """Clear collected activations"""
        for layer_idx in self.target_layers:
            self.activations[layer_idx] = []

    def remove_hooks(self) -> None:
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self, layer_idx: int) -> list[torch.Tensor]:
        """Get activations for a specific layer"""
        return self.activations[layer_idx]


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Calculate softmax entropy"""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def main() -> int:
    args = parse_args()

    target_layers = [int(x.strip()) for x in args.layers.split(",")]
    print(f"Target layers: {target_layers}")

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载模型
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "attn_implementation": "flash_attention_2",  # Faster attention
    }
    
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs.pop("torch_dtype", None)
        print("Using BitsAndBytes 4-bit quantization")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except Exception:
        # Fallback: disable flash attention
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    # Get number of model layers
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")

    # Validate target layers
    for layer_idx in target_layers:
        if layer_idx >= num_layers:
            print(f"Error: Layer {layer_idx} out of range (max: {num_layers - 1})")
            return 1

    # Load dataset
    print(f"Loading dataset: {args.dataset}/{args.dataset_config}")
    data = load_dataset(args.dataset, args.dataset_config, split="train")
    data = data.filter(lambda x: len(x["text"]) > 100)

    # Initialize collector
    collector = MultiLayerCollector(model, target_layers)

    # Collect data
    all_entropies: list[torch.Tensor] = []
    collected = 0
    
    # Get LM head and final norm to compute per-layer "pseudo-entropy"
    lm_head = model.lm_head
    final_norm = model.model.norm
    
    # Collect entropy for each layer separately
    layer_entropies: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}

    print(f"Collecting data from {len(target_layers)} layers...")
    with torch.no_grad():
        for item in tqdm(data, total=args.samples, desc="Collecting"):
            if collected >= args.samples:
                break

            text = item["text"]
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
            ).to(device)

            _ = model(**inputs)  # Trigger hooks to collect hidden states
            
            # Calculate entropy for each layer at exit
            for layer_idx in target_layers:
                acts = collector.activations[layer_idx][-1]  # [1, seq, hidden]
                # Calculate logits via final_norm + lm_head
                normed = final_norm(acts.to(device))
                layer_logits = lm_head(normed)  # [1, seq, vocab]
                layer_entropy = compute_entropy(layer_logits).detach().cpu()
                layer_entropies[layer_idx].append(layer_entropy.squeeze(0))

            collected += 1
            
            if collected % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    collector.remove_hooks()

    print("\nProcessing and saving data...")

    # Save data for each layer (each layer has its own entropy)
    for layer_idx in target_layers:
        activations = collector.get_activations(layer_idx)
        flat_activations = [x.squeeze(0) for x in activations]
        all_hidden = torch.cat(flat_activations, dim=0)
        
        # Entropy for this layer
        all_layer_entropy = torch.cat(layer_entropies[layer_idx], dim=0)
        
        min_len = min(all_hidden.shape[0], all_layer_entropy.shape[0])
        all_hidden = all_hidden[:min_len]
        entropies = all_layer_entropy[:min_len]

        hidden_path = os.path.join(args.save_dir, f"hidden_states_layer{layer_idx}.pt")
        entropy_path = os.path.join(args.save_dir, f"entropies_layer{layer_idx}.pt")

        torch.save(all_hidden, hidden_path)
        torch.save(entropies, entropy_path)

        size_mb = all_hidden.element_size() * all_hidden.numel() / 1024 / 1024
        print(f"  Layer {layer_idx}: {all_hidden.shape} ({size_mb:.1f} MB), entropy_mean={entropies.mean():.3f}")

    print("\n✅ Data collection complete!")
    print(f"Files saved to: {args.save_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
