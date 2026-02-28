"""
Training loop for CS336 Assignment 1

Usage:
    python main.py \
        --train_path data/train.npy \
        --val_path data/val.npy \
        --checkpoint_dir checkpoints/

Run `python main.py --help` for full argument list.
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    gradient_clip,
    load_checkpoint,
    lr_cosine_schedule,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer LM")

    # Data
    p.add_argument("--train_path", type=str, required=True, help="Path to tokenized training data (.npy)")
    p.add_argument("--val_path", type=str, default=None, help="Path to tokenized validation data (.npy)")

    # Model
    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--rope_theta", type=float, default=10_000.0)

    # Optimizer
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Schedule
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--max_iters", type=int, default=10_000)
    p.add_argument("--cosine_cycle_iters", type=int, default=None, help="Defaults to max_iters if not set")

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])

    # Logging & checkpointing
    p.add_argument("--log_interval", type=int, default=50, help="Log training loss every N steps")
    p.add_argument("--val_interval", type=int, default=500, help="Evaluate validation loss every N steps")
    p.add_argument("--val_batches", type=int, default=20, help="Number of batches to average for validation loss")
    p.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints")
    p.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every N steps")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_dataset(path: str) -> np.ndarray:
    """Load tokenized dataset with memory mapping for large files."""
    return np.load(path, mmap_mode="r")


@torch.no_grad()
def eval_val_loss(model, val_data, batch_size, context_length, device, num_batches):
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        # logits: (batch, seq_len, vocab_size) — flatten for cross_entropy
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # --- Data ---
    train_data = load_dataset(args.train_path)
    val_data = load_dataset(args.val_path) if args.val_path else None
    print(f"Train tokens: {len(train_data):,}")
    if val_data is not None:
        print(f"Val tokens:   {len(val_data):,}")

    # --- Model ---
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
        dtype=dtype,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # --- Optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # --- Resume from checkpoint ---
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from checkpoint at step {start_iter}")

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Training loop ---
    model.train()
    t0 = time.time()

    for step in range(start_iter, args.max_iters):
        # Update learning rate
        lr = lr_cosine_schedule(
            it=step,
            max_learning_rate=args.lr_max,
            min_learning_rate=args.lr_min,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Forward + backward
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        gradient_clip(model.parameters(), args.grad_clip)
        optimizer.step()

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            tokens_per_sec = (args.log_interval * args.batch_size * args.context_length) / max(elapsed, 1e-9)
            print(
                f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{tokens_per_sec:.0f} tok/s | {elapsed:.1f}s elapsed"
            )
            t0 = time.time()

        # Validation
        if val_data is not None and step % args.val_interval == 0 and step > 0:
            val_loss = eval_val_loss(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                args.device,
                args.val_batches,
            )
            val_ppl = math.exp(val_loss)
            print(f"  >> val loss {val_loss:.4f} | val ppl {val_ppl:.2f}")

        # Checkpointing
        if args.checkpoint_dir and step % args.checkpoint_interval == 0 and step > 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step:07d}.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"  >> checkpoint saved to {ckpt_path}")

    # Final checkpoint
    if args.checkpoint_dir:
        ckpt_path = os.path.join(args.checkpoint_dir, f"step_{args.max_iters:07d}_final.pt")
        save_checkpoint(model, optimizer, args.max_iters, ckpt_path)
        print(f"Final checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
