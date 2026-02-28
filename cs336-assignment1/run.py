"""
Text generation / decoding for CS336 Assignment 1 — Section 6.

Loads a trained TransformerLM checkpoint and generates text from a prompt.

Usage (with tokenizer):
    python run.py \
        --checkpoint checkpoints/step_0010000_final.pt \
        --tokenizer tokenizer.pt \
        --prompt "Once upon a time" \
        --max_tokens 200 --temperature 0.8 --top_p 0.95

Usage (raw token IDs, no tokenizer):
    python run.py \
        --checkpoint checkpoints/step_0010000_final.pt \
        --prompt_ids "1 234 56 789" \
        --max_tokens 100
"""

import argparse
import torch

from cs336_basics.model import TransformerLM


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a trained TransformerLM")

    # Checkpoint
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")

    # Model architecture (must match the checkpoint)
    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--rope_theta", type=float, default=10_000.0)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])

    # Tokenizer (optional — if omitted, use --prompt_ids)
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to a saved Tokenizer object (torch.save'd). If omitted, use --prompt_ids instead.",
    )
    p.add_argument("--eos_token", type=str, default="<|endoftext|>", help="End-of-sequence special token string")

    # Prompt
    p.add_argument("--prompt", type=str, default=None, help="Text prompt (requires --tokenizer)")
    p.add_argument("--prompt_ids", type=str, default=None, help="Space-separated token IDs as prompt, e.g. '1 234 56'")

    # Generation hyperparameters
    p.add_argument("--max_tokens", type=int, default=200, help="Maximum number of new tokens to generate")
    p.add_argument(
        "--temperature", type=float, default=1.0, help="Softmax temperature τ. Lower = more peaked, 0 → greedy"
    )
    p.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling threshold p (1.0 = disabled)")

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Zero out logits outside the nucleus (smallest set of tokens summing to >= p).
    Returns filtered logits (not probabilities).
    """
    if p >= 1.0:
        return logits

    probs = torch.softmax(logits, dim=-1)
    # Sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Shift by one: the token that pushes cumsum over p is still included
    remove = (cumulative - sorted_probs) >= p
    sorted_probs[remove] = 0.0
    # Scatter back to original order
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_indices, sorted_probs)
    return filtered  # unnormalized; torch.multinomial re-normalizes


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    context_length: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    eos_id: int | None,
    device: str,
) -> list[int]:
    """
    Autoregressively generate tokens from prompt_ids.
    Returns only the newly generated token IDs (not including the prompt).
    """
    model.eval()
    ids = list(prompt_ids)
    generated = []

    for _ in range(max_tokens):
        # Truncate to context window from the right
        window = ids[-context_length:]
        input_ids = torch.tensor([window], dtype=torch.long, device=device)  # (1, seq_len)

        logits = model(input_ids)  # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]  # (vocab_size,) — last position

        # Temperature scaling
        if temperature > 0:
            next_logits = next_logits / temperature
            probs = top_p_filter(next_logits, top_p)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            # Greedy
            next_id = int(next_logits.argmax().item())

        ids.append(next_id)
        generated.append(next_id)

        if eos_id is not None and next_id == eos_id:
            break

    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    # --- Load model ---
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

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    iteration = ckpt.get("iteration", "?")
    print(f"Loaded checkpoint (step {iteration})")

    # --- Tokenizer ---
    tokenizer = None
    eos_id = None

    if args.tokenizer:
        tokenizer = torch.load(args.tokenizer, weights_only=False)
        # Find EOS token ID
        if hasattr(tokenizer, "vocab"):
            for tok_id, tok_bytes in tokenizer.vocab.items():
                try:
                    if tok_bytes.decode("utf-8") == args.eos_token:
                        eos_id = tok_id
                        break
                except UnicodeDecodeError:
                    pass
        print(f"Tokenizer loaded. EOS id: {eos_id}")

    # --- Prompt ---
    if args.prompt is not None:
        if tokenizer is None:
            raise ValueError("--prompt requires --tokenizer. Use --prompt_ids for raw IDs.")
        prompt_ids = tokenizer.encode(args.prompt)
        prompt_text = args.prompt
    elif args.prompt_ids is not None:
        prompt_ids = [int(t) for t in args.prompt_ids.split()]
        prompt_text = f"[token ids: {prompt_ids}]"
    else:
        raise ValueError("Provide either --prompt (with --tokenizer) or --prompt_ids.")

    print(f"\nPrompt: {prompt_text}")
    print(f"Prompt length: {len(prompt_ids)} tokens")
    print(f"Generating up to {args.max_tokens} tokens (temp={args.temperature}, top_p={args.top_p})...\n")
    print("-" * 60)

    # --- Generate ---
    new_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        context_length=args.context_length,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id,
        device=args.device,
    )

    # --- Decode & print ---
    if tokenizer is not None:
        full_ids = prompt_ids + new_ids
        print(tokenizer.decode(full_ids))
    else:
        print("Generated token IDs:", new_ids)

    print("\n" + "-" * 60)
    print(f"Generated {len(new_ids)} tokens.")


if __name__ == "__main__":
    main()
