from jaxtyping import Int
from jaxtyping import Bool
from jaxtyping import Float
import math
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # initialize
        init_std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=init_std, a=-3 * init_std, b=3 * init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return out


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        # initialize
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_norm = torch.sqrt((1 / self.d_model) * einsum(x**2, "... d_model -> ...").unsqueeze(-1) + self.eps)
        result = x * self.weight / rms_norm

        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # weights
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out = self.w1.forward(x)
        silu_w1_out = silu(w1_out)
        w3_out = self.w3.forward(x)

        out = self.w2.forward(silu_w1_out * w3_out)
        return out


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super(RoPE, self).__init__()
        self.theta = theta
        self.d_k = d_k
        assert d_k % 2 == 0, "kq dims should be even"
        freq_range = torch.arange(d_k // 2, device=device)
        freq = 1.0 / (self.theta ** (2 * freq_range / d_k))
        max_pos_consts = torch.arange(max_seq_len, device=device)
        # position-angle pairs: (max_seq_len, d_k // 2)
        # pyrefly: ignore
        pos_angs = einsum(max_pos_consts, freq, "i,j->i j")
        self.register_buffer("cos_cached", pos_angs.cos(), persistent=False)
        self.register_buffer("sin_cached", pos_angs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        cos_fetch = self.cos_cached[token_positions]  # pyrefly: ignore
        sin_fetch = self.sin_cached[token_positions]  # pyrefly: ignore
        rotated_x = self._rotate_half(x)
        # repeat the cos/sin fetches
        cos_fetch_2rep = torch.repeat_interleave(cos_fetch, repeats=2, dim=-1)
        sin_fetch_2rep = torch.repeat_interleave(sin_fetch, repeats=2, dim=-1)
        out = x * cos_fetch_2rep + rotated_x * sin_fetch_2rep
        return out

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        # [x0, x1, x2, x3, x4, x5] -> [-x1, x0, -x3, x2, -x5, x4]
        # by row-major, [[-x1, x0], [-x3, x2], [-x5, x4]].flatten() gives it
        out = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
        return out


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x_max = torch.max(x, dim=i, keepdim=True)[0]
    vals = torch.exp(x - x_max)
    out = vals / vals.sum(dim=i, keepdim=True)
    return out


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... keys d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
) -> Float[torch.Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    scale = math.sqrt(d_k)
    qk_vals = einsum(Q, K, "... q d_k, ... k d_k-> ... q k") / scale
    if mask is not None:
        qk_vals = qk_vals.masked_fill(~mask, float("-inf"))
    softmax_scores = softmax(qk_vals, -1)
    out = einsum(softmax_scores, V, "... q k, ... k d_v-> ... q d_v")
    return out


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(CausalMultiHeadSelfAttention, self).__init__()
        self.d_kv = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        # weights
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        # positional embeddings
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_kv, max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        x: Float[torch.Tensor, " ... seq_len d_model"],
        token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None,
    ) -> Float[torch.Tensor, " ... seq_len d_model"]:
        seq_len = x.size(-2)
        # linear projections
        q_full = self.q_proj.forward(x)
        k_full = self.k_proj.forward(x)
        v_full = self.v_proj.forward(x)
        # split into multiple heads
        q_heads = rearrange(q_full, "... seq_len (n_h d_kv) -> ... n_h seq_len d_kv", n_h=self.num_heads)
        k_heads = rearrange(k_full, "... seq_len (n_h d_kv) -> ... n_h seq_len d_kv", n_h=self.num_heads)
        v_heads = rearrange(v_full, "... seq_len (n_h d_kv) -> ... n_h seq_len d_kv", n_h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            q_heads = self.rope.forward(q_heads, token_positions)
            k_heads = self.rope.forward(k_heads, token_positions)

        # causal masking and attention
        causal_mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask=causal_mask)

        attn_out_combined = rearrange(attn_out, "... n_h seq_len d_kv -> ... seq_len (n_h d_kv)")
        out = self.output_proj.forward(attn_out_combined)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(TransformerBlock, self).__init__()
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self, x: Float[torch.Tensor, " batch seq_length d_model"]
    ) -> Float[torch.Tensor, " batch seq_length d_model"]:
        y = x + self.attn.forward(self.ln1.forward(x))
        out = y + self.ffn.forward(self.ln2.forward(y))
        return out


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(TransformerLM, self).__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[torch.Tensor, " batch seq_length"]) -> Float[torch.Tensor, " batch seq_length vocab_size"]:
        embs = self.token_embeddings.forward(x)
        for block in self.layers:
            embs = block.forward(embs)
        embs = self.ln_final.forward(embs)
        out = self.lm_head.forward(embs)
        return out
