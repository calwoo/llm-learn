import math
import torch
import torch.nn as nn
from einops import einsum


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
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_norm = torch.sqrt((1 / self.d_model) * einsum(x**2, "... d_model -> ...").unsqueeze(-1) + self.eps)
        result = x * self.gain / rms_norm

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
