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
