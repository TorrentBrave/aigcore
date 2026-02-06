from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    gamma = clamp((r - βslow) / (βfast - βslow), 0, 1)
    YaRN 会根据每个维度的波长与原始窗口(4096)的比值 r 来分类：

    高频区 (r>βfast):
        这些维度的波长非常短，处理的是极其精细的邻近位置关系。
        策略：完全不缩放。保持 100% 的原始分辨率，防止模型“散光”。

    低频区 (r<βslow):
        这些维度的波长很长，处理的是宏观的、远距离的语义联系。
        策略：完全线性缩放。将它们除以 scaling_factor,让长文本挤进视野。

    过渡区 (βslow<r<βfast):
        处于中间地带。
        策略：平滑插值。这就是你代码里计算 gamma 的地方。它让缩放比例从 0 到 1 丝滑过渡，避免在不同维度之间产生剧烈的数学断层

    βfast and slow像是一个频率过滤器
    βfast 确保细节不被缩放
    βslow 确保大局被正确缩放
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        t = t / scaling_factor

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | tuple | None = None,
):
    # assert rope_scaling is None
    # 1. 解析 rope_scaling
    scaling_factor = 1.0
    if rope_scaling is not None:
        # 如果是元组（为了绕过哈希报错），转回字典
        if isinstance(rope_scaling, tuple):
            scaling_dict = dict(rope_scaling)

        # Qwen3 通常在配置里叫 "factor"
        # 如果没有配置 factor，默认为 1.0
        scaling_factor = scaling_dict.get("factor", 1.0)
    rotary_emb = RotaryEmbedding(
        head_size, rotary_dim, max_position, base, scaling_factor
    )
    return rotary_emb


"""
它只记录最近一次的结果。因为在一个模型里，虽然有几十个 Layer,但它们的 RoPE 参数(维度、base、scaling)通常是完全一样的
如果删掉它：

你的 Qwen3 模型有几十个 Qwen3DecoderLayer(比如 24 层或 32 层）。
    每一层初始化时，都会跑一遍 get_rope。
    get_rope 内部会运行 RotaryEmbedding 的 __init__。
    __init__ 里面有一堆数学运算(torch.arange, pow, einsum, cos, sin)
    结果：你会重复计算 32 遍一模一样的 Cos/Sin 矩阵，浪费了 32 倍的计算时间。
如果保留它：
    第一层初始化：老老实实计算，存入缓存。
    第二层到第三十二层：瞬间完成。秘书直接从抽屉里拿出第一层算好的结果复用。
3. 为什么它让你刚才那么痛苦？
因为它有一个副作用：它必须给你的参数“拍照片”（生成哈希值）来当索引。
    数字、字符串、元组：长得结实，可以拍照。
    字典、列表：长得不稳定（随时会变），拍不了照，所以报错。
这就是为什么我们之前非要费劲把 dict 转成 tuple
"""
