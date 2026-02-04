import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    """
    预先计算旋转位置编码所需的cos和sin矩阵
    YaRN(Yet another ROPE extensioN): 推理时动态扩展模型的上下文窗口(Extrapolation)

    torch.arange(0, dim, 2) 从 0 到 dim, 每隔 2 取一个数
        [0: dim // 2] 切片操作,强制选


    """
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


"""
绝对位置编码: 构建一维向量,通过加法对嵌入向量增加位置信息
旋转位置编码: 构建分组旋转矩阵,通过矩阵乘对特征分量嵌入位置信息

P(m,i): 位置编号为m,第i个角度对应的分组向量,d是注意力特征维度
    P(m,i) = [sin(m*theta_i), cos(m*theta_i)]

theta: 定义角度
    (1) 决定旋转的快慢: theta_i = 1 / (1 / b^2((i - 1)/d)))
        i in [1, d/2]: rope 不是对所有维度统一旋转,而是两两一组,把它们看作一个个二维平面上的点,i 就是这些组的编号

Q1: 为什么高维度的慢旋转能捕捉长距离?
A1: 避免角度重合
    (1) 相位偏移: 维度低,旋转快,如果两个词距离远,指针可能转了几十圈回到原点,模型分不清两个词是距离1,101 or 201
    (2) 位置与角度对应唯一: 旋转慢,即使两个词距离1000,指针可能才转30度
"""
