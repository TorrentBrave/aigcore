"""
Q1: 为什么RMSNorm和LayerNorm都在token特征维度上操作而非跨batch?
A1: BatchNorm是在处理图像数据时常用的归一化方式
    图像数据通常有强烈的空间相关性,即相邻的像素通常会有相似的值或模式。因此,图像的像素特征在一个batch中通常有相似的分布,这使得在整个batch上做归一化是合理的。BatchNorm通过计算每个特征(比如每个通道)的均值和方差，能有效地减轻这些空间相关性带来的影响，并保证训练时每一层的输入保持一定的分布,从而加速收敛。
    而在NLP任务中,每个token通常是一个具有特定语义和上下文信息的单位,比如每个token代表一个词。每个token的特征是通过模型的embedding层或Transformer层计算得到的,并包含了该token的语义信息。不同token的语义内容不同,所以它们的特征应该独立地进行归一化处理。
    如果归一化操作发生在batch维度上,会导致不考虑每个token的独立性。用于归一化的数据来自不同的batch,包含不同的token内容和信息,如果跨batch进行标准化,会丢失token间的独立性,使得token之间存在耦合关系,比如一些padding token并没有实际意义,但是被加入了归一化计算,进而影响模型的学习效果

Q2: 为什么使用RMSNorm而不是LayerNorm?
A2: (1) 计算过程比更简单,因为它不涉及均值的计算,并且减少了一个可学习参数
        LayerNorm在归一化时需要计算每个token的均值和方差,并使用它们来标准化输入。
        而RMSNorm只需要计算特征的平方和,减少了计算复杂度和内存消耗
    (2) 处理大型模型时,输入的特征维度可能非常大,计算均值和方差的开销相对较大。RMSNorm去除了均值计算,因此可以节省计算资源,特别是在高维数据中，计算效率更高
    (3) 在各种场景中实验发现,使用RMSNorm能够减少约7%~64%的计算时间

Q3: token 独立标准化的作用
A3: (1) 变长序列: 在推理时，句子长度是动态的。如果 normalization 依赖于其他 token(Batch 维度),那么当句子变长时,均值和方差会剧烈波动。
    (2) 并行计算: 独立化让每个 token 的归一化可以并行完成，不需要等待其他 Batch 的统计结果。
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    x.shape: [batch_size, seq_length, embedding_dim]
    gamma: scale parameter which can learn named weight for each token

    torch.rsqrt: x.pow(2).mean(-1, keepdim=True) + self.eps 的平方根倒数
    直接调用 rsqrt 比 先 sqrt 再 1 / 在 GPU 上更高效
    keepdim: eg [1, 2, 4] -> [1, 2, 1] 而不是 [1, 2]

    Llama 系列模型标配的归一化层,比标准的LayerNorm少了减去均值的步骤,计算更简单,训练更稳定
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / (var + self.eps).sqrt() + self.bias
