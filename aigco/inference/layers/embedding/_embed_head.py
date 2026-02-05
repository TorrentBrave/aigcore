import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from aigco.inference.utils import get_context

"""
它们就像一张纸的正反面

你可以把那个巨大的词表矩阵(Vocab Matrix)想象成一张“翻译对照表”:

    Embedding(入口/脚）：是查表。

        逻辑:给你一个编号(ID),你去矩阵里把对应的那一行向量“拿出来”。

        操作:Lookup(查表)。

    Head(出口/头）：是投影。

        逻辑：给你一个算好的向量，你去矩阵里比对，看看它和哪一行的向量最像。

        操作: Linear(矩阵乘法)
"""


class VocabParallelEmbedding(nn.Module):
    """
    Token embedding save memory

    Embedding: output=W[input_id]
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    save flots

    Head: logits = XW^T

    两个类共用权重
    (1) 省显存：一个 15 万词表、4096 维的模型，一个矩阵就要占约 150000*4096*2 bytes ≈ 1.2GB。共用能省出一块大肥肉。
    (2) 语义一致性: 如果“猫”在入口处的向量是 V,那么模型在出口处想表达“猫”时,吐出的向量也应该接近 V。
    (3) 训练更稳: 共用权重相当于给模型加了约束，让词向量在训练时能同时受到输入和输出的监督
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
