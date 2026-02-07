import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import aigco


def test_self_attention_shapes():
    """验证 Self-Attention 各阶段张量的形状是否正确"""
    torch.manual_seed(42)

    # 1. 模拟输入数据
    sentence_len = 6
    embedding_dim = 16
    d_k, d_q, d_v = 24, 24, 28

    dummy_input = torch.randn(sentence_len, embedding_dim)

    # 2. 手动计算逻辑 (来自你的原代码)
    W_K = nn.Parameter(torch.rand(embedding_dim, d_k))  # fllow gradient
    W_Q = nn.Parameter(torch.rand(embedding_dim, d_q))
    W_V = nn.Parameter(torch.rand(embedding_dim, d_v))

    K = dummy_input @ W_K
    Q = dummy_input @ W_Q
    V = dummy_input @ W_V

    # 断言投影形状
    assert K.shape == (sentence_len, d_k)
    assert Q.shape == (sentence_len, d_q)
    assert V.shape == (sentence_len, d_v)

    # 3. 计算 Attention
    attention_scores = (K @ Q.T) / math.sqrt(d_k)
    attention_weight = F.softmax(attention_scores, dim=-1)
    context_vectors = attention_weight @ V

    # 断言结果形状
    assert attention_weight.shape == (sentence_len, sentence_len)
    assert context_vectors.shape == (sentence_len, d_v)


def test_aigco_self_attention_layer():
    """验证 aigco 库中封装的 SelfAttention 层"""
    d_in, d_k, d_q, d_v = 16, 24, 24, 28
    model = aigco.llm.layers.SelfAttention(d_in, d_k, d_q, d_v)

    # 测试完整句子输入
    sample_input = torch.randn(6, d_in)
    output = model(sample_input)

    assert output.shape == (6, d_v)
    assert not torch.isnan(output).any(), "输出包含 NaN"


def test_single_token_input():
    """验证单词输入是否也能正常工作"""
    d_in, d_k, d_q, d_v = 16, 24, 24, 28
    model = aigco.llm.layers.SelfAttention(d_in, d_k, d_q, d_v)

    token_input = torch.randn(1, d_in)
    output = model(token_input)

    assert output.shape == (1, d_v)
