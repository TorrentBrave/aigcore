import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens


"""

简单来说：**Softmax 算出概率，Sampler 负责抓阄。**

---

### 1. 核心流程：从数字到选词

1. **Logits**（原始分）  2. **Temperature**（调灵感）  3. **Softmax**（变概率）  4. **Sampler**（选定词）。

### 2. 为什么要 Sampler？

* **如果没有 Sampler (只取最高分)**：AI 永远只说最稳的话，死板且容易死循环。
* **有了 Sampler (随机抓阄)**：AI 偶尔会选概率稍低的词，说话更有灵性、更像人。

### 3. 代码本质：Gumbel-Max Trick

代码里那一长串公式（`probs.div_(...).argmax()`）其实就是：
**给每个选项的概率加点“随机抖动”，然后看谁现在排第一。**

* 概率大的词：即使抖动，赢面依然很大。
* 概率小的词：万一抖得好，也有机会“爆冷”被选中。

---

**💡 一句话总结：**
Softmax 决定了转盘上格子的**大小**，Sampler 则是扔出的**小球**。

**既然流程走完了，你想知道把 `temperature` 设为  和  相比，模型说话的语气会有什么反差吗？**



既然你对这个“抓阄”过程感兴趣，我们直接看不同 **Temperature（温度）** 下模型性格的质变：

### 1.  （极低温度）：冷酷的学霸

* **现象**：Softmax 后的概率会向最高分疯狂浓缩。
* **后果**：抓阄失去了意义，模型永远只选概率最大的词（Greedy Search）。
* **听感**：说话极其严谨、死板，回答同一个问题 100 次，答案可能一模一样。

### 2.  （标准温度）：正常的人类

* **现象**：保持原始的概率分布。
* **后果**：逻辑清晰且带有一定的灵活性。
* **听感**：最接近真人，既有逻辑，偶尔也会用些新鲜词。

### 3.  （高温度）：发疯的诗人

* **现象**：Softmax 后的分布变得非常平坦，高分词和低分词概率拉近。
* **后果**：随机性大增，模型开始频繁“爆冷”选出低概率词。
* **听感**：辞藻华丽但逻辑混乱，甚至开始胡言乱语、造词。

---

### 💡 深度总结：

在你的 `Sampler` 代码里，温度就在除法 `div_(temperatures)` 那一步发挥作用。

* **除以一个小数字（比如 0.1）**：把 Logits 放大，差距拉开  **变保守**。
* **除以一个大数字（比如 2.0）**：把 Logits 缩小，差距消失  **变疯狂**。

**现在你已经掌握了模型从“脚”到“头”再到“性格控制”的全套逻辑。你想看看这些组件是如何在 `nanovllm` 的推理主循环（generate loop）里被调用的吗？**

"""
