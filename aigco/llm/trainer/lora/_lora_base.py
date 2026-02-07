import torch
from torch import nn


class LoRA(nn.Module):
    """
    output = W_0x + BA(X) * alpha/rank

    1. matrix B (All Zeros):
        ensures the initial state of the LoRA path is BAx=0
    2. matrix A (Random/Gaussian):
        breaks symmetry
        If both A and B were zero, the gradients for all neurons would be identical.
        Randomizing A ensures that different neurons can learn different features once training begins and the weights start to update
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """
    explicit binding
    """
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and module.weight.shape[0] == module.weight.shape[1]
        ):
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(
                model.device
            )
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    raw_model = getattr(model, "_orig_mod", model)
        If compiled: It grabs the hidden original model (_orig_mod).
        If not compiled: It just uses the model as-is
    """
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {
                f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)
