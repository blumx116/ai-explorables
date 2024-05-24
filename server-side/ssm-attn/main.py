import matplotlib.pyplot as plt
import torch
from transformers import MambaForCausalLM
from transformers.models.mamba.modeling_mamba import MambaCache

from constants import SEQ_LEN
from patch_mamba import debug, globally_patch_model, show
from train_mamba import make_dataset, model, validation_printing

# xdbg = torch.Tensor([[0, 1, 2, 3,]]).long()
#                     0  1  2  3  4  5  6  7  8  9
xdbg = torch.Tensor([[3, 1, 0, 0, 2, 2, 1, 3, 2]]).long()
ydbg = xdbg.clone()
model = MambaForCausalLM.from_pretrained(
    "/Users/carterblum/projects/pair/ai-explorables/server-side/ssm-attn/checkpoints/no-norm-passthrough-is-3/10000"
)
globally_patch_model(model, remove_norm=True)
# validation_printing(model, xs, ys)
# show(xdbg, "Input Sequence")
# show(model(xdbg)["logits"].argmax(dim=-1), "Predictions")
# show(ydbg, "labels")

print(model(xdbg)["logits"])
