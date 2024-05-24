import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, MambaConfig
from tqdm import tqdm
from constants import VOCAB_SIZE, WEIGHT_DECAY, N_EPOCHS


cfg = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
cfg.d_inner = 8
cfg.state_size = 2
cfg.vocab_size = VOCAB_SIZE
cfg.n_layer =  1
cfg.num_hidden_layers = cfg.n_layer
cfg.d_model = cfg.d_inner // 2
cfg.hidden_size = cfg.d_inner // 2
cfg.intermediate_size = 2
cfg.time_step_rank = 1
device = torch.device("cpu")
model = AutoModelForCausalLM.from_config(cfg)

def random_example(length: int, default_pass_through: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
  xs: torch.Tensor = torch.randint(0, VOCAB_SIZE-1, size=(length,))
  first_appearance: int = torch.randint(0, length-2, size=tuple()).item()
  second_appearance: int = torch.randint(first_appearance + 2, length,size=tuple()).item()

  xs[first_appearance] = VOCAB_SIZE -1
  xs[second_appearance] = VOCAB_SIZE -1
  if default_pass_through:
    ys = xs.clone().int()
  else:
    ys = torch.full_like(xs, -100).int()
  ys[second_appearance] = xs[first_appearance + 1]

  return xs, ys

def make_dataset(count: int, length: int, default_pass_through: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
  xs, ys = zip(*[random_example(length, default_pass_through) for _ in range(count)])
  return torch.stack(xs).long(), torch.stack(ys).long()

def train_model(model: nn.Module, xs: torch.Tensor, ys: torch.Tensor, xtest: torch.Tensor, ytest: torch.Tensor) -> nn.Module:
  model = model.to(device)
  xs = xs.to(device)
  ys = ys.to(device)
  xtest = xtest.to(device)
  ytest = ytest.to(device)
  optimizer = optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)

  for epoch in tqdm(range(N_EPOCHS)):
    optimizer.zero_grad()

    logits = model(xs)['logits']
    loss = compute_loss(logits, ys)

    loss.backward()
    optimizer.step()

    if epoch % 250 == 0:
      print("\ntrain loss:", loss.item())
      validation_printing(model, xtest, ytest)

def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
  # writing my own loss function b/c theirs was specifically for language modeling
  # and I think this is just cleaner to read than a lot of inscrutable permutations
  # to labels before passing them in
  return nn.CrossEntropyLoss()(torch.transpose(logits, -1, -2), labels)


def validation_printing(model: nn.Module, xtest: torch.Tensor, ytest: torch.Tensor):
  with torch.no_grad(): # should be unnecessary, but just in case
        testlogits = model(xtest)['logits']
        loss = compute_loss(testlogits, ytest)
        print("\nvalidation loss", loss.item())
        # print(f"{xtest=}")
        # print(f"{ytest=}")
        pred_tokens: torch.Tensor = testlogits.argmax(dim=-1) # (b, s)
        answers: torch.Tensor = ytest.max(dim=-1).values
        plt.imshow(F.one_hot(answers, VOCAB_SIZE).cpu())
        plt.show()
        pred_tokens: torch.Tensor = pred_tokens.gather(-1, ytest.argmax(dim=-1).unsqueeze(-1)) # (b, )
        # print(f"{pred_tokens=}") # (b, s, d)
        relevant_logits: torch.Tensor = testlogits.detach().gather(1, ytest.argmax(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, VOCAB_SIZE)).squeeze(1) # (b, v)
        scores: torch.Tensor = torch.softmax(relevant_logits, dim=-1)
        plt.imshow(scores.cpu(), vmin=0, vmax=1)
        plt.title("Logits on predictions")
        plt.show()

