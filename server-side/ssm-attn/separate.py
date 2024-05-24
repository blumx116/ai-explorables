from sklearn import svm
import torch
from extract_embeddings import extract_embeddings, EmbeddingsByTimestep

x: torch.Tensor= extract_embeddings("/Users/carterblum/projects/pair/ai-explorables/server-side/ssm-attn/checkpoints/no-norm-passthrough-is-3/10000")
y = [0, 0, 0, 1]

model = svm.SVC(kernel='linear')
model.fit(x.squeeze(0).transpose(1, 0), y)
print(model.coef_)

breakpoint()
model