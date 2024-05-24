import math
import os
from dataclasses import dataclass
from typing import Optional

import plotly.graph_objects as go
import torch
from tqdm import tqdm
from transformers import MambaForCausalLM

from patch_mamba import globally_patch_model


@dataclass
class EmbeddingsByTimestep:
    embeddings: torch.Tensor  # (n_tsteps, intermediate_size, v)
    timesteps: list[int]  # (n_tsteps, )


def extract_embeddings(
    checkpoint_path: str,
    feature_name: Optional[str] = None,
    layer: Optional[int] = None,
) -> torch.Tensor:
    if feature_name is None:
        feature_name = "raw_hidden_states"
    if layer is None:
        layer = 0
    model: MambaForCausalLM = MambaForCausalLM.from_pretrained(checkpoint_path)
    globally_patch_model(model, remove_norm=True)
    vocab_size: int = model.config.vocab_size
    print(vocab_size)
    input_tokens: torch.Tensor = torch.arange(0, vocab_size).unsqueeze(0)  # (1, s := v)
    model(input_tokens)
    return getattr(
        model.backbone.layers[layer].mixer, feature_name
    )  # (intermediate_size, s) in the case of 'raw_hidden_states'


def extract_embeddings_over_time(
    checkpoints_path: str,
    feature_name: Optional[str] = None,
    layer: Optional[int] = None,
) -> EmbeddingsByTimestep:
    folders: list[str] = os.listdir(
        checkpoints_path
    )  # each folder is assumed to contain a checkpoint generated with `save_pretrained`
    # and its name is assumed to be an integer representing the timestep
    tsteps: list[int] = sorted(list(map(int, folders)))
    embeddings: list[torch.Tensor] = []
    for tstep in tqdm(tsteps):
        source_folder: str = os.path.join(checkpoints_path, str(tstep))
        embeddings.append(extract_embeddings(source_folder, feature_name, layer))
    return EmbeddingsByTimestep(
        embeddings=torch.concat(embeddings, dim=0), timesteps=tsteps
    )


if __name__ == "__main__":
    data: EmbeddingsByTimestep = extract_embeddings_over_time(
        checkpoints_path="/Users/carterblum/projects/pair/ai-explorables/server-side/ssm-attn/checkpoints/no-norm-passthrough-is-3",
    )

    json_data = {
        "embeddings": data.embeddings.detach().numpy().tolist(),
        "timesteps": data.timesteps,
    }

    import json

    with open("embeddings_over_time.json", "w") as f:
        json.dump(json_data, f)

    xdata: torch.Tensor = data.embeddings[:, 0, :]
    ydata: torch.Tensor = data.embeddings[:, 1, :]
    zdata: torch.Tensor = data.embeddings[:, 2, :]

    tokenColorMap = {0: "#FAEDCB", 1: "#C9E4DE", 2: "#DBCDF0", 3: "#FD8A8A"}

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xdata[-1, :],
                y=ydata[-1, :],
                z=zdata[-1, :],
                mode="markers",
                marker=dict(color=list(tokenColorMap.values()), size=25),
            )
        ]
    )

    background_kwargs = dict(
        backgroundcolor="#000225",  # dark navy
        gridcolor="#A9A9A9",  # dark gray
        zerolinecolor="white",
        showbackground=True,
    )

    def make_3d_axis_config(data: torch.Tensor) -> dict:
        background_kwargs = dict(
            backgroundcolor="#000225",  # dark navy
            gridcolor="#A9A9A9",  # dark gray
            zerolinecolor="white",
            showbackground=True,
        )
        dmin = data.min() - 0.5
        dmax = data.max() + 0.5

        return dict(
            range=[dmin, dmax],
            tickmode="linear",
            dtick=1,
            tick0=math.ceil(dmin),
            autorange=False,
            **background_kwargs,
        )

    fig.update_layout(
        scene=dict(
            xaxis=make_3d_axis_config(xdata),
            yaxis=make_3d_axis_config(ydata),
            zaxis=make_3d_axis_config(zdata),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        scene_camera=dict(eye=dict(x=-3, y=0, z=0)),
    )

    frames: list[go.Frame] = [
        go.Frame(
            data=go.Scatter3d(
                x=xdata[t, :],
                y=ydata[t, :],
                z=zdata[t, :],
                marker=dict(color=list(tokenColorMap.values())),
                name=f"Frame {t}",
            )
        )
        for t in range(len(data.timesteps))
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    redraw=True, fromcurrent=False, mode="immediate"
                                )
                            ),
                        ],
                    )
                ],
            )
        ]
    )

    fig.update(frames=frames)

    fig.show()
