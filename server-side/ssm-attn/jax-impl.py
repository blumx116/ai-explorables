import json
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflowjs as tfjs
import torch
from einops import rearrange
from flax.training.common_utils import onehot
from jax import Array
from jax.random import key
from safetensors import safe_open
from transformers import MambaForCausalLM

from patch_mamba import debug, globally_patch_model, show


class MambaMixer(nn.Module):
    intermediate_size: int
    kernel_size: int
    ssm_state_size: int
    dt_rank: int
    hidden_size: int

    @staticmethod
    def SSM(A: Array, deltaB_u: Array, C: Array) -> Array:
        """
        Arguments:
            A: (b, h, s, d)
            deltaB_u: (b, h, s, d)
            C: (b, s, d)
        Returns:
            outputs: (b, h, s)
        """
        C = rearrange(C, "b s d -> b 1 s d")
        b, h, s, d = A.shape
        grouped_ABC: Array = jnp.concatenate([A, deltaB_u, C], axis=1)
        # ABC: (b, (2h + 1), s, d)
        grouped_ABC = rearrange(
            grouped_ABC, "b h s d -> s b h d"
        )  # reorder because scan chops along the leading dim
        ssm_state_initial: Array = jnp.zeros((b, h, d))

        def __SSM_single(state: Array, ABC_concat: Array) -> tuple[Array, Array]:
            """
            Arguments:
                state: (b, h, d)
                ABC_concat: (b, 2h+1, d)
            """
            Ai, deltaB_ui, Ci = jnp.split(ABC_concat, [h, 2 * h], axis=1)
            # Ai: (b, h, d)
            # deltaB_ui: (b, h, d)
            # Ci: (b, 1, d)
            Ci = rearrange(Ci, "b 1 d -> b d 1")

            state = Ai * state + deltaB_ui
            output: Array = state @ Ci  # (b, h, 1)
            return state, output

        final_state, outputs = jax.lax.scan(
            __SSM_single, ssm_state_initial, grouped_ABC
        )
        # final-state: (b, h, d)
        # outputs: [(b, h, 1) x s]
        outputs = jnp.concatenate(outputs, axis=-1)  # (b, h, s)

        return outputs

    @nn.compact
    def __call__(self, x: Array) -> Any:
        # x: (b, s, model_dim)
        projected_states: Array = nn.Dense(
            features=2 * self.intermediate_size, use_bias=False, name="in_proj"
        )(x)
        projected_states = rearrange(projected_states, "b s d -> b d s")
        # projected_states: (b, 2 * self.intermediate_size, s)

        raw_hidden_states, gate = jnp.split(projected_states, 2, axis=1)
        # raw_hidden_states, gate: (b, self.intermediate_size, s)
        raw_hidden_states = rearrange(raw_hidden_states, "b h s -> b s h")
        # nn Conv expects spatial dim last

        hidden_states = raw_hidden_states
        hidden_states: Array = nn.silu(
            nn.Conv(
                features=self.intermediate_size,
                kernel_size=self.kernel_size,
                feature_group_count=self.intermediate_size,
                padding=[(self.kernel_size - 1, 0)],
            )(raw_hidden_states)
        )  # (b, s, d)

        ssm_parameters: Array = nn.Dense(
            features=self.dt_rank + (2 * self.ssm_state_size),
            use_bias=False,
            name="x_proj",
        )(hidden_states)
        # ssm_parameters: (b, s, dt_rank + (2 * d))

        timestep, B, C = jnp.split(
            ssm_parameters,
            [self.dt_rank, self.dt_rank + self.ssm_state_size],
            axis=-1,
        )
        # timestep: (b, s, dt_rank)
        # B, C: (b, s, d)
        discrete_time_step: Array = nn.Dense(
            features=self.intermediate_size, name="dt_proj"
        )(
            timestep
        )  # (b, s, h)
        discrete_time_step = nn.softplus(discrete_time_step)
        discrete_time_step = rearrange(discrete_time_step, "b s h -> b h s 1")

        A_log: Array = self.param(
            "A_log",
            nn.initializers.zeros_init(),
            (self.intermediate_size, self.ssm_state_size),
        )
        A: Array = -jnp.exp(A_log)
        discrete_A: Array = jnp.exp(rearrange(A, "h d -> 1 h 1 d") * discrete_time_step)
        # discrete_A: (batch, intermediate_size, seq_len, ssm_state)

        discrete_B: Array = discrete_time_step * rearrange(B, "b s d -> b 1 s d")
        deltaB_u: Array = discrete_B * rearrange(hidden_states, "b s h -> b h s 1")
        # deltaB_u: (batch, intermediate_size, seq_len, ssm_state)

        scan_outputs: Array = self.SSM(
            discrete_A, deltaB_u, C
        )  # (batch, intermediate_size, seq_len)
        D: Array = self.param(
            "D", nn.initializers.zeros_init(), (self.intermediate_size,)
        )
        scan_outputs += rearrange(hidden_states, "b s h -> b h s") * rearrange(
            D, "h -> 1 h 1"
        )

        scan_outputs *= nn.silu(gate)

        scan_outputs = rearrange(scan_outputs, "b h s -> b s h")

        return nn.Dense(features=self.hidden_size, use_bias=False, name="out_proj")(
            scan_outputs
        )


class MambaBlock(nn.Module):
    intermediate_size: int
    kernel_size: int
    ssm_state_size: int
    dt_rank: int
    hidden_size: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x + MambaMixer(
            self.intermediate_size,
            self.kernel_size,
            self.ssm_state_size,
            self.dt_rank,
            self.hidden_size,
        )(x)


class MambaModel(nn.Module):
    n_layers: int
    vocab_size: int
    intermediate_size: int
    kernel_size: int
    ssm_state_size: int
    dt_rank: int
    hidden_size: int

    @nn.compact
    def __call__(self, x: Array) -> Any:
        x = onehot(x, self.vocab_size)
        embeds: Array = nn.Dense(features=self.hidden_size, use_bias=False)(x)
        hidden_values: Array = nn.Sequential(
            [
                MambaBlock(
                    self.intermediate_size,
                    self.kernel_size,
                    self.ssm_state_size,
                    self.dt_rank,
                    self.hidden_size,
                )
            ]
            * self.n_layers
        )(embeds)
        out_logits: Array = nn.Dense(
            features=self.vocab_size, use_bias=False, name="unembed"
        )(hidden_values)
        return out_logits

    @classmethod
    def from_hf(cls, path: Path) -> tuple["MambaModel", dict]:
        cfg = json.load(open(path / "config.json"))
        n_layers: int = cfg["n_layer"]
        model = cls(
            n_layers=n_layers,
            vocab_size=cfg["vocab_size"],
            intermediate_size=cfg["intermediate_size"],
            ssm_state_size=cfg["state_size"],
            kernel_size=cfg["conv_kernel"],
            dt_rank=cfg["time_step_rank"],
            hidden_size=cfg["hidden_size"],
        )
        state = model.init(key(777), jnp.array([[0]]))
        tensors = safe_open(path / "model.safetensors", framework="jax")

        name_mapping: dict[str, str] = {
            "params/Dense_0/kernel": "backbone.embeddings.weight",
            "params/unembed/kernel": "backbone.embeddings.weight",
        }

        for i in range(n_layers):
            hf_layer_name: str = f"backbone.layers.{i}.mixer"
            jax_layer_name: str = f"params/MambaBlock_{i}/MambaMixer_0"
            updates: dict[str, str] = {
                "A_log": "A_log",
                "D": "D",
                "Conv_0/bias": "conv1d.bias",
                "Conv_0/kernel": "conv1d.weight",
                "dt_proj/bias": "dt_proj.bias",
                "dt_proj/kernel": "dt_proj.weight",
                "in_proj/kernel": "in_proj.weight",
                "out_proj/kernel": "out_proj.weight",
                "x_proj/kernel": "x_proj.weight",
            }

            updates = {
                f"{jax_layer_name}/{jax_param_name}": f"{hf_layer_name}.{hf_param_name}"
                for jax_param_name, hf_param_name in updates.items()
            }

        name_mapping = {**name_mapping, **updates}

        def __load_param(jax_params: dict, jax_path: str, hf_name: str) -> None:
            layers: list[str] = jax_path.split("/")
            nested_layers, var_name = layers[:-1], layers[-1]

            for layer in nested_layers:
                jax_params = jax_params[layer]

            old_tensor: Array = jax_params[var_name]
            jax_shape = old_tensor.shape
            new_tensor: Array = tensors.get_tensor(hf_name)
            hf_shape = new_tensor.shape

            if "conv1d.weight" in hf_name:
                new_tensor = rearrange(new_tensor, "h 1 k -> k 1 h")
            elif "proj.weight" in hf_name:
                new_tensor = new_tensor.transpose((1, 0))
            elif "unembed" in jax_path:
                new_tensor = new_tensor.transpose((1, 0))

            if old_tensor.shape == new_tensor.shape:
                jax_params[var_name] = new_tensor
                print(
                    f"loaded {hf_name}({tuple(hf_shape)})-> {jax_path}({tuple(jax_shape)})"
                )
            else:
                print(
                    f"FAILED {hf_name}({new_tensor.shape}) -> {jax_path}({old_tensor.shape})"
                )

        for param in traverse_tree(state):
            __load_param(state, param, name_mapping[param])

        for e in traverse_tree(state):
            if not e in name_mapping.keys():
                print(e)

        print("-" * 10)

        for name in name_mapping.keys():
            print(name)

        return model, state


def traverse_tree(tree: dict, sep="/") -> list[str]:
    results: list[str] = []
    for key in tree.keys():
        if isinstance(tree[key], dict):  # recur
            children: list[str] = traverse_tree(tree[key], sep)
            for child in children:
                results.append(key + sep + child)
        else:
            results.append(key)
    return results


jax_model, state = MambaModel.from_hf(
    Path(
        "/Users/carterblum/projects/pair/ai-explorables/server-side/ssm-attn/checkpoints/no-norm-passthrough-is-3/10000"
    )
)
x = [[0, 1, 2, 3, 0]]
jax_y = jax_model.apply(state, jnp.array(x))

pt_model = MambaForCausalLM.from_pretrained(
    "/Users/carterblum/projects/pair/ai-explorables/server-side/ssm-attn/checkpoints/no-norm-passthrough-is-3/10000"
)
globally_patch_model(pt_model, remove_norm=True)

with debug():
    print("done")
    pt_y = pt_model(torch.Tensor(x).long())["logits"]

import pdb

pdb.set_trace()
print()
