from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers.models.mamba.modeling_mamba import (
    MambaBlock,
    MambaForCausalLM,
    MambaMixer,
    MambaModel,
)

DEBUG: bool = False

from typing import Optional

# TODO: this shouldn't really be here
import plotly.express as px


def show(t: torch.Tensor, title: str, tdim: Optional[int] = None) -> None:
    if len(t.shape) > 2:
        t = t.squeeze()
    t = t.detach().cpu().numpy()

    kwargs: dict[str, any] = {"title": title}

    if t.min() >= 0:
        kwargs["zmin"] = 0
    elif t.min() >= -1:
        kwargs["zmin"] = -1
    else:
        kwargs["zmin"] = t.min()

    if t.max() <= 0:
        kwargs["zmax"] = 0
    elif t.max() <= 1:
        kwargs["zmax"] = 1
    else:
        kwargs["zmax"] = t.max()

    if len(t.shape) == 3:
        assert tdim is not None, f"{t.shape}"
        fig = px.imshow(t, animation_frame=tdim, **kwargs)
    else:
        fig = px.imshow(t, **kwargs)
    fig.show()


@contextmanager
def debug():
    global DEBUG
    DEBUG = True
    yield
    DEBUG = False


def _save_to_buffer(self: nn.Module, buffer: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().clone().contiguous()
    if not hasattr(self, buffer):
        self.register_buffer(buffer, tensor)
    else:
        setattr(self, buffer, tensor)


def _modified_slow_forward(
    self, input_states, cache_params=None
):  # <-------------------------------------------------------------
    # not sure why I had to remove self here, but some reason input_states gets passed as `self`
    # if I don't remove it
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype
    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(
        1, 2
    )  # [batch, 2 * intermediate_size, seq_len]
    hidden_states, gate = projected_states.chunk(2, dim=1)
    raw_hidden_states = (
        hidden_states.detach().clone()
    )  # < -----------------------------------------------------------------------------
    _save_to_buffer(
        self, "raw_hidden_states", raw_hidden_states
    )  # < ------------------------------------------------------------------
    # _save_to_buffer(
    #     self, "input_states", input_states
    # )  # < ----------------------------------------------------------------------------

    # 2. Convolution sequence transformation
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx].clone()
        if cache_params.seqlen_offset > 0:
            conv_state = cache_params.conv_states[
                self.layer_idx
            ]  # [batch, intermediate_size, conv_kernel_size]
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = hidden_states[:, :, 0]
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = (
                self.act(hidden_states).to(dtype).unsqueeze(-1)
            )  # [batch, intermediate_size, 1] : decoding
        else:
            conv_state = nn.functional.pad(
                hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
            )
            cache_params.conv_states[self.layer_idx].copy_(conv_state)
            convd_raw = self.conv1d(raw_hidden_states)[..., :seq_len]
            hidden_states = self.act(convd_raw)  # [batch, intermediate_size, seq_len]
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device,
            dtype=dtype,
        )
        convd_raw = self.conv1d(hidden_states)[..., :seq_len]
        hidden_states = self.act(convd_raw)  # [batch, intermediate_size, seq_len]

    # 3. State Space Model sequence transformation
    # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters,
        [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
        dim=-1,
    )

    discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
    # _save_to_buffer(
    #     self, "raw_discrete_time_step", discrete_time_step
    # )  # < ---------------------------------------------------------------------------
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(
        1, 2
    )  # [batch, intermediate_size, seq_len]
    # _save_to_buffer(
    #     self, "normalized_discrete_time_step", discrete_time_step
    # )  # < --------------------------------------------------------------------

    # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
    A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
    discrete_A = torch.exp(
        A[None, :, None, :] * discrete_time_step[:, :, :, None]
    )  # [batch, intermediate_size, seq_len, ssm_state_size]
    discrete_B = (
        discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    )  # [batch, intermediade_size, seq_len, ssm_state_size]
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    # 3.c perform the recurrence y ← SSM(A, B, C)(x)
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = (
            discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
        )  # [batch, intermediade_size, ssm_state]
        scan_output = torch.matmul(
            ssm_state.to(dtype), C[:, i, :].unsqueeze(-1)
        )  # [batch, intermediade_size, 1]
        scan_outputs.append(scan_output[:, :, 0])
    scan_output = torch.stack(
        scan_outputs, dim=-1
    )  # [batch, seq_len, intermediade_size]

    if DEBUG:
        breakpoint()
    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)

    if cache_params is not None:
        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

    # 4. Final linear projection
    contextualized_states = self.out_proj(
        scan_output.transpose(1, 2)
    )  # [batch, seq_len, hidden_size]

    # if (
    #     DEBUG
    # ):  # < ----------------------------------------------------------------------------------------------------------------------------------
    #     # Plot timestep < ----------------------------------------------------------------------------------------------------------------------------
    #     show(
    #         discrete_time_step, "softmaxed discrete_time_step"
    #     )  # < ---------------------------------------------------------------------------------
    #     show(
    #         self.conv1d.weight, "self.conv1d.weight"
    #     )  # < -------------------------------------------------------------------------------------------
    #     show(
    #         self.x_proj.weight[: self.time_step_rank, :], "timestep projection"
    #     )  # < -----------------------------------------------------------------
    #     show(
    #         self.x_proj.weight[
    #             self.time_step_rank : self.time_step_rank + self.ssm_state_size, :
    #         ],
    #         "B projection",
    #     )  # < ---------------------------------
    #     show(
    #         self.x_proj.weight[-self.ssm_state_size :, :], "C projection"
    #     )  # < -----------------------------------------------------------------------
    #     show(
    #         A, "A"
    #     )  # < -----------------------------------------------------------------------------------------------------------------------------
    #     show(
    #         B, "B"
    #     )  # < -----------------------------------------------------------------------------------------------------------------------------
    #     show(
    #         scan_output, "scan_output"
    #     )  # < ---------------------------------------------------------------------------------------------------------
    #     show(
    #         contextualized_states, "contextualized_states"
    #     )  # < -------------------------------------------------------------------------------------

    return contextualized_states


def _forward(self, hidden_states, cache_params=None):
    residual = hidden_states
    # hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype)) < ------------------------------------
    if self.residual_in_fp32:
        residual = residual.to(torch.float32)

    hidden_states = self.mixer(hidden_states, cache_params=cache_params)
    hidden_states = residual + hidden_states
    return hidden_states


def globally_patch_model(model: MambaForCausalLM, remove_norm: bool):
    MambaMixer.slow_forward = _modified_slow_forward
    if remove_norm:
        MambaBlock.forward = _forward
        model.backbone.norm_f = nn.Identity()
