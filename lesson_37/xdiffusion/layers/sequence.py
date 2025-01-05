"""Flexible isotropic sequence models.

Base class used specifically in state space models. Based on the implementation
from:

https://github.com/state-spaces/s4/blob/main/src/models/sequence/backbones/model.py
"""

from einops import rearrange
from functools import partial
import torch
import torch.nn as nn
from typing import Mapping, Optional, Union

from xdiffusion.layers.drop import StochasticDepth
from xdiffusion.layers.utils import to_list, to_dict, Normalization, DropoutNd
from xdiffusion.utils import instantiate_partial_from_config


class SequenceResidualBlock(torch.nn.Module):
    """Flexible residual block design."""

    def __init__(
        self,
        d_input: int,
        i_layer: int = None,  # Only needs to be passed into certain residuals like Decay
        prenorm: bool = True,
        bidirectional: bool = False,
        dropout: float = 0.0,
        tie_dropout: bool = False,
        transposed: bool = False,
        layer_config: Optional[Mapping] = None,  # Config for black box module
        residual_config: Optional[Mapping] = None,  # Config for residual function
        norm_config: Optional[
            Union[str, Mapping]
        ] = None,  # Config for normalization layer
        pool_config: Optional[Mapping] = None,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.i_layer = i_layer
        self.d_input = d_input
        self.prenorm = prenorm
        self.bidirectional = bidirectional
        self.transposed = transposed

        self.layer = instantiate_partial_from_config(layer_config)(d_input)
        if self.bidirectional:
            self.reverse_layer = instantiate_partial_from_config(layer_config)(d_input)
            self.bidirectional_linear = nn.Linear(
                2 * self.layer.d_output, self.layer.d_output
            )

        # Residual
        # d_residual is the output dimension after residual
        if residual_config is None:
            self.residual = None
            self.d_residual = self.layer.d_output
        else:
            self.residual = instantiate_partial_from_config(residual_config)(
                i_layer, d_input, self.layer.d_output
            )
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate since Normalization has some special cases
        if norm_config is None:
            self.norm = None
        elif isinstance(norm_config, str):
            self.norm = Normalization(
                d_norm, transposed=self.transposed, _name_=norm_config
            )
        else:
            self.norm = instantiate_partial_from_config(norm_config)(
                self.d_output, transposed=self.transposed
            )

        # Pool
        self.pool = instantiate_partial_from_config(pool_config)(
            self.d_residual, transposed=self.transposed
        )

        # Dropout
        dropout_cls = (
            partial(DropoutNd, transposed=self.transposed)
            if tie_dropout
            else nn.Dropout
        )
        self.drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        # Stochastic depth
        self.drop_path = (
            StochasticDepth(drop_path, mode="row") if drop_path > 0.0 else nn.Identity()
        )

    @property
    def d_output(self):
        return self.pool.d_output if self.pool is not None else self.d_residual

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def forward(self, x, state=None, **kwargs):
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm(y)

        # Black box layer
        y_for, new_state = self.layer(y, state=state, **kwargs)
        if self.bidirectional:
            assert state is None
            y_rev, _ = self.reverse_layer(y, state=state, **kwargs)
            if self.transposed:
                y = torch.cat([y_for, y_rev], dim=1)
            else:
                y = torch.cat([y_for, y_rev], dim=-1)
            y = self.bidirectional_linear(y)
        else:
            y = y_for

        # Residual
        if self.residual is not None:
            y = self.residual(x, self.drop_path(self.drop(y)), self.transposed)

        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm(y)

        # Pool
        if self.pool is not None:
            y, _ = self.pool(y)

        return y, state

    def step(self, x, state, **kwargs):
        assert not self.bidirectional
        y = x

        # Pre-norm
        if self.norm is not None and self.prenorm:
            y = self.norm.step(y)

        # Black box layer
        y, state = self.layer.step(y, state, **kwargs)

        # Residual
        if self.residual is not None:
            y = self.residual(
                x, y, transposed=False
            )  # NOTE this would not work with concat residual function (catformer)

        # Post-norm
        if self.norm is not None and not self.prenorm:
            y = self.norm.step(y)

        # Pool
        if self.pool is not None:
            y, _ = self.pool(y)

        return y, state


class SequenceModel(nn.Module):
    """Flexible isotropic deep neural network backbone.

    A SequenceModule is generally a model that transforms an input of shape
    (n_batch, l_sequence, d_model) to (n_batch, l_sequence, d_output)

    Options:
      - d_model: Model dimension. Inputs generally have shape (batch, length, d_model).
      - n_layers: Number of repeating blocks.
      - transposed: Transpose inputs so each layer receives (batch, d_model, length).
      - dropout: Dropout parameter applied on every residual and every layer.
      - tie_dropout: Tie dropout mask across sequence like nn.Dropout1d/nn.Dropout2d.
      - prenorm: Pre-norm vs. post-norm placement of the norm layer.
      - bidirectional: Concatenate two copies of each layer like a bi-LSTM.
      - n_repeat: Each layer is repeated n times per stage before applying (optional) pooling.
      - Layer config, must be specified.
      - residual: Residual config, or None for no residual.
      - norm: Normalization config (e.g. layer vs batch), or None for no norm.
      - pool: Config for pooling layer per stage, or None for no pooling.
      - track_norms: Log norms of each layer output.
      - dropinp: Input dropout.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        transposed: bool = False,
        dropout: int = 0.0,
        tie_dropout: bool = False,
        prenorm: bool = True,
        bidirectional: bool = False,
        n_repeat: int = 1,
        layer_config: Optional[Mapping] = None,
        residual_config: Optional[Mapping] = None,
        norm_config: Optional[Union[str, Mapping]] = None,
        pool_config: Optional[Mapping] = None,
        track_norms: bool = True,
        dropinp: int = 0.0,
    ):
        super().__init__()
        # Save arguments needed for forward pass
        self.d_model = d_model
        self.transposed = transposed
        self.track_norms = track_norms
        # Input dropout (not really used)
        dropout_fn = (
            partial(DropoutNd, transposed=self.transposed)
            if tie_dropout
            else nn.Dropout
        )
        self.drop = dropout_fn(dropinp) if dropinp > 0.0 else nn.Identity()

        layer_params = to_list(layer_config["params"], recursive=False)

        # Some special arguments are passed into each layer
        for _layer_param in layer_params:
            # If layers don't specify dropout, add it
            if _layer_param.get("dropout", None) is None:
                _layer_param["dropout"] = dropout
            # Ensure all layers are shaped the same way
            _layer_param["transposed"] = transposed

        # Duplicate layers
        layers = layer_params * n_layers * n_repeat

        # Instantiate layers
        _layers = []
        d = d_model
        for l, layer in enumerate(layers):
            # Pool at the end of every n_repeat blocks
            pool_cfg = pool_config if (l + 1) % n_repeat == 0 else None
            block = SequenceResidualBlock(
                d,
                l + 1,
                prenorm=prenorm,
                bidirectional=bidirectional,
                dropout=dropout,
                tie_dropout=tie_dropout,
                transposed=transposed,
                layer_config={"target": layer_config["target"], "params": layer},
                residual_config=residual_config,
                norm_config=norm_config,
                pool_config=pool_cfg,
            )
            _layers.append(block)
            d = block.d_output

        self.d_output = d
        self.layers = nn.ModuleList(_layers)
        if prenorm:
            if norm_config is None:
                self.norm = None
            elif isinstance(norm_config, str):
                self.norm = Normalization(
                    self.d_output, transposed=self.transposed, _name_=norm_config
                )
            else:
                self.norm = instantiate_partial_from_config(norm_config)(
                    self.d_output, transposed=self.transposed
                )
        else:
            self.norm = nn.Identity()

    def forward(self, inputs, *args, state=None, **kwargs):
        """Inputs assumed to be (batch, sequence, dim)"""
        if self.transposed:
            inputs = rearrange(inputs, "b ... d -> b d ...")
        inputs = self.drop(inputs)

        # Track norms
        if self.track_norms:
            output_norms = [torch.mean(inputs.detach() ** 2)]

        # Apply layers
        outputs = inputs
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            outputs, state = layer(outputs, *args, state=prev_state, **kwargs)
            next_states.append(state)
            if self.track_norms:
                output_norms.append(torch.mean(outputs.detach() ** 2))
        if self.norm is not None:
            outputs = self.norm(outputs)

        if self.transposed:
            outputs = rearrange(outputs, "b d ... -> b ... d")

        if self.track_norms:
            metrics = to_dict(output_norms, recursive=False)
            self.metrics = {f"norm/{i}": v for i, v in metrics.items()}

        return outputs, next_states

    @property
    def d_state(self):
        d_states = [layer.d_state for layer in self.layers]
        return sum([d for d in d_states if d is not None])

    @property
    def state_to_tensor(self):
        # Slightly hacky way to implement this in a curried manner (so that the function can be extracted from an instance)
        # Somewhat more sound may be to turn this into a @staticmethod and grab subclasses using hydra.utils.get_class
        def fn(state):
            x = [
                _layer.state_to_tensor(_state)
                for (_layer, _state) in zip(self.layers, state)
            ]
            x = [_x for _x in x if _x is not None]
            return torch.cat(x, dim=-1)

        return fn

    def default_state(self, *batch_shape, device=None):
        return [
            layer.default_state(*batch_shape, device=device) for layer in self.layers
        ]

    def step(self, x, state, **kwargs):
        # Apply layers
        prev_states = [None] * len(self.layers) if state is None else state
        next_states = []
        for layer, prev_state in zip(self.layers, prev_states):
            x, state = layer.step(x, state=prev_state, **kwargs)
            next_states.append(state)

        x = self.norm(x)

        return x, next_states
