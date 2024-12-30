import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, List
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

class DepthwisePointwiseConvBlock(nn.Module):
    def __init__(self, in_features=768, reduction_factor=8):
        super().__init__()
        self.in_channels = 1
        # Calculate intermediate dimension based on reduction factor
        self.intermediate_dim = in_features // reduction_factor
        
        # Pointwise convolution to reduce dimensions (like LoRA's A matrix)
        self.pointwise = nn.Conv2d(
            self.in_channels, 
            self.in_channels,
            kernel_size=1,
            groups=1,
            bias=False
        )
        
        # Depthwise convolution for spatial dependencies (like LoRA's B matrix)
        self.depthwise = nn.Conv2d(
            self.in_channels,
            self.in_channels, 
            kernel_size=3,
            padding=1,
            groups=self.in_channels,
            bias=False
        )

    def forward(self, x):
        batch_size, seq_len, in_features = x.shape
        
        # Reshape for 2D convolution [batch, channel, height, width]
        x = x.view(batch_size, self.in_channels, in_features, seq_len)
        
        # Apply pointwise -> depthwise convolutions
        x = self.pointwise(x)
        x = self.depthwise(x)
        
        # Reshape back to original dimensions
        x = x.view(batch_size, seq_len, in_features)
        return x

class EnsembleLayer(nn.Module, BaseTunerLayer):
    """
    Custom convolutional adapter layer for PEFT.
    """

    adapter_layer_names = ("conv_adapter_layers",)

    def __init__(self, base_layer: nn.Module, adapter_name: str):
        super().__init__()
        self.base_layer = base_layer
        self.conv_adapter_layers = nn.ModuleDict({})
        self.update_layer(adapter_name)
        self._active_adapter = adapter_name
        self.merged_adapters = []

    def update_layer(self, adapter_name: str):
        self.conv_adapter_layers[adapter_name] = DepthwisePointwiseConvBlock()

    def enable_adapters(self, enabled: bool) -> None:
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(self.active_adapters)
            self._disable_adapters = False
        else:
            if self.merged:
                self.unmerge()
            # disable grads on all adapter layers
            for layer_name in self.adapter_layer_names:
                layer = getattr(self, layer_name)
                layer.requires_grad_(False)
            self._disable_adapters = True

    def merge(self, adapter_names: Optional[List[str]] = None):
        adapter_names = self.check_adapters_to_merge(adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        if len(adapter_names) > 1:
            raise ValueError(
                f"Trying to merge {len(adapter_names)} adapters, but CustomConv "
                f"does not allow merging more than one adapter at a time"
            )
        merged_adapters = set(self.merged_adapters)
        if merged_adapters:
            warnings.warn(f"Already merged with {merged_adapters}. Unmerging first.")
            self.unmerge()

        self.base_layer, self.conv_adapter_layers[adapter_names[0]] = (
            self.conv_adapter_layers[adapter_names[0]],
            self.base_layer,
        )
        self.merged_adapters.append(adapter_names[0])

    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        # popping one element is sufficient because CustomConv
        # does not allow merging more than one adapter at a time.
        merged_name = self.merged_adapters.pop()
        self.base_layer, self.conv_adapter_layers[merged_name] = (
            self.conv_adapter_layers[merged_name],
            self.base_layer,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self._disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if len(self._active_adapter) != 1:
                raise ValueError(
                    f"Trying to run forward with {len(self._active_adapter)} active "
                    f"adapters, but CustomConv does not allow inference with more than one adapter at a time"
                )
            active_adapter = self._active_adapter[0]
            result = self.conv_adapter_layers[active_adapter](x, *args, **kwargs)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ensemble." + rep

    def set_adapter(self, adapter_name):
        self._active_adapter = adapter_name

    def check_adapters_to_merge(self, adapter_names):
        # This function should check and return the list of adapters to merge
        if adapter_names is None:
            adapter_names = list(self.conv_adapter_layers.keys())
        return adapter_names
