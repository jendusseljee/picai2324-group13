# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adaptions made for the PICAI challenge.

import warnings
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.utils import alias, deprecated_arg, export


class DAFT(nn.Module):
    def __init__(
        self,
        P: int, # Length of tabular data
        C: int, # channels in feature map
        r: int
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(P+C, (P+C)//r)
        self.fc2 = nn.Linear((P+C)//r, C*2)

    def forward(
            self, 
            x_img: torch.Tensor, # Feature map
            x_tab: torch.Tensor # Tabular data
        ) -> torch.Tensor:

        # Global Average Pooling
        x_pooled = F.adaptive_avg_pool3d(x_img.unsqueeze(1), output_size=1)
        x_pooled = x_pooled.view(x_pooled.size(0), -1)

        # Add tabular features
        x = torch.cat((x_pooled, x_tab), 1)

        # Learnable part
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Reconstruct image features
        a, b = torch.tensor_split(x, 2, dim=1)
        a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = a * x_img + b

        return x
    
class SkipConnection(nn.Module):
    def __init__(self, submodule, dim: int = 1) -> None:
        super().__init__()
        self.submodule = submodule
        self.dim = dim

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        y_img = self.submodule(x_img, x_tab)

        return torch.cat([x_img, y_img], dim=self.dim)
    

class ModuleWrapper(nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x_img, x_tab):
        x = self.module(x_img)
        return x
    
class Sequential(nn.Module):
    def __init__(self, mod1, mod2, mod3=None) -> None:
        super().__init__()
        self.mod1 = mod1
        self.mod2 = mod2
        self.mod3 = mod3

    def forward(self, x_img, x_tab):
        x = self.mod1(x_img, x_tab)
        x = self.mod2(x, x_tab)

        if self.mod3 is not None:
            x = self.mod3(x, x_tab)

        return x


class UNetDAFT(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        num_tab_features: int,
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
        daft_r: int = 7
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.num_tab_features = num_tab_features
        self.daft_r = daft_r
        
        self.current_batch_tabular = None

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ModuleWrapper(ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            ))
            return mod
        mod = ModuleWrapper(Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        ))
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        daft = DAFT(
            P=self.num_tab_features,
            C=in_channels,
            r=self.daft_r
        )

        down_layer = self._get_down_layer(in_channels, out_channels, strides=1, is_top=False)

        return Sequential(
            daft,
            down_layer
        )

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Union[Convolution, nn.Sequential]

        conv = ModuleWrapper(Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        ))

        if self.num_res_units > 0:
            ru = ModuleWrapper(ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            ))
            conv = Sequential(conv, ru)

        return conv

    def forward(self, x_img, x_tab) -> torch.Tensor:
        x_seg = self.model(x_img, x_tab)
        return x_seg

