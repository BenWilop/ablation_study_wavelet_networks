import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional, List

##############################################################
# 1. G-Lifting Layers (Spectro-Temporal Layers               #
# Input: Audio timeseries: [B, T]                            #
# Output: Time, frequency plane with C kernels: [B, C, T, S] #
#                                                            #
# Fixed or learned kernel that gets broadcasted to           #
# scales 2^j for 1, 2, ..., 2^j < T / kernel_size            #
##############################################################

class WaveletLayer(nn.Module):
    """
    Input: f:R->R
    Output: B(f):G->R (exactly one channel)

    Group convolution with (real) morlet wavelet kernel
    p(x) = cos(5t) * e^{-t^2/2}
    B(f)(s,t) = (f(x) * p(x/s)/s)(t)
    """
    def __init__(self, S: Optional[int], kernel_size: int):
        super().__init__()
        self.S = S
        self.kernel_size = kernel_size
        time = t.linspace(-3, 3, kernel_size)
        self.kernel = t.cos(5.0 * time) * t.exp(-time ** 2 / 2).unsqueeze(0).unsqueeze(0)  # [K] -> [1, 1, K] to convolve with [Cin Cout T]

    def forward(self, x):  # -> [B, 1, T, S]
        # x [B, 1, T]
        T = x.shape[2]
        if self.S is not None:
            n_scales = self.S
        else:
            n_scales = max(1, int(math.log2(T / self.kernel_size)))
        
        outputs = []
        for j in range(n_scales):
            scale = 2 ** j
            padding = (self.kernel_size - 1) * scale // 2
            out = F.conv1d(x, self.kernel, dilation=scale, padding=padding)  # [B, 1, T]
            out = out / scale
            outputs.append(out.unsqueeze(-1))  # [B, 1, T, 1]
        return t.cat(outputs, dim=-1)  # [B, 1, T, S]

    
class LiftingConvolutionLayer(nn.Module):
    """
    Input: f:R->R
    Output: B(f):G->R^C (multiple channels)

    Group convolution with learned kernel p
    B(f)(s,t) = (f(x) * p(x/s)/s)(t)
    """
    def __init__(self, S: Optional[int], kernel_size: int, output_channels: int):
        super().__init__()
        self.S = S
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.kernel = nn.Parameter(
            t.randn(output_channels, 1, kernel_size)
        )

    def forward(self, x):  # -> [B, C, T, S]
        # x [B, 1, T]
        T = x.shape[2]
        if self.S is not None:
            n_scales = self.S
        else:
            n_scales = max(1, int(math.log2(T / self.kernel_size)))

        outputs = []
        for j in range(n_scales):
            scale = 2 ** j
            padding = (self.kernel_size - 1) * scale // 2
            out = F.conv1d(x, self.kernel, dilation=scale, padding=padding)  # [B, C, T]
            out = out / scale
            outputs.append(out.unsqueeze(-1))  # [B, C, T, 1]
        return t.cat(outputs, dim=-1)  # [B, C, T, S]


##############################################################
# 2. G-Convolutional Layers                                  #
# Input: Time, frequency plane with C kernels: [B, C, T, S]  #
# Output: Time, frequency plane with C kernels: [B, C, T, S] #
#                                                            #
# Performs convolutions over T, S and outputs T2, S2 shape   #        
# scales 2^j for 1, 2, ..., 2^j < T / kernel_size            #
##############################################################

class GroupConvolutionLayer(nn.Module):
    """
    Input: f:G->G
    Output: B(f):G->R^C (multiple channels)

    Group convolution with learned kernel p
    B(f)(s,t) = sum_z (f(x, z) * p(x/s)/s^2)(t)
    """
    def __init__(self, S: Optional[int], input_channels: int, output_channels: int, kernel_size: int):
        super().__init__()
        self.S = S
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.kernel = nn.Parameter(
            t.randn(output_channels, input_channels, kernel_size)
        )

    def forward(self, x):  # -> [B, C, T, S]
        # x [B, C, T, S]
        B, C, T, S_in = x.shape
        if self.S is not None:
            n_scales = self.S
        else:
            n_scales = max(1, int(math.log2(T / self.kernel_size)))

        integration_weights = t.tensor([1] + [2**(z-1) for z in range(1, S_in)], device=x.device, dtype=x.dtype)  # [S_in]
        integration_weights = integration_weights.view(1, 1, 1, S_in)  # [B, C, T, n_scales]

        outputs = []
        for j in range(n_scales):
            scale = 2 ** j
            padding = scale * (self.kernel_size - 1) // 2
            
            x_perm = x.permute(0, 3, 1, 2)  # [B, S, C, T]
            x_reshaped = x_perm.reshape(B * S_in , C, T)  # [B*S_in, C, T]
            conv_out = F.conv1d(x_reshaped, self.kernel, dilation=scale, padding=padding)  # [B*S_in, C=out_channels, T]
            conv_out = conv_out.reshape(B, S_in , self.kernel.shape[0], T)  # [B, S_in, C, T]
            conv_out = conv_out.permute(0, 2, 3, 1)  # [B, C, T, S_in]
            out = (conv_out * integration_weights).sum(dim=3) / (scale**2)  # [B, C, T]

            outputs.append(out.unsqueeze(-1))  # [B, C, T, 1]
        return t.cat(outputs, dim=-1)  # [B, C, T, S]
    

##############################################################
# 3. AudioClassifier                                         #
# Input: Time, frequency plane with C kernels: [B, C, T, S]  #
# Output: probabilities of n_classes: [n_classes]            #
#                                                            #
# G-Lifting Layers                                           #
# G-Convolutional Layers 1, ..., G-Convolutional Layers (n-1)#
# Mean-pooling over time                                     #
# G-Convolution Layer: kernel_size=1, n_channels=n_classes   # 
# Max-Pooling over scale                                     #
# Softmax                                                    #
##############################################################

class AudioClassifier(nn.Module):
    def __init__(self, G_lifting_layer: nn.Module, G_convolutional_layers: List[nn.Module], n_classes: int, bn_eps: float):
        super().__init__()
        self.G_lifting_layer = G_lifting_layer
        self.bn_G_lifting_layer = nn.BatchNorm2d(
            G_lifting_layer.output_channels if hasattr(G_lifting_layer, 'output_channels') else 1,  # in case of WaveletLayer,
            eps=bn_eps
        )
        self.G_convolutional_layers = nn.ModuleList(G_convolutional_layers[:-1])
        self.G_convolutional_layer_output = G_convolutional_layers[-1]
        self.bn_G_convolutional_layers = nn.ModuleList([
            nn.BatchNorm2d(
                layer.out_channels if hasattr(layer, 'out_channels') else layer.output_channels, 
                eps=bn_eps
            )
            for layer in self.G_convolutional_layers
        ])
        self.n_classes = n_classes
        if hasattr(self.G_convolutional_layer_output, 'out_channels'):
            output_channels = self.G_convolutional_layer_output.out_channels
        else:
            output_channels = self.G_convolutional_layer_output.output_channels
        self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

    def forward(self, x):  # -> [n_classes]
        # x [B, T]
        x = self.G_lifting_layer(x) # [B, C, T, S]
        x = self.pool(x)  # [B, C, T/4, S]
        x = self.bn_G_lifting_layer(x)  # [B, C, T/4, S]
        x = F.relu(x)  # [B, C, T/4, S]

        for conv, bn in zip(self.G_convolutional_layers, self.bn_G_convolutional_layers):
            x = conv(x)
            x = self.pool(x)
            x = bn(x)
            x = F.relu(x) # [B, C, T/4^k, S]

        x = t.mean(x, dim=2, keepdim=True)  # [B, C, 1, S]  Mean-pool over time
        x = self.G_convolutional_layer_output(x)  # [B, C, 1, S]
        x = t.max(x, dim=3, keepdim=True)[0]  # [B, C, 1, 1] Max-pool over scale
        x = x.squeeze(2).squeeze(2) # [B, C] <- each chanell will be one class, i.e. [B, n_classes]
        print(x)
        return x  # Logits


