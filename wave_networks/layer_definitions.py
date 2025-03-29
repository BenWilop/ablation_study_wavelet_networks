import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from jaxtyping import Float

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
    def __init__(self, S: int | None, kernel_size: int):
        super().__init__()
        self.S = S
        self.kernel_size = kernel_size
        time = t.linspace(-3, 3, kernel_size)
        self.kernel = t.cos(5.0 * time) * t.exp(-time ** 2 / 2)

    def forward(self, x: Float[Tensor, "B T"]) -> Float[Tensor, "B 1 T S"]:
        T = x.shape[1]
        if self.S is not None:
            n_scales = self.S
        else:
            n_scales = max(1, int(t.log2(T / self.kernel_size)))
        x = x.unsqueeze(1)  # [B, 1, T]
        
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
    def __init__(self, S: int | None, kernel_size: int, output_channels: int):
        super().__init__()
        self.S = S
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.kernel = nn.Parameter(
            t.randn(output_channels, 1, kernel_size)
        )

    def forward(self, x: Float[Tensor, "B T"]) -> Float[Tensor, "B C T S"]:
        T = x.shape[1]
        if self.S is not None:
            n_scales = self.S
        else:
            n_scales = max(1, int(t.log2(T / self.kernel_size)))
        x = x.unsqueeze(1)  # [B, 1, T]

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
    B(f)(s,t) = (f(x) * p(x/s)/s)(t)
    """
    def __init__(self, S: int | None, input_channels: int, output_channels: int, kernel_size: int):
        super().__init__()
        self.S = S
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.kernel = nn.Parameter(
            t.randn(input_channels, output_channels, 1, kernel_size)
        )

    def forward(self, x: Float[Tensor, "B C T S"]) -> Float[Tensor, "B C T S"]:
        T = x.shape[2]
        if self.S is not None:
            n_scales = self.S
        else:
            n_scales = max(1, int(t.log2(T / self.kernel_size)))

        outputs = []
        for j in range(n_scales):
            scale = 2 ** j
            padding = (self.kernel_size - 1) * scale // 2
            out = t.zeros(x.shape[0], self.kernel.shape[1], T, device=x.device, dtype=x.dtype)  # [B, C, T]
            for z in range(n_scales):
                if z == 0:
                    integration_weight = 1
                else:
                    integration_weight = 2**(z-1)
                out += integration_weight * F.conv1d(x[:, :, :, z], self.kernel, dilation=scale, padding=padding)  # [B, C, T]
            out = out / scale**2
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
    def __init__(self, G_lifting_layer: nn.Module, G_convolutional_layers: list[nn.Module], n_classes: int, bn_eps: float):
        super().__init__()
        self.G_lifting_layer = G_lifting_layer
        self.bn_G_lifting_layer = nn.BatchNorm2d(
            G_lifting_layer.output_channels if hasattr(G_lifting_layer, 'output_channels') else 1  # in case of WaveletLayer,
            eps=bn_eps
        )
        self.G_convolutional_layers = nn.ModuleList(G_convolutional_layers[:-1])
        self.G_convolutional_layer_output = G_convolutional_layers[-1]
        self.bn_G_convolutional_layers = nn.ModuleList([
            nn.BatchNorm2d(layer.out_channels, eps=bn_eps) for layer in self.G_convolutional_layers
        ])
        self.n_classes = n_classes
        assert self.G_convolutional_layers[-1].out_channels == n_classes
        self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

    def forward(self, x: Float[Tensor, "B T"]) -> Float[Tensor, "n_classes"]:
        x = self.G_lifting_layer(x) # [B, C, T, S]
        x = self.pool(x)  # [B, C, T/4, S]
        x = self.bn_G_lifting_layer(x)  # [B, C, T/4, S]
        x = F.relu(x)  # [B, C, T/4, S]

        for conv, bn in zip(self.G_convolutional_layers, self.bn_G_convolutional_layers):
            x = conv(x)
            x = self.pool(x)
            x = bn(x)
            x = F.relu(x) # [B, C, T/4^k, S]

        x = t.mean(x, dim=2, keepdim=True)  # [B, C, 1, S]
        x = self.G_convolutional_layer_output(x)  # [B, C, 1, S]
        x = t.mean(x, dim=3, keepdim=True)  # [B, C, 1, 1]
        x = x.squeeze(2).squeeze(2) # [B, C]
        return x  # Logits


