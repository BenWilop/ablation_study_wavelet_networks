import torch as t
import torch.nn.functional as F

from wave_networks.layer_definitions import *


class StandardConvolutionalNetwork(t.nn.Module):
    """
    Standard NN that uses multiple layers of 1D convolutions over the time axis.
    Code copied from https://github.com/dwromero/wavelet_networks/blob/master/experiments/UrbanSound8K/models/MNets_EtAl2016.py
    """
    def __init__(self):
        super().__init__()
        # Parameters of the model
        use_bias = False
        eps = 2e-5
        n_channels = 256
        n_classes = 10

        # Conv Layers
        self.c1 = t.nn.Conv1d(in_channels=1,          out_channels=n_channels, kernel_size=80, stride=4, padding=(80 // 2), dilation=1, bias=use_bias)
        self.c2 = t.nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=3,  stride=1, padding=(3 // 2),  dilation=1, bias=use_bias)
        self.c3 = t.nn.Conv1d(in_channels=n_channels, out_channels=n_classes,  kernel_size=1,  stride=1, padding=0,         dilation=1, bias=use_bias)
        # BatchNorm Layers
        self.bn1 = t.nn.BatchNorm1d(num_features=n_channels, eps=eps)
        self.bn2 = t.nn.BatchNorm1d(num_features=n_channels, eps=eps)

        # Pooling
        self.pool = t.max_pool1d

        # Initialization
        for m in self.modules():
            if isinstance(m, t.nn.Conv1d):
                m.weight.data.normal_(0, t.prod(t.Tensor(list(m.weight.shape)[1:]))**(-1/2))
                if use_bias: m.bias.data.fill_(0.0)

    def forward(self, x):
        out = t.relu(self.bn1(self.c1(x)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        out = t.relu(self.bn2(self.c2(out)))
        out = self.pool(out, kernel_size=4, stride=4, padding=0)
        # Global pooling
        out = t.mean(out, dim=-1, keepdim=True) # pool over the time axis
        out = self.c3(out)
        out = out.view(out.size(0), 10)
        return out


def build_WaveletNetwork() -> t.nn.Module:
    eps = 2e-5
    n_channels_G = int(256 / 1.7) 
    n_classes = 10
    
    lifting_layer = LiftingConvolutionLayer(S=9, kernel_size=79, output_channels=n_channels_G)
    group_conv1 = GroupConvolutionLayer(S=3, input_channels=n_channels_G,
                                        output_channels=n_channels_G, kernel_size=3)
    group_conv2 = GroupConvolutionLayer(S=3, input_channels=n_channels_G,
                                        output_channels=n_classes, kernel_size=1)

    G_convolutional_layers = [group_conv1, group_conv2]
    return AudioClassifier(G_lifting_layer=lifting_layer,
                            G_convolutional_layers=G_convolutional_layers,
                            n_classes=n_classes,
                            bn_eps=eps)


def build_minimal_WaveletNetwork() -> t.nn.Module:
    eps = 2e-5
    n_channels_G = 1
    n_classes = 10
    
    lifting_layer = LiftingConvolutionLayer(S=None, kernel_size=79, output_channels=1)
    group_conv1 = GroupConvolutionLayer(S=None, input_channels=n_channels_G,
                                        output_channels=n_channels_G, kernel_size=11)
    group_conv2 = GroupConvolutionLayer(S=None, input_channels=n_channels_G,
                                        output_channels=n_classes, kernel_size=1)

    G_convolutional_layers = [group_conv1, group_conv2]
    return AudioClassifier(G_lifting_layer=lifting_layer,
                            G_convolutional_layers=G_convolutional_layers,
                            n_classes=n_classes,
                            bn_eps=eps)


def build_2D_ConvolutionalNet() -> t.nn.Module:
    n_classes = 10
    wavelet_layer = WaveletLayer(S=None, kernel_size=79)

    conv1 = t.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, bias=False)
    conv2 = t.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=False)
    conv3 = t.nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1, bias=False)
    conv_layers = [conv1, conv2, conv3]
    
    model = AudioClassifier(G_lifting_layer=wavelet_layer,
                            G_convolutional_layers=conv_layers,
                            n_classes=n_classes,
                            bn_eps=2e-5)
    return model