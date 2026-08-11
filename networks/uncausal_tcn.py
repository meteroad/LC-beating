import torch
import torch.nn as nn
import torchsummary
import sys
from nnAudio import Spectrogram
# from wavebeat.plot import plt_netout
import torchaudio
import torch.nn.functional as F

import torch.nn as nn
from torch.nn.utils import weight_norm


class NonCausalTemporalLayer(nn.Module):
    def __init__(
            self,
            inputs,
            outputs,
            dilation,
            kernel_size=5,
            stride=1,
            padding=4,
            dropout=0.1,
            residual=True):
        super(NonCausalTemporalLayer, self).__init__()

        self.conv1 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv2 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding),
                dilation=dilation * 2)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv1d(outputs * 2, outputs, 1)

        self.downsample = nn.Conv1d(inputs, outputs, 1)\
            if inputs != outputs else None
        
        self.residual = residual
        
        self.res = nn.Sequential(
            nn.Conv1d(inputs, outputs, 1))
        # self._initialise_weights(self.conv1, self.conv2, self.downsample)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y = torch.cat([y1, y2], 1)
        y = self.elu(y)
        y = self.dropout(y)
        y = self.conv(y)
        
        if self.downsample is not None:
            y = y + self.downsample(x)
        
        if self.residual:
            y = y + self.res(x)

        return y
    
    def _initialise_weights(self, *layers):
        for layer in layers:
            if layer is not None:
                layer.weight.data.normal_(0, 0.01)


class NonCausalTemporalConvolutionalNetwork(nn.Module):

    def __init__(self, inputs, channels, kernel_size=5, dropout=0.1):

        super(NonCausalTemporalConvolutionalNetwork, self).__init__()

        self.layers = []
        n_levels = len(channels)

        for i in range(n_levels):
            dilation = 2 ** i

            n_channels_in = channels[i - 1] if i > 0 else inputs
            n_channels_out = channels[i]

            self.layers.append(
                NonCausalTemporalLayer(
                    n_channels_in,
                    n_channels_out,
                    dilation,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )
        
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):

        y = self.net(x)
        
        return y

class MainModule(nn.Module):

    def __init__(
            self,
            channels=20,
            tcn_kernel_size=5,
            dropout=0.1,
            downbeats=False,
            **kwargs):

        super(MainModule, self).__init__()
        target_sr = 16000
        length_sec = 30
        octave_num= 9
        bins_per_o = 9
        fmin = 16 # C2
        n_fft = 512
        hop_length = 160
        fmax = fmin * (2 ** octave_num) # C8
        freq_bins = octave_num * bins_per_o
        
        self.spec_layer = Spectrogram.STFT(n_fft=n_fft, 
                                  freq_bins=freq_bins,
                                  hop_length=hop_length,
                                  freq_scale='log',
                                  fmin=fmin,
                                  fmax=fmax,
                                  output_format='Magnitude')
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        self.conv1 = nn.Conv2d(1, channels, (3, 3), padding=(1, 0))
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool2d((1, 3))

        self.conv2 = nn.Conv2d(channels, channels, (1, 12), padding=(0, 0))
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool2d((1, 3))
        
        self.conv3 = nn.Conv2d(channels, channels, (3, 3), padding=(1, 0))
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout)
        self.pool3 = nn.MaxPool2d((1, 3))

        self.tcn = NonCausalTemporalConvolutionalNetwork(
            channels,
            [channels] * 11,
            tcn_kernel_size,
            dropout)
        
        self.beat = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(channels, 1, 1)
        )
        
        self.dbeat = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(channels, 1, 1)
        )
        
        self.nobeat = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(channels, 1, 1)
        )


    def forward(self, x):
        """
        Feed a tensor forward through the BeatNet.

        Arguments:
            x {torch.Tensor} -- A PyTorch tensor of size specified in the constructor.
        Returns:
            torch.Tensor -- A PyTorch tensor of size specified in the
                            constructor.
        """
        x = self.spec_layer(x)
        # x = self.amplitude_to_db(x)
        spec = x[0,:,:].detach().cpu().numpy()
        x = x.permute(0, 2, 1)  
        x = x.unsqueeze(1)
        
        y = self.conv1(x)
        y = self.elu1(y)
        y = self.pool1(y)
        y = self.dropout1(y)

        y = self.conv2(y)
        y = self.elu2(y)
        y = self.pool2(y)
        y = self.dropout2(y)
        
        y = self.conv3(y)
        y = self.elu3(y)
        y = self.pool3(y)
        y = self.dropout3(y)

        y = y.view(-1, y.shape[1], y.shape[2])
        y = self.tcn(y)
        
        beat = self.beat(y)
        dbeat = self.dbeat(y)
        y = torch.cat([beat, dbeat], 1)
        y = torch.sigmoid(y)
        return y


if __name__ == "__main__":
    input_size = (1, 160000)
    net = MainModule()
    torchsummary.summary(net, input_size, device="cpu")