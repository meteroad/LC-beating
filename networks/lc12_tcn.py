from math import log2
import torch
import torch.nn as nn
from nnAudio import Spectrogram
import torchaudio
import torch.nn.functional as F
import torchsummary

import torch.nn as nn

# 倾斜金字塔型的TCN, 

class defConv2(nn.Module):
    def __init__(self, in_channels, out_channels, left_kernel=5, latency=2, dilation=2, bias=True):
        super().__init__()
        self.dilation = dilation
        self.left_attn_len = left_kernel
        
        # calculate conv list(based on bias) 
        dilation_list = []
        # right kernel
        dilation_list.append(-latency)
        # left kernel
        for j in range(left_kernel):
            dilation_list.append(j*dilation) # dilation list 是从小到大的
        
        self.dilation_list = dilation_list
        self.conv_list = []
        for i in range(len(dilation_list)):
            self.conv_list += [nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)]
        self.conv_list = nn.ModuleList(self.conv_list)
        
    def forward(self, input):  # 小于0和大于等于0可能需要单独处理
        # input [b x n_dim x T]
        # output [b x n_dim x T]
        B,feature_size,T = input.shape
        # input = F.pad(input, (0,(self.layer)*(self.left_attn_len-1)), mode="constant", value=0)
        i = 0
        tensors = torch.zeros((B, feature_size, T+(self.dilation)*(self.left_attn_len-1))).to(input.device)
        for conv in self.conv_list: 
            feature = F.pad(conv(input), (0,(self.dilation)*(self.left_attn_len-1)), mode="constant", value=0)
            tensor = torch.roll(feature, shifts=self.dilation_list[i], dims=-1)
            tensors = tensors + tensor
            i += 1
        
        # y = torch.sum(tensors, dim=0)
        y = tensors[:,:,:T]
                      
        return y

class latent_CausalConv1d(nn.Module):
    def __init__(
            self,
            inputs,
            outputs,
            dilation,
            kernel_size=5,
            dropout=0.1,
            residual=True,
            latency=0):
        super(latent_CausalConv1d, self).__init__()

        self.conv1 = defConv2(
                inputs,
                outputs,
                left_kernel = kernel_size,
                latency=latency,
                dilation=dilation)
        self.conv2 = defConv2(
                inputs,
                outputs,
                left_kernel = kernel_size,
                latency=latency,
                dilation=dilation*2)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv1d(outputs * 2, outputs, 1)

        self.downsample = nn.Conv1d(inputs, outputs, 1)\
            if inputs != outputs else None
        
        self.residual = residual
        
        self.res = nn.Sequential(
            nn.Conv1d(inputs, outputs, 1))

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

class CausalTemporalConvolutionalNetwork(nn.Module):
    def __init__(self, inputs, channels, kernel_size=5, dropout=0.1, latency=0):
        super(CausalTemporalConvolutionalNetwork, self).__init__()

        self.layers = []
        n_levels = len(channels)
        unused_latency = latency

        for i in range(n_levels):
            dilation = 2 ** i

            n_channels_in = channels[i - 1] if i > 0 else inputs
            n_channels_out = channels[i]
            layer_latency = round(unused_latency/(n_levels-i))
            
            self.layers.append(
                latent_CausalConv1d(
                    n_channels_in,
                    n_channels_out,
                    dilation,
                    kernel_size,
                    latency=layer_latency,
                    dropout=dropout
                )
            )  
            unused_latency = unused_latency - layer_latency     
        
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
            latency=10,
            **kwargs):
        super(MainModule, self).__init__()
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
        
        self.conv1 = nn.Conv1d(1, channels, (3, 3), padding=(1, 0))
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool2d((1, 3))

        self.conv2 = nn.Conv1d(channels, channels, (1, 12), padding=(0, 0))
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool2d((1, 3))
        
        self.conv3 = nn.Conv1d(channels, channels, (3, 3), padding=(1, 0))
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout)
        self.pool3 = nn.MaxPool2d((1, 3))

        self.tcn = CausalTemporalConvolutionalNetwork(
            channels,
            [channels] * 10,
            tcn_kernel_size,
            dropout,
            latency)
        
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
        # y = self.out(y)
        y = torch.sigmoid(y)

        return y

if __name__ == "__main__":
    # input_size = (1, 160000)
    input = torch.randn(2, 1, 160000)
    net = MainModule()
    output =net(input)
    # torchsummary.summary(net, input_size, device="cpu")
    print("1")