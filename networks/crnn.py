# unidirectional lstm

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("/home/fd-lamt-04/lulu/BeatNet")
from log_spect import LOG_SPECT

# hop length 320, not final version of CRNN

class MainModule(nn.Module):  #beat_downbeat_activation
    def __init__(self, dim_in=250, num_cells=150, num_layers=2, device="cuda:0", mode='online'):
        super(MainModule, self).__init__()

        self.mode = mode
        self.sample_rate = 16000
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate) ## 需要仔细看这里是10还是20
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)    

        self.proc = LOG_SPECT(sample_rate=self.log_spec_sample_rate, win_length=self.log_spec_win_length, hop_size=self.log_spec_hop_length, n_bands=[24], mode = self.mode)

        """ ----------------------------------------------"""

        self.dim_in = dim_in
        self.dim_hd = num_cells
        self.num_layers = num_layers
        # self.device = device
        self.conv_out = 150
        self.kernelsize = 10
        self.conv1 = nn.Conv1d(1, 2, self.kernelsize)
        self.linear0 = nn.Linear(2*int((self.dim_in-self.kernelsize+1)/2), self.conv_out)     #divide to 2 is for max pooling filter
        self.lstm1 = nn.LSTM(input_size=self.conv_out,  # self.dim_in
                            hidden_size=self.dim_hd,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=False,
                            )

        self.linear = nn.Linear(in_features=self.dim_hd,
                                out_features=3)

        self.softmax = nn.Softmax(dim=1)

        # self.to(device)

    def forward(self, data):
        b, _, _ = data.shape
        self.device = data.device
        x = self.preprocess(data)
        x = torch.reshape(x, (-1, self.dim_in))
        x = x.unsqueeze(0).transpose(0, 1)
        # x = x.transpose(0,1)
        x = self.conv1(x)
        x = F.relu(x)
        y = F.max_pool1d(x, 2)
        n_f = self.num_flat_features(y)
        # x = y.view(b, -1, n_f)
        x = torch.reshape(y, (b,-1,n_f))
        vis1 = x[0,:,:].detach().cpu().numpy()
        x = self.linear0(x)
        vis2 = x[0,:,:].detach().cpu().numpy()

        x, _ = self.lstm1(x)

        out = self.linear(x)
        out = out.transpose(1, 2)
        out = F.softmax(out, 1) # important
        vis3 = out[0,:,:].detach().cpu().numpy()
        return out

    def final_pred(self, input):
        return self.softmax(input)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def preprocess(self, input):
        # default type=="online"
        ret = []
        input = input.cpu()
        input = input.numpy()
        for b in range(input.shape[0]):
            audio = input[b,0,:]
            feats = self.proc.process_audio(audio).T
            ret.append(feats)
        ret = np.array(ret)
        ret = torch.from_numpy(ret)
        ret = ret.to(self.device)
        return ret

if __name__ == "__main__":
    net = MainModule(250, 150, 2, "cpu")
    # input = torch.rand((1,174,272))
    input = torch.rand((8,1,160000))
    output = net(input)
    output = net.final_pred(output)
    print("1")