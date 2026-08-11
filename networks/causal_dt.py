import math
import torch 
from torch import nn 
from torch.nn import TransformerEncoderLayer as torchTransformerEncoderLayer
import torch.nn.functional as F 
from torch.nn.modules.normalization import LayerNorm
from nnAudio import Spectrogram

# dilated control causal transformer, latency=10

class DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding(nn.Module):
    def __init__(self, dmodel, num_heads, dropout=0, Er_provided=False, left_kernel=5, latency=10):
        super(DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding, self).__init__()
        self.left_attn_len = left_kernel
        self.right_attn_len = 1
        self.latency = latency 
        
        self.dmodel = dmodel
        self.num_heads = num_heads
        self.head_dim = dmodel // num_heads
        assert self.head_dim * num_heads == dmodel, "embed_dim must be divisible by num_heads"

        self.key = nn.Linear(dmodel, dmodel)
        self.value = nn.Linear(dmodel, dmodel)
        self.query = nn.Linear(dmodel, dmodel)
        self.dropout = nn.Dropout(dropout)
        self.Er_provided = Er_provided
        
        if not Er_provided:
            self.Er = nn.Parameter(torch.randn(num_heads, self.head_dim, self.left_attn_len+self.right_attn_len))

    def forward(self, query, key, value, layer=0):
        #query, key, and value: (batch, time, dmodel), float tensor
        #Er: (num_head, head_dim, attn_len)
        batch, time, d_model = query.shape

        q = self.query(query).reshape(batch, time, self.num_heads, 1, self.head_dim).transpose(1, 2)  #(batch, num_head, time, 1, head_dim)
        k = self.key(key).reshape(batch, time, self.num_heads, 1, self.head_dim).transpose(1, 2)  #(batch, num_head, time, 1, head_dim)
        v = self.value(value).reshape(batch, time, self.num_heads, 1, self.head_dim).transpose(1, 2)  #(batch, num_head, time, 1, head_dim)

        k = torch.cat(
                        (
                        self.kv_roll(k[:, 0: 8], layer, padding_value=0),
                        ),
                    dim=1
                    )

        v = torch.cat(
                        (
                        self.kv_roll(v[:, 0: 8], layer, padding_value=0),
                        ),
                    dim=1
                    )
        
        Er_t = self.Er.unsqueeze(1).unsqueeze(0)  #(1, num_head, 1, head_dim, attn_len)

        qk = torch.matmul(q, k.transpose(-2, -1))
        attn_mask = torch.zeros_like(qk).masked_fill_((qk==0), 1e-9)
        #attn = torch.matmul(q, k.transpose(-2, -1) + Er_t) / math.sqrt(self.head_dim)
        attn = (qk + torch.matmul(q, Er_t)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn + attn_mask, dim=-1)

        out = torch.matmul(attn, v) #(batch, num_head, time, 1, head_dim)
        out = out.squeeze(-2).transpose(1, 2).reshape(batch, time, d_model)

        return self.dropout(out), attn

    def kv_roll(self, tensor, layer, padding_value=0):
        #tensor: (batch, num_head, time, 1, head_dim)
        _, _, time, _, _ = tensor.shape
        
        # calculate kernel list(based on bias) 
        dilation_list = []
        # right kernel
        dilation_list.append(-round((1+layer)/11*self.latency))
        # left kernel
        for j in range(self.left_attn_len):
            dilation_list.append(j * 2**(layer)) # dilation list 是从小到大的       
        # self.dilation_list = dilation_list  

        tensor = F.pad(tensor, (0, 0, 0, 0, 0, (2**layer)*(self.left_attn_len-1)), mode='constant', value=padding_value)  # 绝对因果
        #(batch, num_head, time+(2**layer)*(self.attn_len//2), 1, head_dim)

        tensor = torch.cat([torch.roll(tensor, shifts=i, dims=2) for i in dilation_list], dim=-2) 
        #(batch, num_head, time+(2**layer)*(self.attn_len//2), attn_len, head_dim)

        return tensor[:, :, :time, :, :]    #(batch, num_head, time, attn_len, head_dim)

class CausalConv2d(nn.Module):
    def __init__(self, kernel=(3,3), in_channels=1, out_channels=1):
        super(CausalConv2d, self).__init__()
        self.dilation_list = []
        self.conv_list = []
        for i in range(kernel[0]):
            self.dilation_list.append(i)
        for i in range(len(self.dilation_list)):
            self.conv_list +=[nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel[1]))]
        self.conv_list = nn.ModuleList(self.conv_list)
        
    def forward(self, input):
        dilation =self.dilation_list[0]
        y = self.conv_list[0](input)
        y = F.pad(y, pad=[0, dilation])
        y = y[:,:,dilation:, :]
        for i in range(1, len(self.conv_list)):
            dilation = self.dilation_list[i]
            x = self.conv_list[i](input)
            x = x[:,:,dilation:,:]
            n_dim = x.size()[-2]
            y[:, :, :n_dim, :] += x     
        return y   
        

class DilatedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, Er_provided=False, attn_len=5, norm_first=False, layer_norm_eps=1e-5, latency=10):
        super(DilatedTransformerLayer, self).__init__()
        self.self_attn = DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding(d_model, nhead, dropout, Er_provided, attn_len, latency=latency)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, x, layer=0):
        #x: (batch, time, dmodel)
        #Er: (num_head, head_dim, attn_len)
        if self.norm_first:
            x_ = self._sa_block(self.norm1(x), layer)[0]
            x = x + x_
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, layer)[0])
            x = self.norm2(x + self._ff_block(x))
        return x, x_

    # self-attention block
    def _sa_block(self, x, layer=0):
        x, attn = self.self_attn(x, x, x, layer)
        return self.dropout1(x), attn

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class MainModule(nn.Module):
    def __init__(self, attn_len=5, ntoken=2, dmodel=64, nhead=8, d_hid=64, nlayers=11, norm_first=True, dropout=.1, latency=0):
        super(MainModule, self).__init__()
        self.nhead = nhead
        self.nlayers = nlayers
        self.attn_len = attn_len # 这个不能是1，淦
        self.head_dim = dmodel // nhead
        self.dmodel = dmodel
        assert self.head_dim * nhead == dmodel, "embed_dim must be divisible by num_heads"
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

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3),  padding=(1, 0))#126
        self.conv1 = CausalConv2d(in_channels=1, out_channels=8,kernel=(3,3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3))#42
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(1, 12), padding=(0, 0))#31
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3))#10
        self.dropout2 = nn.Dropout(p=dropout)
        
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=dmodel, kernel_size=(3, 3), padding=(1, 0))#5
        self.conv3 = CausalConv2d(in_channels=32, out_channels=dmodel,kernel=(3,3))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3))#1
        self.dropout3 = nn.Dropout(p=dropout)
        
        self.Transformer_layers = nn.ModuleDict({})
        for idx in range(nlayers):
            self.Transformer_layers[f'time_attention_{idx}'] = DilatedTransformerLayer(dmodel, nhead, d_hid, dropout, Er_provided=False, attn_len=attn_len, norm_first=norm_first, latency=latency)
            
        self.out_linear = nn.Conv1d(dmodel, ntoken, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        #x: (batch, instr, time, dmodel), FloatTensor
        #batch, time, dmodel = x.shape
        spec = self.spec_layer(x)
        _, _, time = spec.shape
        # batch, instr, time, melbin = x.shape
        x = spec.permute(0, 2, 1)  
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = torch.relu(x)
        x = self.dropout3(x)    #(batch*instr, channel, time, 1)

        x = x.reshape(-1, self.dmodel, time).transpose(1, 2)    #(batch*instr, time, channel=dmodel)
        t = []

        for layer in range(self.nlayers):
            x, skip = self.Transformer_layers[f'time_attention_{layer}'](x, layer=layer)
            
        x = x.permute(0, 2, 1)
        x = self.out_linear(x)
        x = self.sigmoid(x)

        return x

if __name__ == '__main__':
    DEVICE = 'cpu'
    model = MainModule()
    input = torch.randn(4, 1, 160000)
    model.to(DEVICE)
    # output, tempo = model(input)
    output = model(input)
    # model.eval()

    for name, param in model.state_dict().items():
        print(name, param.shape)
    # name: str
    # param: Tensor

    total = sum([param.nelement() for param in model.parameters()])
    print(total)

