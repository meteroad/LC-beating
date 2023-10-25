import torch
from collections import deque
from networks.lc42_tcn import MainModule

class FixedSizeQueue:
    """
        queue for the inference process of lc beating mechanism
    """
    def __init__(self, max_size, size):
        # queue initialization
        zero_tensor = torch.zeros(size)
        self.queue = deque([zero_tensor for _ in range(max_size)], maxlen=max_size)
        self.max_size = max_size

    def enqueue(self, item):
        # put item into queue while pop the same size of item
        while len(self.queue) >= self.max_size:
            self.queue.pop()
        self.queue.appendleft(item)

    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        else:
            raise IndexError("queue is empty")

    def size(self):
        return len(self.queue)
    
class tcn_infer():
    """
        inference module of tcn
    """
    def __init__(
        self,
        layer_info,
    ):
        super(tcn_infer, self).__init__()
        defconv1_module = layer_info.conv1
        defconv1_list = defconv1_module.conv_list._modules
        self.conv1_infer = latent_CausalConv1d_infer(convs=defconv1_list, dilation_list=defconv1_module.dilation_list)
        
        defconv2_module = layer_info.conv2
        defconv2_list = defconv2_module.conv_list._modules
        self.conv2_infer = latent_CausalConv1d_infer(convs=defconv2_list, dilation_list=defconv2_module.dilation_list)
        
        self.conv = layer_info.conv
        self.dropout = layer_info.dropout
        self.elu = layer_info.elu
        
        self.downsample = layer_info.downsample
        self.res = layer_info.res
        self.residual = layer_info.residual
         
    def infer(self, input):
        y1 = self.conv1_infer.infer(input)
        y2 = self.conv2_infer.infer(input)
        y = torch.cat([y1, y2], 1)
        y = self.elu(y)
        y = self.dropout(y)
        y = self.conv(y)
        
        if self.downsample is not None:
            y = y + self.downsample(input)
        
        if self.residual:
            y = y + self.res(input)
         
        return y
    
    
class latent_CausalConv1d_infer():
    """ inference module for latent_CausalConv1d module"""
    def __init__(
        self,
        convs,
        dilation_list,
        dropout=0.1,
        channels=20,
        
    ):
        super(latent_CausalConv1d_infer, self).__init__()
        self.dropout = dropout
        self.latency = dilation_list[0]
        self.dilation_list = dilation_list
        buffer_length = dilation_list[-1]-dilation_list[0]+1
        self.tcn_queue = FixedSizeQueue(max_size=buffer_length, size=[1, channels, 1]) # tcn buffer setup
        self.state_dicts = convs
        
    def infer(self, input):
        self.tcn_queue.enqueue(input)
        output = torch.zeros(input.shape)
        
        for i, sd in enumerate(self.state_dicts):
            tmp = list(self.tcn_queue.queue)[self.dilation_list[i]-self.latency]
            conv = self.state_dicts[sd]
            output += conv(tmp)
       
        return output
        

class lc42_inference():
    def __init__(
        self,
        channels=20,
        downbeat=True,
        latency=40,
        max_length = 2048,
        weight_path='./model/lc42_tcn-round2_device0-7/157.pkl'
    ):
        super(lc42_inference, self).__init__()
        self.channels = channels
        self.latency = latency
        self.max_length = max_length
        
        # initialize queue for conv1 and conv3
        self.conv1_queue = FixedSizeQueue(max_size=3, size=[1, 1, 81])
        # self.conv2_queue = FixedSizeQueue(max_size=1, size=[1, 20, 26])
        self.conv3_queue = FixedSizeQueue(max_size=3, size=[1, 20, 5])
        
        
        # model loading
        self.weights = torch.load(weight_path, map_location=torch.device('cpu'))
        self.state_dict = self.weights['state_dict']      
        self.model = MainModule(channels=channels)   
        self.model.load_state_dict(self.state_dict)
        
        self.model.eval()
        
        self.tcn = self.model.tcn.layers
        self.tcn_modules = []
        
        for layer in self.tcn:
            tcn_module = tcn_infer(layer_info=layer)
            self.tcn_modules.append(tcn_module) 
        
        print("initialization finished.")
        
            
    def inference_by_frame(self, x): 
        self.conv1_queue.enqueue(x)
        conv1_input = torch.stack(list(self.conv1_queue.queue)[:])
        conv1_input = conv1_input.permute(2, 1, 0, 3)

        # conv1_input [B, 1, 3, F], F=81, lc = 1
        conv1_output = self.model.conv1(conv1_input)
        conv1_output = self.model.elu1(conv1_output)
        conv1_output = self.model.dropout1(conv1_output)
        conv1_output = self.model.pool1(conv1_output)
        
        # conv1_output [B, channel, 3, F], F=26
        conv1_output = conv1_output[:,:,1,:]
        conv2_input = conv1_output.unsqueeze(2)
        
        conv2_output = self.model.conv2(conv2_input)
        conv2_output = self.model.elu2(conv2_output)
        conv2_output = self.model.dropout2(conv2_output)
        conv2_output = self.model.pool2(conv2_output)
        
        # conv2_output [B, channel, 1, F], F=5, lc = 1
        conv2_output = conv2_output[:,:,0,:]
        self.conv3_queue.enqueue(conv2_output)
        conv3_input = torch.stack(list(self.conv3_queue.queue)[:])
        conv3_input = conv3_input.permute(1, 2, 0, 3)
        
        conv3_output = self.model.conv3(conv3_input)
        conv3_output = self.model.elu3(conv3_output)
        conv3_output = self.model.dropout3(conv3_output)
        conv3_output = self.model.pool3(conv3_output)
        
        tcn_input = conv3_output[:,:,1,:] # 
           
        for t_modules in self.tcn_modules:
            tcn_input = t_modules.infer(tcn_input) # TODO: something wrong, but i don know how and where
            
        tcn_output = tcn_input
        
        # [1, channels, 1]
        beat_act = self.model.beat(tcn_output)    
        downbeat_act = self.model.dbeat(tcn_output)         
        
        y = y = torch.cat([beat_act, downbeat_act], 1)
        y = torch.sigmoid(y)
        y = y.squeeze(0)
        y = y.squeeze(-1)
        
        return y

        
if __name__ == "__main__":
    duration = 1001
    input = torch.randn(1, 1, duration, 81)
    infer = lc42_inference(channels=20)
    stacked_tensor = None
    
    for i in range(duration):
        o = infer.inference_by_frame(input[:,:,i,:])
        if stacked_tensor is None:
            stacked_tensor = o.unsqueeze(0)
        else:
            stacked_tensor = torch.cat((stacked_tensor, o.unsqueeze(0)), dim=0)
    
    print("1")
        
    
    