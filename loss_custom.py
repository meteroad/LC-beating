import torch
import torch.nn.functional as F
import os
from torch import Tensor
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
def target_process(target):  
    # target = Tensor.long(target) 
    target = Tensor.float(target) 
    for shift in [-2, -1, 1, 2]:
        target = target + torch.roll(target/(2**abs(shift)), shifts=shift, dims=-1) 
        # target = target | torch.roll(target, shifts=shift, dims=-1) 
    target[target>1] = 1. 
    target = Tensor.float(target)
    return target  

class Weight_BCELoss(torch.nn.Module): # 暂时还可以用的一个loss
    """ linear add """
    def __init__(self, coef=1.0, target_aug = True):
        super(Weight_BCELoss, self).__init__()
        self.coef = coef
        self.target_aug = target_aug

    def forward(self, input, target, target_widen=True):
        # if self.target_aug:
        #     target = target_process(target)

        # split out the channels
        beat_act_target = target[:,0,:] # [B, C, T]
        downbeat_act_target = target[:,1,:] # [B, C, T]
        nobeat_act_target = target[:, 2, :] # [B, C, T]

        beat_act_input = input[:,0,:] # [B, C, T]
        downbeat_act_input = input[:,1,:] # [B, C, T]
        nobeat_act_input = input[:,2,:] # [B, C, T]

        # beat errors
        target_beats = beat_act_target[beat_act_target == 1]
        input_beats =  beat_act_input[beat_act_target == 1]
        b_f = target_beats.shape[-1]/(target.shape[-1]*target.shape[0])

        # beat_loss = F.binary_cross_entropy(input_beats, target_beats)
        beat_loss = F.binary_cross_entropy(beat_act_input, beat_act_target)

        # no beat errors
        target_no_beats = beat_act_target[beat_act_target == 0]
        input_no_beats = nobeat_act_input[beat_act_target == 0]
        nb_f = target_no_beats.shape[-1]/(target.shape[-1]*target.shape[0])

        # no_beat_loss = F.binary_cross_entropy(input_no_beats, target_no_beats)
        no_beat_loss = F.binary_cross_entropy(nobeat_act_input, nobeat_act_target)

        # downbeat errors
        target_downbeats = downbeat_act_target[downbeat_act_target == 1]
        input_downbeats = downbeat_act_input[downbeat_act_target == 1]
        d_f = target_downbeats.shape[-1]/(target.shape[-1]*target.shape[0])

        # downbeat_loss = F.binary_cross_entropy(input_downbeats, target_downbeats)
        downbeat_loss = F.binary_cross_entropy(downbeat_act_input, downbeat_act_target)

        # sum up losses
        # total_loss = beat_loss/b_f + no_beat_loss/nb_f + downbeat_loss/d_f
        total_loss = beat_loss/b_f + downbeat_loss/d_f

        return total_loss, beat_loss, downbeat_loss
    
class SoftBCELoss(torch.nn.Module): # default loss
    """ linear add """
    def __init__(self, target_aug = True):
        super(SoftBCELoss, self).__init__()
        self.target_aug = target_aug

    def forward(self, input, target, target_widen=True):        
        # split out the channels
        beat_act_target = target[:,0,:]
        downbeat_act_target = target[:,1,:]

        beat_act_input = input[:,0,:]
        downbeat_act_input = input[:,1,:]
        
        if self.target_aug:
            beat_act_target = target_process(beat_act_target)
            downbeat_act_target = target_process(downbeat_act_target)
        
        # beat errors  
        target_beats = beat_act_target[beat_act_target > 0]
        input_beats =  beat_act_input[beat_act_target > 0]
        
        beat_loss = F.binary_cross_entropy(input_beats, target_beats, reduce=False)

        # no beat errors
        target_no_beats = beat_act_target[beat_act_target == 0]
        input_no_beats = beat_act_input[beat_act_target == 0]

        no_beat_loss = F.binary_cross_entropy(input_no_beats, target_no_beats, reduce=False)

        # downbeat errors
        target_downbeats = downbeat_act_target[downbeat_act_target > 0]
        input_downbeats = downbeat_act_input[downbeat_act_target > 0]

        downbeat_loss = F.binary_cross_entropy(input_downbeats, target_downbeats, reduce=False)

        # no downbeat errors
        target_no_downbeats = downbeat_act_target[downbeat_act_target == 0]
        input_no_downbeats = downbeat_act_input[downbeat_act_target == 0]

        no_downbeat_loss = F.binary_cross_entropy(input_no_downbeats, target_no_downbeats, reduce=False)

        num = beat_act_target.numel()
        alpha, gamma = 0.5, 0.5

        total_beat_loss = alpha * beat_loss.sum() / num + (1 - alpha) * no_beat_loss.sum() / num
        total_downbeat_loss = gamma * downbeat_loss.sum() / num + (1 - gamma) * no_downbeat_loss.sum() / num
        total_loss = total_beat_loss + total_downbeat_loss

        return total_loss, total_beat_loss, total_downbeat_loss
    
class SoftBCELoss_new(torch.nn.Module):
    """ linear add """
    def __init__(self):
        super(SoftBCELoss_new, self).__init__()

    def forward(self, input, target, ifdownbeat, target_widen=True):
        
        downbeat_valid = torch.Tensor([k != 0 for k in ifdownbeat]).to(target.device).bool()
        downbeat_valid = downbeat_valid.unsqueeze(1).repeat(1, target.shape[-1])
        
        if target_widen:
            target = target_process(target)
        
        # split out the channels
        beat_act_target = target[:,0,:]
        downbeat_act_target = target[:,1,:]

        beat_act_input = input[:,0,:]
        downbeat_act_input = input[:,1,:]
        
        beat_loss = F.binary_cross_entropy(beat_act_input, beat_act_target)
        downbeat_loss = F.binary_cross_entropy(downbeat_act_input, downbeat_act_target, reduce=False)

        downbeat_loss = downbeat_loss.masked_select(downbeat_valid).mean()

        total_loss = beat_loss + downbeat_loss

        return total_loss, beat_loss, downbeat_loss
    