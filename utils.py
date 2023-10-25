
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

class Logger(SummaryWriter):
    """
        logger class for logging messages
    """
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)
        self.log_idx = 0

    def log_training(self, training_loss, validation_loss, iteration):
        self.add_scalar("training.loss", training_loss, iteration)
        self.add_scalar("validation.loss", validation_loss, iteration)

    def log(self, loss_term, loss, iteration):
        self.add_scalar(loss_term, loss, iteration)

    def log_feature_maps(self, term, feature_maps):
        channels = feature_maps.shape[1]
        norm = torch.clamp(feature_maps.max(), 1e-12)
        for i in range(channels):
            self.add_image(term + '_' + str(i), feature_maps[0, i:i+1, :, :] / norm)

def log(message, log_file):
    message_bytes = '{}\n'.format(message).encode(encoding='utf-8')
    log_file.write(message_bytes)
    print(message)


def center_crop(x, length: int):
    start = (x.shape[-1]-length)//2
    stop  = start + length
    return x[...,start:stop]

def causal_crop(x, length: int):
    stop = x.shape[-1] - 1
    start = stop - length
    return x[...,start:stop]
