import os
import gc
import torch
import importlib
import time
from utils import log, Logger
from torch import Tensor

from utils import center_crop, causal_crop
from loss_custom import Weight_BCELoss, SoftBCELoss, SoftBCELoss_new
from data import DownbeatDataset
from eval import evaluate, find_beats
from plot import make_table, plt_netout
import glob
import numpy as np
import scipy.stats as st

class Trainer(object):
    def __init__(self, config):    
        self.iftestmode = config.iftestmode
        self.h5_file_augment = True
        if self.iftestmode:
            self.h5_file_augment = False
            
        self.only_beat = config.only_beat
        self.only_downbeat = config.only_downbeat
                      
        # Path and file settings
        self.model_path = config.model_path
        self.network = config.network
        self.mark = config.mark        
        self.h5_dir = config.root_path
        self.folder = config.folder
        
        if self.folder == -1:
            self.subset = "full-train"
            self.folder = 0
        else:
            self.subset = "train"
            
        
        # Create logger
        self.log_file = config.log_file
        self.checkname = config.checkname
        self.ifcheckpoint = config.ifcheckpoint
        
        # set scalar writer 
        # tensorboard --logdir="./vis_scalar" --host=127.0.0.1
        scalar_writer_path = config.scalar_writer +'/%s-%s-%s' % (self.network, self.mark, str(self.folder))
        if not os.path.exists(scalar_writer_path):
            os.mkdir(scalar_writer_path)
        self.writer = Logger(scalar_writer_path)
        self.global_step = 0
        
        # set checkpoint path 
        self.net_rootpath = os.path.join(self.model_path, '%s-%s-%s'%(self.network, self.mark, str(self.folder)))
        if not os.path.exists(self.net_rootpath):
            os.mkdir(self.net_rootpath)

        # Training settings
        self.lr = config.lr
        self.device = torch.device(config.device, index=config.device_index)
        # self.device = "cpu"
        self.max_epochs = config.max_epochs 
        self.epoch_batch=config.epoch_batch
        self.patience = config.patience
        self.sr = config.sr
        
        self.batch_size = config.batch_size 
        self.optimizer = None
        self.clip_grad = config.clip_grad  
        self.causal = False            
       
        # target settings        
        self.hop_length = config.hop_length
        self.target_sr = int(self.sr/self.hop_length)
        self.target_factor = config.target_factor
        
        # length settings
        self.length_sec = config.length_sec
        self.target_length = config.target_length
        self.train_length = config.train_length
        self.eval_length = config.eval_length
        self.limit_length = config.limit_length

        # training sets settings                       
        self.train_sets = config.train_sets
        self.num_workers = config.num_workers
        self.shuffle = True
        
        # eval settings
        self.eval_mode = config.eval_mode
        self.inference_model = config.inference_model
        self.peak_type = config.peak_type       
        
        # loss settings
        self.loss = config.loss
        self.loss_param = config.loss_param
        
        self.kernel = torch.Tensor(self.get_kernel()).to(self.device)
        
        # setup the dataloaders
        self.configure_dataloader()     
        self.build_model()   
        self.configure_loss()
        self.configure_optimizers()
              
    def configure_dataloader(self):
        train_datasets = []
        val_datasets = []
        for dataset in self.train_sets:
            train_dataset = DownbeatDataset(h5_dir=self.h5_dir,
                                            dataset=dataset,
                                            audio_sample_rate=self.sr,
                                            target_factor=self.target_factor,
                                            # subset="train", TODO
                                            subset = self.subset,
                                            augment=False,
                                            folder=self.folder,
                                            length=self.train_length,
                                            device=self.device,
                                            rand = True,
                                            h5_file_augment=self.h5_file_augment,
                                            randseed = 28)
            train_datasets.append(train_dataset)

            val_dataset = DownbeatDataset(h5_dir=self.h5_dir,
                                        dataset=dataset,
                                        audio_sample_rate=self.sr,
                                        target_factor=self.target_factor,
                                        subset="val",
                                        augment=False,
                                        folder=self.folder,
                                        length=self.eval_length,
                                        device=self.device,
                                        rand = True,
                                        randseed = 28)
            val_datasets.append(val_dataset)
        train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset_list = torch.utils.data.ConcatDataset(val_datasets)

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset_list, 
                                                        shuffle=self.shuffle,
                                                        batch_size=self.batch_size,
                                                        num_workers=self.num_workers,
                                                        pin_memory=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset_list, 
                                                    shuffle=self.shuffle,
                                                    batch_size=1,
                                                    num_workers=self.num_workers,
                                                    pin_memory=False) 
        log('Creating loader for {} train file and {} val file'.format(len(train_dataset_list), len(val_dataset_list)), self.log_file)
        
    def get_kernel(self, kernlen=31, nsig=3):      
        interval = (2*nsig+1.)/kernlen 
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1) 
        kern1d = np.diff(st.norm.cdf(x))     
        return [kern1d]
    
    def configure_loss(self):
        # 所有的lossfunc返回值均有三个
        if self.loss =="weightloss":
            self.lossfunc = Weight_BCELoss()   
        if self.loss == "softbce": # not good
            self.lossfunc = SoftBCELoss()   
        if self.loss == "softbce_n":
            self.lossfunc = SoftBCELoss_new()
             
    def build_model(self):
        network_module = importlib.import_module('networks.' + self.network)
        self.net = network_module.MainModule()
        self.net.to(self.device)
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        # self.optimizer = torch.optim.RAdam(self.net.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "max", patience=20, factor=0.5)
        
    def reset_grad(self):
        # not sure
        self.optimizer.zero_grad()

    def validation_step(self, batch):
        audio, beat, tempo, chord, metadata, ifdownbeat = batch
          
        # cut the audio to several pieces to avoid memory out        
        if audio.shape[-1] % self.eval_length >= self.limit_length:
            valpiece_num = int(audio.shape[-1]/self.eval_length) + 1
            ifoverlap = 1
        else:
            valpiece_num = int(audio.shape[-1]/self.eval_length)
            ifoverlap = 0
        
        outputs = []
        audio = audio.to(self.device)  
        for i in range(valpiece_num):
            # crop the input audio signal
            if ifoverlap == 0 or i!=valpiece_num-1:
                audio_start = i*self.eval_length
                audio_stop  = audio_start+self.eval_length      
            else:
                audio_stop = audio.shape[-1]
                audio_start = audio_stop - self.eval_length 
            input_audio = audio[:,:,audio_start:audio_stop]
                              
            # pass the input through the network  
            with torch.no_grad():          
                pred = self.net(input_audio)
            
            # crop the target signals
            target_start = int(audio_start / self.target_factor) 
            if target_start + pred.shape[-1]>beat.shape[-1]:
                target_start = target_start-1
            target_stop = target_start + pred.shape[-1]
            target_beat = beat[:,:,target_start:target_stop]
        
            # compute the validation error using all losses
            target_beat = target_beat.to(self.device)
            
            # computing loss     
            # t_loss, b_loss, d_loss = self.lossfunc(pred, target_beat)
            t_loss, b_loss, d_loss = self.lossfunc(pred, target_beat, ifdownbeat)
            loss = t_loss
            if self.only_beat:
                loss = b_loss
            if self.only_downbeat:
                loss = d_loss 
            print('val_loss:{}'.format(loss), end="\r")
            loss_cpu = torch.tensor(loss.item()).unsqueeze(0)
            b_loss = torch.tensor(b_loss.item()).unsqueeze(0)
            d_loss = torch.tensor(d_loss.item()).unsqueeze(0)

            # remove the nan loss
            if np.isnan(loss_cpu.numpy().item()):
                continue      
            
            # move tensors to cpu for logging
            output = {
                "input" : input_audio.cpu(),
                "target_beat": target_beat.cpu(),
                "pred"  : pred.detach().cpu(),
                "Filename" : metadata['Filename'][0].decode("utf-8")+'_'+str(i),
                "Genre" : metadata['Genre'],
                "dataset" : metadata['dataset'],
                "Time signature" : metadata['Time signature'],
                "val loss" : loss_cpu,
                "beat_loss" : b_loss,
                "downbeat_loss" : d_loss,
            }
            outputs.append(output)
        return outputs

    def validation_epoch_end(self, validation_step_outputs, epoch):
        # flatten the output validation step dicts to a single dict
        outputs = {
            "input" : [],
            "target_beat" : [],
            "pred" : [],
            "Filename" : [],
            "Genre" : [],
            "dataset" : [],
            "Time signature" : [],
            "val loss" : [],
            "beat_loss" : [],
            "downbeat_loss" : []
            }

        metadata_keys = ["Filename", "Genre", "dataset", "Time signature"]

        for out in validation_step_outputs:
            for key, val in out.items():
                if key not in metadata_keys:
                    bs = val.shape[0]
                else:
                    bs = len(val)
                for bidx in np.arange(bs):
                    if key not in metadata_keys:
                        outputs[key].append(val[bidx,...])
                    else:
                        if key == "Filename":
                            outputs[key].append(val)
                        else:
                            outputs[key].append(val[bidx])

        # compute metrics 
        songs = []
        val_losses = []
        beat_losses = []
        downbeat_losses = []

        beat_f1_scores = []
        downbeat_f1_scores = []
        for idx in np.arange(len(outputs["input"])):
            t = outputs["target_beat"][idx].squeeze()
            p = outputs["pred"][idx].squeeze()
            f = outputs["Filename"][idx]
            g = outputs["Genre"][idx]
            d = outputs["dataset"][idx]
            s = outputs["Time signature"][idx]
            l = outputs["val loss"][idx]
            b_l = outputs["beat_loss"][idx]
            d_l = outputs["downbeat_loss"][idx]

            beat_scores, downbeat_scores = self.evaluator.process(p, t, self.target_sr)

            songs.append({
                "Filename" : f,
                "Genre" : g,
                "dataset": d,
                "Time signature" : s,
                "Beat F-measure" : beat_scores['F-measure'],
                "Downbeat F-measure" : downbeat_scores['F-measure'],
                "val loss" : l,
                "beat loss" : b_l,
                "downbeat loss" : d_l
            })

            val_losses.append(l.numpy().item())
            beat_losses.append(b_l.numpy().item())
            beat_f1_scores.append(beat_scores['F-measure'])
            if d != "smc":
                downbeat_losses.append(d_l.numpy().item())
                downbeat_f1_scores.append(downbeat_scores['F-measure'])
            
        val_loss = sum(val_losses)/len(val_losses)
        beat_loss = sum(beat_losses)/len(beat_losses)
        downbeat_loss = sum(downbeat_losses)/len(downbeat_losses)
        beat_f_measure = np.mean(beat_f1_scores)
        downbeat_f_measure = np.mean(downbeat_f1_scores)
        joint_f_measure = np.mean([beat_f_measure, downbeat_f_measure])
        
        self.lr_scheduler.step(joint_f_measure)
        # add metric to text logger
        log('Beat F-measure: {}'.format(beat_f_measure), self.log_file)
        log('Downbeat F-measure: {}'.format(downbeat_f_measure), self.log_file)
        log('Joint F-measure: {}'.format(joint_f_measure), self.log_file)
           
        # add metric to scalar
        self.writer.add_scalar('val_loss/val', val_loss, epoch)
        self.writer.add_scalar('beat_loss/val', beat_loss, epoch)
        self.writer.add_scalar('downbeat_loss/val', downbeat_loss, epoch)
        self.writer.add_scalar('Beat F-measure', beat_f_measure, epoch)
        self.writer.add_scalar('Downbeat F-measure', downbeat_f_measure, epoch)
        self.writer.add_scalar('Joint F-measure', joint_f_measure, epoch)
        
        eval_index = joint_f_measure
        if self.only_downbeat:       
            eval_index = downbeat_f_measure 
        elif self.only_beat:
            eval_index = beat_f_measure
            
        # save model every 5 epoch (changed) or when best epoch changed
        if eval_index > self.best_net_score or epoch % 5 == 0:
            if eval_index > self.best_net_score:
                self.best_net_score = eval_index
                self.best_epoch = epoch  
            net_epoch = self.net.state_dict()  # only save the state dict
            net_path = os.path.join(self.net_rootpath, '%s.pkl' %(str(epoch)))   
            torch.save({
                "epoch": epoch,
                "best_fmeasure": eval_index,
                "best_epoch": self.best_epoch,
                "state_dict": net_epoch,
                "optimizer": self.optimizer.state_dict(),
                'lr': self.lr_scheduler.state_dict()}, 
                net_path)
                 
        log("Epoch costing time: {}, best f measure: {}, best epoch: {}".format(time.time()-self.start, self.best_net_score, self.best_epoch), self.log_file)

    def load_checkpoint(self, checkfile):
        data = torch.load(checkfile)
        return data["state_dict"], data["optimizer"],  data["lr"], data["epoch"], data["best_epoch"],  data["best_fmeasure"]
    
    def load_checkpoint_former(self, checkfile):
        data = torch.load(checkfile)
        return data["epoch"], data["state_dict"], data["best_epoch"], data["best_fmeasure"]
    

        
    def train(self):
        #====================================== Training ===========================================#
        log('\n Training...', self.log_file)

        # loading checkpoint if it exists 
        self.begin_epoch = 0
        self.best_epoch = 0

        # loading checkpoint if it exists 
        self.checkfiles = glob.glob(os.path.join(self.net_rootpath, "*.pkl"))
        if len(self.checkfiles) != 0: # import the latest checkpoint file
            if self.ifcheckpoint == True:
                self.checkfiles.sort()
                if self.checkname == None: 
                    state_dict, optimizer, lr_sche, self.begin_epoch, self.best_epoch, self.best_net_score = self.load_checkpoint(self.checkfiles[-1])
                    
                else:
                    state_dict, optimizer, lr_sche, self.begin_epoch, self.best_epoch, self.best_net_score = self.load_checkpoint(os.path.join(self.net_rootpath, str(self.checkname)+".pkl"))
                    
                self.begin_epoch = self.begin_epoch + 1
                self.net.load_state_dict(state_dict)
                self.optimizer.load_state_dict(optimizer)
                self.lr_scheduler.load_state_dict(lr_sche)
                
                print('{%s} is Successfully Loaded from {%s}' % (self.network, self.net_rootpath))

        trainloader_len = len(self.train_dataloader)
        self.best_net_score = 0.

        # train in the unit of epoch
        for epoch in range(self.begin_epoch, self.max_epochs):   
            self.start = time.time()  # timing begin
            self.net.train(True)
            epoch_loss = 0.
            epoch_beat_loss = 0.
            epoch_downbeat_loss = 0.
            
            for i, data in enumerate(self.train_dataloader): 
                if self.iftestmode and i > self.epoch_batch:
                    break
                
                audio_wav, target_beat, _,_,_,ifdownbeat = data
 
                audio_wav = audio_wav.to(self.device)                
                # target_beat: ground truth
                target_beat = Tensor.float(target_beat)
                # target_beat = target_beat.to(self.device)
                
                # pred: classification result           
                pred = self.net(audio_wav)
                
                # crop the input and target signals
                if self.causal:
                    target = causal_crop(target_beat, pred.shape[-1])
                else:
                    target = center_crop(target_beat, pred.shape[-1])
                
                target = target.to(self.device)

                # t_loss, b_loss, d_loss = self.lossfunc(pred, target)
                t_loss, b_loss, d_loss = self.lossfunc(pred, target, ifdownbeat)
                
                self.reset_grad()
                # Backprop + optimize      
                loss = t_loss
                if self.only_beat:
                    loss = b_loss
                if self.only_downbeat:
                    loss = d_loss     
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.clip_grad)
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_beat_loss += b_loss.item()
                epoch_downbeat_loss += d_loss.item()
                
                # Print the log info
                print("[Training] Epoch-{} [{}/{}], Loss: {:.4f}".format(epoch+1, i, trainloader_len, loss), end='\r')
            
            epoch_loss = epoch_loss/trainloader_len
            epoch_beat_loss = epoch_beat_loss/trainloader_len
            epoch_downbeat_loss = epoch_downbeat_loss/trainloader_len

            self.writer.add_scalar('val_loss/train', epoch_loss, epoch)
            self.writer.add_scalar('beat_loss/train', epoch_beat_loss, epoch)
            self.writer.add_scalar('downbeat_loss/train', epoch_downbeat_loss, epoch)


            print('', end='\r')
            log("[Training] Epoch-{} [{}/{}], Loss: {:.4f}".format(epoch+1, i, trainloader_len, epoch_loss), self.log_file)


            #===================================== Validation ====================================# 
            self.net.train(False)
            self.net.eval()   
            self.evaluator = evaluate(mode=self.eval_mode, inference_model=self.inference_model, peak_type=self.peak_type, device=self.device)        
            
            validation_step_outputs = []
            for batch in self.val_dataloader: 
                outputs = self.validation_step(batch)
                for output in outputs:
                    validation_step_outputs.append(output)              
            self.validation_epoch_end(validation_step_outputs, epoch)
            
            if epoch - self.best_epoch > self.patience:
                log('Early Stopping.', self.log_file)
                gc.collect()
                break

        log('Stop training', self.log_file)
        gc.collect()
            