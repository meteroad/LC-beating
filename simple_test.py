import os
import threading
import time
# import yaml
import sys
import glob
import json
import torch
import numpy as np
from tqdm import tqdm
import importlib
from data import DownbeatDataset
import torch.nn.functional as F
from eval import evaluate
from plot import  make_table, plt_netout
import madmom
from scipy.ndimage.filters import maximum_filter1d, uniform_filter1d
from particle_filtering_cascade import particle_filter_cascade

# torch.backends.cudnn.benchmark = True

class test_func():
    def __init__(self, config):
        self.evaluator1 = evaluate(mode=config.eval_mode, inference_model=config.inference_model, peak_type=config.peak_type, peak_latency=config.peak_latency, device=config.device)
        self.mode = config.eval_mode
        self.inference_model = config.inference_model
        self.peak_type = config.peak_type
        self.peak_latency = config.peak_latency
        
        # self.mode = config.eval_mode
        self.datasets = config.test_sets
        self.network = config.network
        self.pp_network = config.pp_network
        self.mark = config.mark
        self.mark_pp = config.mark_pp
        self.root_path = config.root_path
        self.only_beat = config.only_beat
        self.num_workers = config.num_workers
        self.eval_length = config.eval_length
        self.folder = config.folder
        self.target_factor = config.target_factor
        self.sr = config.sr  
        self.hop_length = config.hop_length  
        self.fps = int(self.sr/self.hop_length)
        
        self.check_num = config.check_num
        
        
        self.thread = False
        
        # add PROGRAM level args
        checkpoint_path = os.path.join(config.model_path, config.network+'-'+config.mark+'-'+str(self.folder))
        checkpoint_path_pp = os.path.join(config.pp_model_path, config.pp_network+'-'+config.mark_pp+'-'+config.network+'-'+config.mark)
        
        self.device = torch.device(config.device, index=config.device_index) if config.device == "cuda" else "cpu"
        # device = "cpu"
        self.target_sr = int(config.sr/config.hop_length)

        # find the checkpoint path
        if config.check_num != "0":
            self.checkpoint_file = os.path.join(checkpoint_path, config.check_num+'.pkl')
        else:
            self.checkpoint_file =  self.findlast(checkpoint_path)
            
        self.checkpoint_file_pp = os.path.join(checkpoint_path_pp, config.check_num_pp+'.pkl')

        # load checkpoint
        network_module = importlib.import_module('networks.' + config.network)
        # net = network_module.MainModule(device=device)
        self.net = network_module.MainModule()
        self.net.to(self.device)   
        if config.mode == "real_time":           
            data = torch.load(self.checkpoint_file, map_location="cpu") 
        else:
            data = torch.load(self.checkpoint_file, map_location=self.device)
        self.net.load_state_dict(data["state_dict"])
        if config.mode == "real_time":
            self.net.cpu()
        self.net.eval()
        print("checkfile {} successfully loaded".format(self.checkpoint_file))
        
        if config.inference_model == "dl":
            pp_network_module = importlib.import_module('networks_pp.' + self.pp_network)
            self.net_pp = pp_network_module.MainModule()
            self.net_pp.to(self.device)
            data_pp = torch.load(self.checkpoint_file_pp, map_location=self.device)
            self.net_pp.load_state_dict(data_pp["state_dict"])
            self.net_pp.eval()  
            print("checkfile {} successfully loaded".format(self.checkpoint_file_pp))
        
        if self.mode == "online":
            if self.inference_model == "DBN":
                self.estimator_b = madmom.features.beats.DBNBeatTrackingProcessor(
                    min_bpm=55,
                    max_bpm=215,
                    observation_lambda=6,
                    transition_lambda=100,
                    fps=self.fps,
                    online = True,
                    )
                self.estimator_d = madmom.features.beats.DBNBeatTrackingProcessor(
                    min_bpm=10,
                    max_bpm=75,
                    observation_lambda=6,
                    transition_lambda=100,
                    fps=self.fps,
                    online = True,
                    )
            elif self.inference_model == "simple":
                distance_b = self.fps/4
                distance_d = self.fps/2
                self.estimator_b = simple_findpeak(pre_max=distance_b, post_max=1, pre_avg=distance_b, post_avg=1, delta=0.2, wait=distance_b)
                self.estimator_d = simple_findpeak(pre_max=distance_d, post_max=1, pre_avg=distance_d, post_avg=1, delta=0.2, wait=distance_d)
                
            # elif self.inference_model == "dl":
            #     self.estimator = 
        
    def calculate_rtf(self):
        rtf_total = []
        for dataset in self.datasets: 
            if dataset == "ballroom" or dataset == "beatles" or dataset == "hainsworth" or dataset == "smc" or dataset == "rwc_popular" or dataset == "hjdb" or dataset == "beatles2":# seen datasets
                subset = "test"
            else: # unseen datasets
                subset = "full-val"
        
            test_dataset = DownbeatDataset(self.root_path,
                                    dataset=dataset,
                                    audio_sample_rate=self.sr,
                                    target_factor=self.target_factor,
                                    subset=subset,
                                    folder=self.folder,
                                    augment=False,
                                    length=self.eval_length)
            
            self.test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                            shuffle=False,
                                                            batch_size=1,
                                                            num_workers=0,
                                                            pin_memory=False)
            
            # net = self.net.to("cpu")
            
            for example in tqdm(self.test_dataloader, ncols=80):
                audio, _, _, _, _,_ = example
                audio = audio.cpu()
                            
                start = time.time()
            
                # pass the input through the network 
                with torch.no_grad():           
                    pred = self.net(audio)
                
                pred = pred.cpu().numpy()
                pred = pred[0,:,:]
                if self.inference_model == "simple":
                    self.estimator_b.reset()
                    self.estimator_d.reset()  
                if not self.inference_model == "PF":   
                    est_beats = self.estimator_b.process_online(pred[0,:])
                    est_downbeats = self.estimator_d.process_online(pred[1,:])
                else:
                    self.estimator = particle_filter_cascade(beats_per_bar=[], fps=self.fps, plot=[], mode=self.mode, ig_threshold=0.2)
                    est_beats, est_downbeats = self.estimator.process(pred[:2, :].T)
                                        
                rtf_total.append([audio.shape[-1]/self.sr, time.time()-start])  
            rtf_total = np.array(rtf_total)
            rtf_total = np.sum(rtf_total, axis=0)
            rtf = rtf_total[-1]/rtf_total[0]
            print("the rtf value is {}".format(rtf)) 
                

    def test(self):    
        # set model to eval mode        
        results = {} # storage for our result metrics
        result_str = ""
      
        # evaluate on each dataset using the test set
        for dataset in self.datasets: 
            if dataset == "ballroom" or dataset == "beatles" or dataset == "hainsworth" or dataset == "smc" or dataset == "rwc_popular" or dataset == "hjdb" or dataset == "beatles2":# seen datasets
                subset = "test"
            else: # unseen datasets
                subset = "full-val"
        
            test_dataset = DownbeatDataset(self.root_path,
                                    dataset=dataset,
                                    audio_sample_rate=self.sr,
                                    target_factor=self.target_factor,
                                    subset=subset,
                                    folder = self.folder,
                                    augment=False,
                                    length=self.eval_length)
            
            self.test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                            shuffle=False,
                                                            batch_size=1,
                                                            num_workers=self.num_workers,
                                                            pin_memory=True)

            # setup tracking of metrics
            results[dataset] = {
                "F-measure" : {
                    "beat" : [],
                    "dbn beat" : [],
                    "downbeat" : [],
                    "dbn downbeat" : [],
                },
                "CMLt" : {
                    "beat" : [],
                    "dbn beat" : [],
                    "downbeat" : [],
                    "dbn downbeat" : [],
                },
                "AMLt" : {
                    "beat" : [],
                    "dbn beat" : [],
                    "downbeat" : [],
                    "dbn downbeat" : [],
                }
            }
                    
            for example in tqdm(self.test_dataloader, ncols=80):
                audio, target_beat, _, _, metadata,_ = example
                
                audio = audio.to(self.device)
                
                beat_scores_F1 = []
                downbeat_scores_F1 = []
                dbn_beat_scores_F1 = []
                dbn_downbeat_scores_F1 = []
                beat_scores_CMLt = []
                downbeat_scores_CMLt = []
                dbn_beat_scores_CMLt = []
                dbn_downbeat_scores_CMLt = []
                beat_scores_AMLt = []
                downbeat_scores_AMLt = []
                dbn_beat_scores_AMLt = []
                dbn_downbeat_scores_AMLt = []
                                
                # pass the input through the network 
                with torch.no_grad():           
                    pred = self.net(audio)
                    if self.inference_model == "dl":
                        pred = self.net_pp(pred)
                    
                pred = pred.to("cpu")
                                   
                if self.only_beat:
                    pred = torch.cat([pred.view(1,-1), pred.view(1,-1)], 0)
                else:
                    pred = pred[:,:2,:].view(2,-1)
            
              
                beat_scores_t, downbeat_scores_t = self.evaluator1.process(
                                                        pred.view(2,-1), 
                                                        target_beat[:,:2,:].view(2,-1),  
                                                        self.target_sr)

                dbn_beat_scores_t, dbn_downbeat_scores_t = beat_scores_t, downbeat_scores_t                                       
                                                        
                # beat/downbeat F1
                beat_scores_F1 = beat_scores_t['F-measure']
                downbeat_scores_F1 = downbeat_scores_t['F-measure']

                # beat/downbeat CMLt
                beat_scores_CMLt = beat_scores_t['Correct Metric Level Total']
                downbeat_scores_CMLt = downbeat_scores_t['Correct Metric Level Total']

                # beat/downbeat AMLt
                beat_scores_AMLt = beat_scores_t['Any Metric Level Total']
                downbeat_scores_AMLt = downbeat_scores_t['Any Metric Level Total']

                # dbn beat/downbeat F1
                dbn_beat_scores_F1 = dbn_beat_scores_t['F-measure']
                dbn_downbeat_scores_F1 = dbn_downbeat_scores_t['F-measure']

                # dbn beat/downbeat CMLt
                dbn_beat_scores_CMLt = dbn_beat_scores_t['Correct Metric Level Total']
                dbn_downbeat_scores_CMLt = dbn_downbeat_scores_t['Correct Metric Level Total']

                # dbn beat/downbeat AMLt
                dbn_beat_scores_AMLt = dbn_beat_scores_t['Any Metric Level Total']
                dbn_downbeat_scores_AMLt = dbn_downbeat_scores_t['Any Metric Level Total']
                

                results[dataset]['F-measure']['beat'].append(beat_scores_F1)
                results[dataset]['CMLt']['beat'].append(beat_scores_CMLt)
                results[dataset]['AMLt']['beat'].append(beat_scores_AMLt)

                results[dataset]['F-measure']['dbn beat'].append(dbn_beat_scores_F1)
                results[dataset]['CMLt']['dbn beat'].append(dbn_beat_scores_CMLt)
                results[dataset]['AMLt']['dbn beat'].append(dbn_beat_scores_AMLt)

                results[dataset]['F-measure']['downbeat'].append(downbeat_scores_F1)
                results[dataset]['CMLt']['downbeat'].append(downbeat_scores_CMLt)
                results[dataset]['AMLt']['downbeat'].append(downbeat_scores_AMLt)

                results[dataset]['F-measure']['dbn downbeat'].append(dbn_downbeat_scores_F1)
                results[dataset]['CMLt']['dbn downbeat'].append(dbn_downbeat_scores_CMLt)
                results[dataset]['AMLt']['dbn downbeat'].append(dbn_downbeat_scores_AMLt)
                
                print("\n")
                print("filename: {}, Genre: {}".format(metadata["Filename"][0], metadata["Genre"][0]))
                print(f"beat {beat_scores_F1:0.3f} mean: {(np.mean(results[dataset]['F-measure']['beat'])):0.3f}  ")
                print(f"downbeat: {downbeat_scores_F1:0.3f} mean: {(np.mean(results[dataset]['F-measure']['downbeat'])):0.3f}")
                
            # append average level
            result_dataset = str(self.folder) +"-"+ f"{dataset} \n" + f"F1 beat: {np.mean(results[dataset]['F-measure']['beat'])}   F1 downbeat: {np.mean(results[dataset]['F-measure']['downbeat'])} \n" + f"CMLt beat: {np.mean(results[dataset]['CMLt']['beat'])}   CMLt downbeat: {np.mean(results[dataset]['CMLt']['downbeat'])} \n" + f"AMLt beat: {np.mean(results[dataset]['AMLt']['beat'])}   AMLt downbeat: {np.mean(results[dataset]['AMLt']['downbeat'])} \n"
            result_str = result_str + result_dataset
            print(result_dataset)

        results_dir = 'results/'+self.network+'-'+self.mark+ '_' + self.check_num+ '_' +self.mode+'_' + self.inference_model+'_' + self.peak_type +'_' +str(self.peak_latency)+'_' +str(self.folder)+'.txt'
        with open(results_dir, 'a') as txt_file:
            txt_file.write(result_str)

    def findlast(self, input_path, file_ext='*.pkl'):
        return max(glob.glob(os.path.join(input_path, file_ext)), key=os.path.getmtime)
    
    def real_time(self, audio):
        self.counter = 0
        self.completed = 0
        if isinstance(audio, str) or audio.all()!=None:
            if self.inference_model == "simple":
                self.estimator_b.reset()
                self.estimator_d.reset()
            while self.completed == 0:
                
                self.activation_extractor_realtime(audio) # Using BeatNet causal Neural network realtime mode to extract activations
                
                if self.thread:
                    x = threading.Thread(target=self.estimator_b.process_online, args=(self.pred), daemon=True)   # Processing the inference in another thread 
                    x.start()
                    x.join()    
                else:
                    # test = time.time()
                    est_beats = self.estimator_b.process_online(self.pred[0,:], reset=False)  # Using particle filtering online inference to infer beat/downbeats, input [L]
                    est_downbeats = self.estimator_d.process_online(self.pred[1,:], reset=False)
                    
                    # print(time.time()-test)
                    print(self.counter)
                    
                self.counter += 1
            return est_beats, est_downbeats
        else:
            raise RuntimeError('An audio object or file directory is required for the realtime usage!')

    def activation_extractor_realtime(self, audio):
        with torch.no_grad():
            if self.counter==0: #loading the audio
                self.audio = audio
            if self.counter<(round(self.audio.shape[-1]/self.hop_length)):
                if self.counter<2:
                    self.pred = np.zeros([2,1])
                else:
                    input_audio = self.audio[:,:,self.hop_length * 0:self.hop_length * (self.counter+1) + self.hop_length]
                    # feats = torch.from_numpy(feats)
                    # feats = feats.unsqueeze(0).unsqueeze(0).to(self.device)
                    self.pred = self.net(input_audio)
                    self.pred = self.pred[0,:,:].cpu().numpy()
            else:
                self.completed = 1
                
class simple_findpeak():
    
    def __init__(self, pre_max=10, pre_avg=10, delta=0.2, wait=10, post_max=1, post_avg=1):
        self.pre_max = int(pre_max)
        self.pre_avg = int(pre_avg)
        self.delta = delta
        self.wait = int(wait)
        self.post_max = int(post_max)
        self.post_avg = int(post_avg)
        self.max_length = self.pre_max + self.post_max
        self.max_origin = np.ceil(0.5 * (self.pre_max - self.post_max))
        self.avg_length = self.pre_avg + self.post_avg
        self.avg_origin = np.ceil(0.5 * (self.pre_avg - self.post_avg))
        
    def reset(self):
        self.last_onset = -np.inf  
        
    def process_online(self, activations, **kwargs):       
        x = activations
        # Using mode='constant' and cval=x.min() effectively truncates
        # the sliding window at the boundaries
        
        mov_max = maximum_filter1d(
            x, int(self.max_length), mode="constant", origin=int(self.max_origin), cval=x.min()
        )
           
        # Here, there is no mode which results in the behavior we want,
        # so we'll correct below.
        mov_avg = uniform_filter1d(
            x, int(self.avg_length), mode="nearest", origin=int(self.avg_origin)
        )    

        # Correct sliding average at the beginning
        n = 0
        # Only need to correct in the range where the window needs to be truncated
        
        while n - self.pre_avg < 0 and n < x.shape[0]:
            # This just explicitly does mean(x[n - pre_avg:n + post_avg])
            # with truncation
            start = n - self.pre_avg
            start = start if start > 0 else 0
            mov_avg[n] = np.mean(x[start : n + self.post_avg])
            n += 1
        
        # Correct sliding average at the end
        n = x.shape[0] - self.post_avg
        # When post_avg > x.shape[0] (weird case), reset to 0
        n = n if n > 0 else 0
        
        while n < x.shape[0]:
            start = n - self.pre_avg
            start = start if start > 0 else 0
            mov_avg[n] = np.mean(x[start : n + self.post_avg])
            n += 1
        
        # First mask out all entries not equal to the local max
        detections = x * (x == mov_max)

        # Then mask out all entries less than the thresholded average
        detections = detections * (detections >= (mov_avg + self.delta))

        # Initialize peaks array, to be filled greedily
        peaks = []
     
        # Remove onsets which are close together in time
        for i in np.nonzero(detections)[0]:
            # Only report an onset if the "wait" samples was reported
            if i > self.last_onset + self.wait:
                peaks.append(i)
                # Save last reported onset
                self.last_onset = i
        
        return np.array(peaks)
    