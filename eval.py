import queue
import madmom
import mir_eval
import numpy as np
import pandas as pd
import scipy.signal
import librosa.util as U
from plot import plt_eval
from param_file.params import *
# from particle_filtering_cascade1 import particle_filter_cascade


def basic_peak_finding(activation, win_size=21, threshold=0.2):  

    peaks = np.where(activation>threshold)[0]
    
    return peaks

def pf_dealt(pred):
    est_beats = pred[:,0]
    est_downbeats = pred[pred[:,1]==1., 0] 
    
    return est_beats, est_downbeats

def find_beats(t, p, 
                smoothing=127, 
                threshold=0.2, 
                distance=None, 
                sample_rate=44100, 
                beat_type="beat",
                filter_type="none",
                peak_type="simple",
                peak_latency=20):
    # 15, 2

    # t is ground truth beats
    # p is predicted beats 
    # 0 - no beat
    # 1 - beat

    N = p.shape[-1]

    if filter_type == "savgol":
        # apply smoothing with savgol filter
        p = scipy.signal.savgol_filter(p, smoothing, 2)   
    elif filter_type == "cheby":
        sos = scipy.signal.cheby1(10,
                                  1, 
                                  45, 
                                  btype='lowpass', 
                                  fs=sample_rate,
                                  output='sos')
        p = scipy.signal.sosfilt(sos, p)

    # normalize the smoothed signal between 0.0 and 1.0
    p /= np.max(np.abs(p))   
    
    # by default, we assume that the min distance between beats is fs/4
    # this allows for at max, 4 BPS, which corresponds to 240 BPM 
    # for downbeats, we assume max of 1 downBPS
    if beat_type == "beat":
        if distance is None:
            distance = sample_rate / 4
    elif beat_type == "downbeat":
        if distance is None:
            distance = sample_rate / 2
    else:
        raise RuntimeError(f"Invalid beat_type: `{beat_type}`.")                                  

    if peak_type == "simple":
        # apply simple peak picking(causal)
        est_beats, _ = scipy.signal.find_peaks(p, height=threshold, distance=distance)
        # np.savetxt("./test.txt", p)

    elif peak_type == "cwt":
        # use wavelets(应该是uncausal的)
        est_beats = scipy.signal.find_peaks_cwt(p, np.arange(1,50))
        
    elif peak_type == "librosa":
        # est_beats = U.peak_pick(p, pre_max=distance, post_max=1, pre_avg=distance, post_avg=1, delta=0.2, wait=distance)
        # est_beats = U.peak_pick(p, pre_max=20, post_max=20, pre_avg=distance, post_avg=5, delta=0.2, wait=distance)
        est_beats = U.peak_pick(p, pre_max=distance, post_max=peak_latency, pre_avg=distance, post_avg=peak_latency, delta=0.2, wait=distance)

    elif peak_type == "thresh":
        # est_beats = U.peak_pick(p, pre_max=distance, post_max=5, pre_avg=distance, post_avg=5, delta=0.1, wait=distance) # TODO
        est_beats = basic_peak_finding(p)

    # compute the locations of ground truth beats
    ref_beats = np.squeeze(np.argwhere(t==1).astype('float32'))
    est_beats = est_beats.astype('float32')
    
    # TODO: add a visualization
    # plt_eval(t, p, est_beats=est_beats, target_sample_rate=sample_rate)

    # compute beat points (samples) to seconds
    ref_beats /= float(sample_rate)
    est_beats /= float(sample_rate)

    # store the smoothed ODF
    est_sm = p 
    return ref_beats, est_beats, est_sm


class evaluate:
    def __init__(self, mode="online", inference_model="PF", peak_type="simple", peak_latency=5, thread=False, device="cpu", threshhold=0.2):    
        self.mode = mode # online, offline, stream
        self.inference_model = inference_model #  PF, DBN, 
        self.thread = thread
        self.peak_type = peak_type
        self.peak_latency = peak_latency
        self.threshhold = threshhold
        self.device = device
        self.fps = int(target_sr/hop_length)
        if self.inference_model == "PF":                 # instantiating a Particle Filter decoder - Is Chosen for online inference
            self.estimator = particle_filter_cascade(beats_per_bar=[], fps=self.fps, plot=[], mode=self.mode)
        elif self.inference_model == "DBN" and self.mode == "online":                # instantiating an HMM decoder - Is chosen for offline inference     
            self.beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
                min_bpm=55,
                max_bpm=215,
                # observation_lambda=6,
                transition_lambda=100,
                fps=self.fps,
                online = True,
                )

            self.downbeat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
                min_bpm=10,
                max_bpm=75,
                # observation_lambda=6,
                transition_lambda=100,
                fps=self.fps,
                online = True,
                )

        elif self.inference_model == "DBN" and self.mode == "offline": 
            self.beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
                min_bpm=55,
                max_bpm=215,
                # observation_lambda=6,
                transition_lambda=100,
                fps=self.fps,
                online=False,
                )

            self.downbeat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
                min_bpm=10,
                max_bpm=75,
                # observation_lambda=6,
                transition_lambda=100,
                fps=self.fps,
                online = False,
                )
        elif self.inference_model == "simple":
            self.inference_model = "simple"
            
    def process(self, pred, target, target_sample_rate):
        t_beats = target[0,:]
        t_downbeats = target[1,:]
        p_beats = pred[0,:]
        p_downbeats = pred[1,:]  

        if self.inference_model == "PF":   # Particle filtering inference (causal)
            self.estimator = particle_filter_cascade(beats_per_bar=[], fps=self.fps, plot=[], mode=self.mode, ig_threshold=0.2)
            pred = self.estimator.process(pred[:2, :].numpy().T)  # Using particle filtering online inference to infer beat/downbeats
            est_beats, est_downbeats = pf_dealt(pred)

        if self.inference_model == "simple":
            ref_beats, est_beats, _ = find_beats(t_beats.numpy(), 
                                                p_beats.numpy(), 
                                                beat_type="beat",
                                                sample_rate=target_sample_rate,
                                                threshold=self.threshhold,
                                                peak_type=self.peak_type,
                                                peak_latency=self.peak_latency) # default 

            ref_downbeats, est_downbeats, _ = find_beats(t_downbeats.numpy(), 
                                                        p_downbeats.numpy(), 
                                                        beat_type="downbeat",
                                                        sample_rate=target_sample_rate,
                                                        threshold=self.threshhold,
                                                        peak_type=self.peak_type,
                                                        peak_latency=self.peak_latency) # 

        elif self.inference_model == "DBN" and self.mode == "online":    # Dynamic bayesian Network 
            est_beats = self.beat_dbn.process_online(p_beats.numpy())
            est_downbeats = self.downbeat_dbn.process_online(p_downbeats.numpy())
                
        elif self.inference_model == "DBN" and self.mode == "offline":
            est_beats = self.beat_dbn.process_offline(p_beats.numpy())
            est_downbeats = self.downbeat_dbn.process_offline(p_downbeats.numpy())
        elif self.inference_model == "dl":
            ref_beats, est_beats, _ = find_beats(t_beats.numpy(), 
                                                p_beats.numpy(), 
                                                beat_type="beat",
                                                sample_rate=target_sample_rate,
                                                threshold=self.threshhold,
                                                peak_type="thresh") # default 

            ref_downbeats, est_downbeats, _ = find_beats(t_downbeats.numpy(), 
                                                        p_downbeats.numpy(), 
                                                        beat_type="downbeat",
                                                        sample_rate=target_sample_rate,
                                                        threshold=self.threshhold,
                                                        peak_type="thresh") # 
                
         
        ref_beats = np.squeeze(np.argwhere(t_beats.numpy()==1).astype('float32'))
        ref_beats /= float(target_sample_rate)
        ref_downbeats = np.squeeze(np.argwhere(t_downbeats.numpy()==1).astype('float32'))
        ref_downbeats /= float(target_sample_rate)
        
        # # evaluate beats - trim beats before 5 seconds.
        beat_scores = mir_eval.beat.evaluate(ref_beats, est_beats)

        # # evaluate downbeats - trim beats before 5 seconds.
        downbeat_scores = mir_eval.beat.evaluate(ref_downbeats, est_downbeats)

        return beat_scores, downbeat_scores 