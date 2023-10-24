import numpy as np
import pyaudio
import torch
import threading
from nnAudio import Spectrogram
import librosa
from collections import deque

class LCBeating():
    def __init__(self, model, mode="stream", post_processing="SF", device="cpu", latency=47, sample_rate=16000, thread=False):
        if model == None:
            self.model = self.test_model
        else:
            self.model = model
        self.mode = mode
        self.device = device
        self.latency = latency
        
        self.post_processing = post_processing
        self.thread = thread
        self.sample_rate = sample_rate
        
        # stft params    
        octave_num= 9
        bins_per_o = 9
        fmin = 16 # C2
        n_fft = 512
        hop_length = 160
        fmax = fmin * (2 ** octave_num) # C8
        freq_bins = octave_num * bins_per_o
        
        self.log_spec_hop_length = int(hop_length)
        self.log_spec_win_length = int(n_fft)
        
        self.proc = Spectrogram.STFT(n_fft=n_fft, 
                                  freq_bins=freq_bins,
                                  hop_length=hop_length,
                                  freq_scale='log',
                                  fmin=fmin,
                                  fmax=fmax,
                                  output_format='Magnitude')
        
        # create streaming feature cache
        audio_list = [0] * int(30 * hop_length)
        self.audio_seq = deque(audio_list)
        
        if self.mode == 'stream':
            self.stream_window = np.zeros(self.log_spec_win_length + 2 * self.log_spec_hop_length, dtype=np.float32)                                          
            self.stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32,
                                             channels=1,
                                             rate=self.sample_rate,
                                             input=True,
                                             frames_per_buffer=self.log_spec_hop_length,) # linux not supported
            
    def test_model(self, audio_feature, index):
        return self.audio_seq[index-self.latency]
    
    def activation_extractor_stream(self):
        with torch.no_grad():
            hop = self.stream.read(self.log_spec_hop_length)
            hop = np.frombuffer(hop, dtype=np.float32)
            self.stream_window = np.append(self.stream_window[self.log_spec_hop_length:], hop)
            
            if self.counter < 5:
                self.pred = np.zeros([1, 2])
            else:
                feats = self.model(self.stream_window)
                self.pred = self.post_processing(feats)
                print("1")
                
                
    def activation_extractor_realtime(self, audio_path):
        with torch.no_grad():
            if self.counter==0: #loading the audio
                self.audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading the data
            
            # 
            if self.counter<(round(len(self.audio)/self.log_spec_hop_length)):
                if self.counter<2:
                    self.pred = np.zeros([1,2])
                else:
                    feats = self.proc(self.audio[self.log_spec_hop_length * (self.counter-2):self.log_spec_hop_length * (self.counter) + self.log_spec_win_length]).T[-1]
                    feats = torch.from_numpy(feats)
                    feats = feats.unsqueeze(0).unsqueeze(0).to(self.device)
                    pred = self.model(feats)[0]
                    pred = self.model.final_pred(pred)
                    pred = pred.cpu().detach().numpy()
                    self.pred = np.transpose(pred[:2, :])
            else:
                self.completed = 1
     
    def process(self, audio_path=None):
        if self.mode == "stream":
            self.counter = 0
            while self.stream.is_active():
                self.activation_extractor_stream()  # Using BeatNet causal Neural network streaming mode to extract activations
                if self.thread:
                    x = threading.Thread(target=self.estimator.process, args=(self.pred), daemon=True)   # Processing the inference in another thread 
                    x.start()
                    x.join() 
                else:
                    output = self.estimator.process(self.pred)       
                self.counter += 1
                
        # real time mode is avalable
        elif self.mode == "realtime":
            self.counter = 0
            self.completed = 0
            if self.post_processing != "SF":
                raise RuntimeError('The inference model for the streaming mode should be set to "SF".')
            if isinstance(audio_path, str) or audio_path.all()!=None:
                while self.completed == 0:
                    self.activation_extractor_realtime(audio_path)  # TODO: 慢！！
                    if self.thread:
                        x = threading.Thread(target=self.estimator.process, args=(self.pred), daemon=True)   # Processing the inference in another thread 
                        x.start()
                        x.join()    
                    else:
                        output = self.estimator.process(self.pred)  # Using particle filtering online inference to infer beat/downbeats
                    self.counter += 1
                return output
            else:
                raise RuntimeError('An audio object or file directory is required for the realtime usage!')
        
                
if __name__ == "__main__":
    block = LCBeating(model=None, mode="realtime")
    output = block.process(audio_path="test.wav")
    
    
        
        