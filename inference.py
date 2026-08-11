import numpy as np
import pyaudio
import torch
import threading
from nnAudio import Spectrogram
from network_infer import lc42_inference
import librosa
import matplotlib.pyplot as plt

"""
real-time mode: Reads an audio file chunk by chunk, and processes each chunk at the time.
stream mode: Uses the system microphone to capture sound and does the process in real-time. Due to training the model on standard mastered songs, it is highly recommended to make sure the microphone sound is as loud as possible. Less reverbrations leads to the better results.
online mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then infers the parameters on interest using particle filtering.
"""

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
        n_fft = 160
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
        
        self.model = model(channels=20)
        self.pred_beat = np.array([])
        self.pred_downbeat = np.array([])
        
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
           
    # TODO: 
    def activation_extractor_realtime(self, audio_path):
        with torch.no_grad():
            if self.counter==0: #loading the audio
                self.audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading the data
            
            # extract activation function
            if self.counter<(round(len(self.audio)/self.log_spec_hop_length)):
                audio_seq = self.audio[self.log_spec_hop_length * (self.counter):self.log_spec_hop_length * (self.counter) + self.log_spec_win_length]
                audio_seq = torch.tensor(audio_seq)
                audio_seq = audio_seq.unsqueeze(0).unsqueeze(0)
                feats = self.proc(audio_seq)
                feats = torch.mean(feats, dim=-1)
                feats = feats.unsqueeze(0)
                pred = self.model.inference_by_frame(feats)
                pred = pred.numpy()
                self.pred_beat = np.concatenate((self.pred_beat, np.array([pred[0]])))
                self.pred_downbeat = np.concatenate((self.pred_downbeat, np.array([pred[1]])))
            else:
                self.completed = 1
     
    def process(self, audio_path=None):
        if self.mode == "stream":
            self.counter = 0
            while self.stream.is_active():
                self.activation_extractor_stream()  # Using lc_beating causal mode to extract activations
                if self.thread:
                    x = threading.Thread(target=self.estimator.process, args=(self.pred), daemon=True)   # Processing the inference in another thread 
                    x.start()
                    x.join() 
                else:
                    output = self.estimator.process(self.pred)       
                self.counter += 1
                
        # real time mode is available
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
                        # output = self.estimator.process(self.pred)  # Using simple peak finding method to post process the result of self.pred
                        output = self.pred_beat
                    self.counter += 1
                return output
            
            else:
                raise RuntimeError('An audio object or file directory is required for the realtime usage!')
            
        elif self.mode == "online":
            print("online mode is not available now")
        
                
if __name__ == "__main__":
    block = LCBeating(model=lc42_inference, mode="realtime")
    output = block.process(audio_path="test.wav")
    print("1")
    
    # plotting part
    x = np.arange(1000)
    y = output[5000:6000]

    plt.plot(x, y)

    plt.title("visualization")
    plt.xlabel("t")
    plt.ylabel("activation")

    # save fig
    plt.savefig("visualization.png")
        
    
        
        