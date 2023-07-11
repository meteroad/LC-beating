import os
import threading
from importlib_metadata import metadata
import torch 
from param_file.params import *
import random
import torchaudio
import numpy as np
from torch_time_stretch import *
import torchaudio
import json

import torch.nn.functional as F
import time
import glob
import h5py
# from plot import plt_netout
# from add_beeping import beeping
torchaudio.set_audio_backend("soundfile")

"""
返回beat、downbeat、non_beat三个数组
"""

class DownbeatDataset(torch.utils.data.Dataset):
    """ Downbeat Dataset. """
    def __init__(self, 
                 h5_dir, 
                 audio_sample_rate=16000, 
                 target_factor=32, # 应该和hop_length是相同的
                 dataset="ballroom",
                 folder = 0,
                 subset="train", 
                 length=160000, 
                 augment=False,
                 pad_mode='constant',
                 device="cpu",
                 rand = True,
                 h5_file_augment = False,
                 randseed = 16):      
        self.audio_sample_rate = audio_sample_rate
        self.target_factor = target_factor
        self.target_sample_rate = audio_sample_rate / target_factor
        self.subset = subset
        self.dataset = dataset
        self.audio_dir = os.path.join(h5_dir, dataset)
        self.json_file = os.path.join(h5_dir, "dict", "8_fold_h5.json")
        self.folder = folder
        self.length = length
        self.augment = augment
        self.pad_mode = pad_mode
        self.device = device

        self.target_length = int(self.length / self.target_factor)+1
        self.inst_len = self.length/self.audio_sample_rate

        self.h5_files = []
        h5_ext = ".h5"

        # get all of the annotation files
        if dataset == "gtzan":
            self.former_h5_files = glob.glob(os.path.join(self.audio_dir, "**", "*" + h5_ext))
            if len(self.former_h5_files) == 0: # try from the root audio dir
                self.former_h5_files = glob.glob(os.path.join(self.audio_dir, "*" + h5_ext))
        else:
            train_list, val_list, test_list = self.get_files()
             
            # eight folder validation
            if self.subset == "train":
                self.former_h5_files = train_list
            elif self.subset == "val":
                self.former_h5_files = val_list
            elif self.subset == "test":
                self.former_h5_files = test_list
            elif self.subset in ["full-train", "full-val"]:
                self.former_h5_files = train_list + val_list + test_list

        if h5_file_augment:   
            self.h5_file_augmentation()
        else:
            self.h5_files = self.former_h5_files

        if rand:
            random.seed(randseed)
            random.shuffle(self.h5_files) # shuffle them

        print(f"Selected {len(self.h5_files)} files for {self.subset} set from {self.dataset} dataset.")

    def get_files(self):
        with open(self.json_file, 'r') as jf:
            json_dict = json.load(jf)
        
        val_num = (self.folder+5) % 8
        test_num = (self.folder+6) % 8
        train_list = []
        val_list = []
        test_list = []
        dataset_files = json_dict[self.dataset]
        for i in range(8):
            if i == val_num:
                val_list = dataset_files[i]
            elif i == test_num:
                test_list = dataset_files[i]
            else:
                for file in dataset_files[i]:
                    train_list.append(file)
        return train_list, val_list, test_list

    def __len__(self):
        length = len(self.h5_files)
        return length

    def __getitem__(self, idx):
        # 创建迭代器
        # start_time = time.time()   
        # get metadata of example
        h5_file = self.h5_files[idx % len(self.h5_files)]     
        # print(h5_file)           
        audio, target_beat, target_tempo, target_chord, metadata = self.load_data(h5_file)
                 
        # do all processing in float32 not float16
        audio = audio.float()
        target_beat = target_beat.float()
               
        # calculate time stretch factor for 
        scale_factor = 1.0    
        if self.augment: 
            scale_factor = np.random.normal(1.0, 0.5)  # 正态分布
            scale_factor = np.clip(scale_factor, a_min=0.6, a_max=1.48)   
                    
        N_audio = audio.shape[-1]   # audio samples
        
        # -- remove the data if doesn't fit the condition -- #
        if N_audio < int(scale_factor*self.length):
            # del self.audio_path[index]
            new_i = torch.randint(0, self.__len__(), (1,))
            new_i = new_i[0]
            return self.__getitem__(new_i)
        
        # random crop of the audio and target if larger than desired
        if (N_audio > self.length*scale_factor) and self.subset not in ['val', 'test', 'full-val']:
            if int(N_audio -self.length*scale_factor) - 1== 0:
                audio_start = 0
            else:  
                audio_start = random.randint(0, int(N_audio -self.length*scale_factor) -1)  
                # audio_start = 0  
                  
            audio_stop  = int(audio_start + self.length*scale_factor)
            target_start = int(audio_start / self.target_factor)
            target_stop = int(audio_stop / self.target_factor)
            audio_cut = audio[:,audio_start:audio_stop]
            target_beat_c = target_beat[:,target_start:target_stop] 
            target_beat = target_beat_c 
            audio = audio_cut                  
        
        # pad the audio and target if is shorter than desired
        if audio.shape[-1] < int(self.length*scale_factor) and self.subset not in ['val', 'test', 'full-val']: 
            pad_size = int(self.length*scale_factor - audio.shape[-1])
            padl = pad_size - int(pad_size / 2)
            padr = int(pad_size / 2)
            audio = F.pad(audio, (padl, padr), mode=self.pad_mode)
        if target_beat.shape[-1] < int(self.target_length*scale_factor) and self.subset not in ['val', 'test', 'full-val']: 
            pad_size = int(self.target_length*scale_factor - target_beat.shape[-1])
            padl = pad_size - int(pad_size / 2)
            padr = int(pad_size / 2)
            target_beat = F.pad(target_beat, (padl, padr))
            
        # do augmentation
        if self.augment: 
            audio, target_beat, target_tempo, target_chord = self.apply_augmentations(audio, target_beat, target_tempo, target_chord, scale_factor=scale_factor)
        
        if audio.shape[-1] > self.length and self.subset not in ['val', 'test', 'full-val']:
            audio = audio[:,:self.length] 
        
        ifdownbeat = 1
        no_beat = 1-target_beat[0,:]
        target_beat = torch.cat([target_beat, no_beat.unsqueeze(0)],dim=0)
        if metadata["dataset"] == "smc":
            ifdownbeat = 0
        return audio, target_beat, target_tempo, target_chord, metadata, ifdownbeat
        
    def h5_file_augmentation(self):
        if self.dataset == "hainsworth" or self.dataset =="hjdb" or self.dataset == "beatles" or self.dataset =="smc" or self.dataset == "beatles2":
            for path in self.former_h5_files:
                self.h5_files.extend([path]*3)
        if self.dataset == "rwc_popular":
            for path in self.former_h5_files:
                self.h5_files.extend([path]*6)   
        if self.dataset == "ballroom":
            for path in self.former_h5_files:
                self.h5_files.append(path)  
        print("dataset {} successfully augmented".format(self.dataset))    

    def load_data(self, h5_file):
        # first load the audio file       
        audio, sr, beat_samples, beat_sec, downbeat_samples, downbeat_sec, tempo, chord_samples, filename, Genre, Time_signature = self.get_h5data(h5_file)
         
        # transfer to tensor
        audio = torch.from_numpy(audio)
        
        # transfer stereo to mono if needed
        dim = audio.shape
        if dim[0] != 1:
            audio = (audio[0] + audio[1]) / 2
            audio = audio.unsqueeze(0)
        
        # normalize all audio inputs -1 to 1
        audio /= audio.abs().max()

        t = audio.shape[-1]/self.audio_sample_rate # audio length in sec
        N = int(t * self.target_sample_rate) + 1   # audio length in samples
        target_beat = torch.zeros(2,N)  
        if tempo != None:  
            if type(tempo) is bytes:
                tempo = int(float(tempo.decode('utf-8'))) 
                target_tempo = torch.tensor(tempo)
            else: 
                
                target_tempo = torch.tensor(tempo)
        else:
            target_tempo = torch.zeros(1)
        # TODO: not assigned right now
        target_chord = torch.zeros(1)

        # now convert from seconds to new sample rate
        beat_samples = np.array(beat_sec * self.target_sample_rate)
        downbeat_samples = np.array(downbeat_sec * self.target_sample_rate)

        # check if there are any beats beyond the file end and cut them off
        beat_samples = beat_samples[beat_samples < N]
        downbeat_samples = downbeat_samples[downbeat_samples < N]

        beat_samples = beat_samples.astype(int)
        downbeat_samples = downbeat_samples.astype(int)

        target_beat[0,beat_samples] = 1  # first channel is beats
        target_beat[1,downbeat_samples] = 1  # second channel is downbeats
        
        dataset = h5_file.split("/")[-2]
        
        # print(threading.enumerate()) 
        metadata = {
            "Filename" : filename,
            "Genre" : Genre,
            "dataset" : dataset,
            "Time signature" : Time_signature
        }
        return audio, target_beat, target_tempo, target_chord, metadata
    
    def apply_augmentations(self, audio, target_beat, target_tempo, target_chord, scale_factor=1.0):  
        # phase inversion
        if np.random.rand() < 0.5:      
            audio = -audio                                 
        
        # apply time stretching on GPU
        if scale_factor != 1.0:
            audio = audio.unsqueeze(0)
            audio = time_stretch(audio, float(1/scale_factor), self.audio_sample_rate)
            audio = audio.squeeze(0)   
                    
            # change tempo to stretched version
            target_tempo = target_tempo * scale_factor

            # now we update the targets beat and downbeat based on new tempo
            dbeat_ind = (target_beat[1,:] == 1).nonzero(as_tuple=False)
            dbeat_sec = dbeat_ind / self.target_sample_rate
            new_dbeat_sec = (dbeat_sec / scale_factor).squeeze()
            new_dbeat_ind = (new_dbeat_sec * self.target_sample_rate).long()

            beat_ind = (target_beat[0,:] == 1).nonzero(as_tuple=False)
            beat_sec = beat_ind / self.target_sample_rate
            new_beat_sec = (beat_sec / scale_factor).squeeze()
            new_beat_ind = (new_beat_sec * self.target_sample_rate).long()
            
  
            new_size = self.target_length
            # cut the beat index and downbeat index below new size
            new_beat_pos = torch.where(new_beat_ind < new_size)[0]
            new_downbeat_pos = torch.where(new_dbeat_ind < new_size)[0]
            if new_beat_pos.shape[-1] != 1 and new_beat_pos.shape[-1] != 0:
                new_beat_ind = new_beat_ind[:(new_beat_pos[-1]+1)]
            if new_downbeat_pos.shape[-1] != 1 and new_downbeat_pos.shape[-1] != 0:
                new_dbeat_ind = new_dbeat_ind[:(new_downbeat_pos[-1]+1)]
            
            # now convert indices back to target_beat vector            
            stretched_target = torch.zeros(2,new_size)
            if new_beat_pos.shape[-1] != 0:
                stretched_target[0,new_beat_ind] = 1
            if new_downbeat_pos.shape[-1] != 0:
                stretched_target[1,new_dbeat_ind] = 1
            target_beat = stretched_target
            
        
        # do some paddings
        if audio.shape[-1] < self.length:
            pad_size = int(self.length - audio.shape[-1])
            padl = pad_size - (pad_size // 2)
            padr = pad_size // 2
            audio = torch.nn.functional.pad(audio, 
                                             (padl, padr), 
                                             mode=self.pad_mode)

        # normalize the audio
        audio /= audio.float().abs().max()

        return audio, target_beat, target_tempo, target_chord
    
    def get_h5data(self, h5_file):
        # initialization
        beat_samples = None
        downbeat_samples = None
        tempo = None
        chord_samples = None
        Genre = "None"
        Time_signature = "None"
                
        f = h5py.File(h5_file, 'r')
        audio = f["audio_16k"][()] # import audio of 16k hz
        sr = f["sr"][()]
        if "filename" in f.keys():
            filename =f["filename"][()]
        if "Filename" in f.keys():
            filename =f["Filename"][()]
        if "beat_samples" in f.keys():
            beat_samples =f["beat_samples"][()]
        if "beat_sec" in f.keys():
            beat_sec = f["beat_sec"][()]
        if "downbeat_samples" in f.keys():
            downbeat_samples = f["downbeat_samples"][()]
        if "downbeat_sec" in f.keys():
            downbeat_sec = f["downbeat_sec"][()]
        if "Genre" in f.keys():          
            Genre = f["Genre"][()]
        if "Time signature" in f.keys():
            Time_signature = f["Time signature"][()]
        if "tempo" in f.keys():
            tempo = f["tempo"][()]
        if "chord_samples" in f.keys():
            chord_samples = f["chord_samples"][()]           
        
        f.close()            
        return audio, sr, beat_samples, beat_sec, downbeat_samples, downbeat_sec, tempo, chord_samples, filename, Genre, Time_signature
    
if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INDEX = 0
    dataset = "ballroom"
    device = torch.device(DEVICE, index=INDEX)
    train_loader = DownbeatDataset(h5_dir=root_path,
                            audio_sample_rate=16000, 
                            target_factor=hop_length,
                            dataset=dataset,
                            folder=1,
                            subset="train", 
                            length=160000,  
                            augment=True,
                            pad_mode='constant',
                            h5_file_augment=True,
                            # device=device,
                            )
    train_dataloader = torch.utils.data.DataLoader(train_loader, 
                                                shuffle=True,
                                                batch_size=8,
                                                num_workers=2,
                                                pin_memory=False)
    
    val_loader = DownbeatDataset(h5_dir=root_path,
                            audio_sample_rate=16000, 
                            target_factor=hop_length,
                            dataset=dataset,
                            subset="full-val", 
                            length=160000, 
                            augment=False,
                            pad_mode='constant')
    val_dataloader = torch.utils.data.DataLoader(val_loader, 
                                                shuffle=True,
                                                batch_size=1,
                                                num_workers=2,
                                                pin_memory=False)
    
    for ii, data in enumerate(train_dataloader):
        audio, target_beat, target_tempo, target_chord, meta_data, _ = data
        # beeping(audio, target_beat=target_beat, sample_rate=hop_length,name=meta_data["Filename"], beeping_file="D:/coding/beat/BeatNet/BeatNet/src/BeatNet/beeping/beep.wav", target_path="D:/coding/beat/BeatNet/BeatNet/src/BeatNet/audio_output")
        print("{}/{}".format(ii, len(train_dataloader)))