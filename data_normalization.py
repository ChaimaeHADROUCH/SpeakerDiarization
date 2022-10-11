import argparse
from pathlib import Path
from typing import Tuple,  Union, Optional, Text ,Callable
from typing_extensions import Literal


from torch.utils.tensorboard import SummaryWriter
from sampling import SpeechChunkSampler
from config import Config
import torchaudio
import json
import torch
from torchaudio.transforms import Spectrogram, InverseSpectrogram
import pandas as pd
from einops import reduce
from torch_audiomentations import AddBackgroundNoise
from config import Config
from config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path, help="Path to config.yml")
args = parser.parse_args()
logdir = args.config.parent
config = load_config(args.config)




class Data_normalization():
    def __init__(
        self,
        noise_path: Union[Text, Path],
        noise_min_snr: float,
        noise_max_snr: float,
        noise_probability: float,
        fft_size: int = 512,
        phase_diff_mode: Literal["first", "mean"] = "first",
        
        
    ):
        super().__init__()
        assert phase_diff_mode in ["first", "mean"]
        self.fft_size = fft_size
        self.phase_diff_mode = phase_diff_mode
        self.spectrogram = Spectrogram(n_fft=fft_size, power=None, hop_length= fft_size//2) 
        self.num_freq_bins = fft_size // 2 + 1 # Complex STFT
        self.noise = AddBackgroundNoise(
            noise_path, noise_min_snr, noise_max_snr, mode="per_example", p=noise_probability
        )
    
    @staticmethod
    def from_config(config: Config) -> 'Data_normalization':
        
        return Data_normalization(
            noise_path=config.noise,
            noise_min_snr=config.noise_min_snr,
            noise_max_snr=config.noise_max_snr,
            noise_probability=config.noise_probability,  
        )

    def num_output_frames(self, num_samples: int) -> int:
        rnd_input = torch.randn(1, 1, num_samples).cpu()
        with torch.no_grad():
            output = self.spectrogram(rnd_input)
        return output.shape[-1]

    def compute(self, waveform:torch.Tensor) -> tuple:
    
       
        # Get complex STFT, shape (batch, channel, freq, frames)
        stft = self.spectrogram(waveform)
        # TODO? CSIPD: concat real and imaginary parts
        # Get phase differences, shape (batch, output_channel, freq, frames)
        if self.phase_diff_mode == "first":
            ipd = torch.angle(stft[:, 1:]) - torch.angle(stft[:, 0:1])
        else:
            ipd = torch.angle(stft) - torch.angle(torch.mean(stft, dim=1, keepdim=True))
        ipd = ipd % (2 * torch.pi)
        magnitude = stft.abs()    
    
        return magnitude , ipd

    def get_normalization( self, batch_size:int = 32):
        total_samples = 0
        mag_freq_sum = 0.
        mag_freq_sum_sq = 0.
        ipd_freq_sum = 0.
        
        feature_computation = Data_normalization.from_config(config)
        noise = feature_computation.noise
        num_chunk_frames = feature_computation.num_output_frames(config.num_chunk_samples)
        sampler = SpeechChunkSampler(noise , config.resolution(num_chunk_frames))
        train_loader = sampler.get_loader(num_workers=4)
        number_batches = train_loader.__len__()
        iterator = iter(train_loader)
        
        
        for idx in range(number_batches):

            chunk = next(iterator)

            magnitude,ipd = feature_computation.compute(chunk["mix"])

            total_samples += magnitude.shape[-1]*magnitude.shape[0]

            magnitude =  reduce(magnitude, "batch channel freq time ->  channel freq", "sum")
            ipd =  reduce(ipd, "batch channel freq time ->  channel freq", "sum")
        

            mag_freq_sum += magnitude # Shape of mag_freq_sum = vector of F frequences
            
            mag_freq_sum_sq += magnitude**2 # Shape of mag_freq_sum = vector of F frequences
            
            # Sum over frame axis (ipd)
            ipd_freq_sum += ipd # Shape of ipd_freq_sum = vector of F frequences
            
        
        

        power_sq = (1/total_samples)*mag_freq_sum_sq
        
        mag_mean = (1/total_samples)*mag_freq_sum
        
        mag_var = power_sq - (mag_mean)**2

        ipd_mean = (1/(total_samples))*ipd_freq_sum
          
        #create .json file
        
        normalized_features={
            "mag_mean": mag_mean,
            "mag_var":mag_var,
            "ipd_mean":ipd_mean
        }
        if isinstance(mag_mean, torch.Tensor):
            normalized_features['mag_mean'] = mag_mean.tolist()
        if isinstance(mag_var, torch.Tensor):
            normalized_features['mag_var'] = mag_var.tolist()
        if isinstance(ipd_mean, torch.Tensor):
            normalized_features['ipd_mean'] = ipd_mean.tolist()

        with open('stats_mono.json', 'w', encoding='utf-8') as json_file:
            json.dump(normalized_features, json_file)
        
        
if __name__ == '__main__':
    Data_normalization.get_normalization(32) 
    
