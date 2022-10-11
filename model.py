
import pytorch_lightning as pl
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.optim import Adam
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dia_metrics import (
    OptimalDiarizationErrorRate,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
    OptimalDiarizationErrorRateThreshold,
)

from utils import merge_dict , pairwise
from pathlib import Path
from typing import Tuple,  Union, Optional, Text ,Callable
from typing_extensions import Literal
from einops import rearrange, reduce
from torch.utils.tensorboard import SummaryWriter
from torch_audiomentations import AddBackgroundNoise
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from config import Config
#from sampling import SpeechChunkSampler
from pyannote.database import FileFinder, get_protocol ,Protocol
import torchaudio
from scipy.signal import get_window
import json
from config import load_config







class PITSpeakerDiarizationLoss(nn.Module):
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate speaker diarization loss assuming inputs have already been permuted.
        :param predictions: torch.Tensor, shape (batch, frames, speaker)
        :param targets: torch.Tensor, shape (batch, frames, speaker)
        :return: per-speaker diarization loss, torch.Tensor, shape (batch, speaker)
        """
        bce = F.binary_cross_entropy(predictions, targets.float(), reduction="none")
        return torch.mean(bce, dim=1)

    def permute_targets(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Permute diarization targets to minimize BCE.

        :param predictions: torch.Tensor, shape (batch, frames, speaker)
        :param targets: torch.Tensor, shape (batch, frames, speaker)
        :return: torch.Tensor, shape (batch, frames, speaker)
        """
        num_speakers = predictions.shape[-1]
        permuted_labels = torch.zeros_like(
            targets, dtype=targets.dtype, device=targets.device
        )
        # LSAP/hungarian algorithm cannot be batched
        for b in range(predictions.shape[0]):
            loss_matrix = []
            for spk in range(num_speakers):
                # Calculate the loss of each predicted speaker with respect to ground-truth speakers
                with torch.no_grad():
                    hyp = predictions[b:b + 1, :, spk:spk + 1].expand(-1, -1, num_speakers)
                    loss_matrix.append(self(hyp, targets[b:b + 1]).squeeze())
            # Solve LSAP and permute accordingly
            for spk1, spk2 in zip(*linear_sum_assignment(torch.stack(loss_matrix).cpu())):
                permuted_labels[b, :, spk1] = targets[b, :, spk2]
        return permuted_labels

class STFTFeatureExtractor(nn.Module):
    def __init__(
        self,
        fft_size: int = 512,
        phase_diff_mode: Literal["first", "mean"] = "first",
        normalize_input=True, 
        mag_var :int = 1.,
        mag_mean :int =0.,
        ipd_mean :int =0.,
        #power_sq :int =0.,
        #epsilon :float = 10e-3
    ):
        super().__init__()
        assert phase_diff_mode in ["first", "mean"]
        self.fft_size = fft_size
        self.phase_diff_mode = phase_diff_mode
        
        self.spectrogram = Spectrogram(n_fft=fft_size, power=None , hop_length= fft_size//2)  # Complex STFT
        
        
        self.num_freq_bins = fft_size // 2 + 1
        self.normalize_input = normalize_input
        self.mag_mean = mag_mean
        #self.mag_var = power_sq - mag_mean**2 + epsilon
        self.mag_var = mag_var
        self.ipd_mean = ipd_mean

    def num_output_features(self, num_channels: int) -> int:
        num_fft_inputs = 1
        if num_channels > 1:
            num_fft_inputs = 2 * num_channels
            if self.phase_diff_mode == "first":
                num_fft_inputs -= 1
        return self.num_freq_bins * num_fft_inputs

    def num_output_frames(self, num_samples: int) -> int:
        rnd_input = torch.randn(1, 1, num_samples).cuda()
        with torch.no_grad():
            output = self.spectrogram(rnd_input)
        return output.shape[-1]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from waveform.
        :param waveform: torch.Tensor, shape (batch, channel, samples)
        :return: torch.Tensor, shape (batch, frames, output_channel * freq_bin)
            If `phase_diff_mode == "mean"`, output_channels = 2 * channel
            If `phase_diff_mode == "first"`, output_channels = 2 * channel - 1
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        
        
        
        if self.normalize_input:
             # Reshape magnitudes and IPD
            pattern = "batch channel freq frame -> batch frame (channel freq)"
            magnitude = rearrange(magnitude, pattern)
            ipd = rearrange(ipd, pattern)
            
            normalized_magnitude = torch.div(torch.sub(magnitude , self.mag_mean),(torch.sqrt(self.mag_var)))
       
            normalized_ipd = torch.sub(ipd ,self.ipd_mean)

             # Concatenate features, shape (batch, frames, features)
            return torch.cat([normalized_magnitude, normalized_ipd], dim=-1)
        else:                     
            return magnitude, ipd 
            
       
    
class StereoNet(pl.LightningModule):



    def __init__(
        self,
        #noise_path: Union[Text, Path],
        #noise_min_snr: float,
        #noise_max_snr: float,
        #noise_probability: float,
        json_path: Union[Text, Path],
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        num_layers: int = 2,
        fft_size: int = 512,
        hidden_size: int = 128,
        #num_input: int = 513,
        num_speakers: int = 4,
        summary_writer: Optional[SummaryWriter] = None,
        test_der_threshold :torch.Tensor =None,
        normalize:bool = False,
        
        
    ): 
        super().__init__()

        self.sample_rate=sample_rate,
        self.num_channels=num_channels
        self.normalize = normalize
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feature_extractor = STFTFeatureExtractor(fft_size).to(device)
        
        self.pit = PITSpeakerDiarizationLoss()
        
        self.test_der_threshold = OptimalDiarizationErrorRateThreshold()
        self.der = OptimalDiarizationErrorRate()
        self.false_alarm = OptimalFalseAlarmRate()
        self.missed_detection = OptimalMissedDetectionRate()
        self.confusion = OptimalSpeakerConfusionRate()
        
        self.test_der = OptimalDiarizationErrorRate(test_der_threshold)
        self.test_false_alarm = OptimalFalseAlarmRate(test_der_threshold)
        self.test_missed_detection = OptimalMissedDetectionRate(test_der_threshold)
        self.test_confusion = OptimalSpeakerConfusionRate(test_der_threshold)
        '''      
        self.noise = AddBackgroundNoise(
            noise_path, noise_min_snr, noise_max_snr, mode="per_example", p=noise_probability
        )
        '''
        self.summary_writer = summary_writer
        self.sample_rate = sample_rate
        
        #load .json_file 
        self.json_path = json_path
        with open(self.json_path) as json_file:
            data = json.load(json_file)
           
        mag_mean = torch.Tensor(data["mag_mean"]).cuda()
        mag_var = torch.Tensor(data["mag_var"]).cuda()
        ipd_mean = torch.Tensor(data["ipd_mean"]).cuda()
        
        
        self.features_extractor = STFTFeatureExtractor( fft_size, "first", mag_mean=mag_mean, mag_var=mag_var,ipd_mean=ipd_mean).to(device)
        num_features = self.features_extractor.num_output_features(num_channels)
        self.input_projection = nn.Linear(num_features, 60)
        
        self.lstm = nn.LSTM(
            input_size=60,
            hidden_size=hidden_size,  # because bidirectional=True
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        ).to(device)

        self.linear1 = nn.Linear(256, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_speakers)
        

    @staticmethod
    def from_config(config: Config, summary_writer: Optional[SummaryWriter] = None, test_der_threshold: torch.Tensor=None ) -> 'StereoNet':
        return StereoNet(
            #noise_path=config.noise,
            #noise_min_snr=config.noise_min_snr,
            #noise_max_snr=config.noise_max_snr,
            #noise_probability=config.noise_probability,
            sample_rate=config.sample_rate,
            num_channels=config.num_channels,
            fft_size=config.fft_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_speakers=config.num_speakers,
            learning_rate=config.learning_rate,
            summary_writer=summary_writer,
            json_path=config.json,
            test_der_threshold=test_der_threshold
        )

    
    def forward(self, mix: torch.Tensor):
        normalized_features = self.features_extractor.forward(mix).float().cuda()
        
        projected_normalized_features = F.relu(self.input_projection(normalized_features))
        
        # Get common output
        outputs = self.lstm(projected_normalized_features )[0]  # (batch, frames, features)
        
        outputs = F.leaky_relu(self.linear1(outputs))
        
        outputs = torch.sigmoid(self.linear2(outputs)).transpose(1,2)
        print("outputs",outputs.shape)
        return outputs
    
    def configure_optimizers(self):
        optimizer = Adam(lr= 0.001, betas=[0.9, 0.999], eps=1e-08, weight_decay=0, amsgrad=False, params=self.parameters())
        
        lr_scheduler = {
        "scheduler": ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=10,
            min_lr=0,
            eps=1e-08,
            verbose=False,
            
        ),
        "interval": "epoch",
        "reduce_on_plateau": True,
        "monitor": "val_der",
        "strict": True,
    }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


    def training_step(self, batch, batch_idx):
        waveform = batch["mix"].to(self.device)
        #waveform = self.noise(waveform, self.sample_rate)
        labels = batch["labels"].to(self.device)
        preds = self(waveform)
        preds = rearrange(preds, "batch speaker frame -> batch frame speaker")
        permuted_labels = self.pit.permute_targets(preds, labels)
        loss = self.pit(preds, permuted_labels).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        waveform = batch["mix"].to(self.device)
        
        labels = batch["labels"].to(self.device)#(batch frame speaker)
        
        preds = self(waveform)# (batch speaker frame)
        
        labels = rearrange(labels, "batch frame speaker -> batch speaker frame")#(batch speaker frame)
        
        for metric in [self.der, self.false_alarm, self.missed_detection, self.confusion, self.test_der_threshold]:
            metric(preds, labels)
        self.log_dict(
            {
                "val_der": self.der,
                "val_fa": self.false_alarm,
                "val_miss": self.missed_detection,
                "val_confusion": self.confusion,
                "test_der_threshold": self.test_der_threshold,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        waveform = batch["mix"].to(self.device)
        
        labels = batch["labels"].to(self.device)
        
        preds = self(waveform)
        
        labels = rearrange(labels, "batch frame speaker -> batch speaker frame")
        
        for metric in [self.test_der, self.test_false_alarm, self.test_missed_detection, self.test_confusion]:
            metric(preds, labels)
        self.log_dict(
            {
                "test_der": self.test_der,
                "test_fa": self.test_false_alarm,
                "test_miss": self.test_missed_detection,
                "test_confusion": self.test_confusion,
                
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
    
