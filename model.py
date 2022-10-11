from pathlib import Path
from typing_extensions import Literal
from typing import Tuple, Union, Optional, Text
import pytorch_lightning as pl
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from pyannote.audio.models.segmentation import PyanNet
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
from torch.utils.tensorboard import SummaryWriter
import time
from torch_audiomentations import AddBackgroundNoise
from config import Config
import csv



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
                # Calculate the joint loss of each predicted speaker with respect to ground-truth speakers
                with torch.no_grad():
                    hyp = predictions[b:b + 1, :, spk:spk + 1].expand(-1, -1, num_speakers)
                    loss_matrix.append(self(hyp, targets[b:b + 1]).squeeze())
            # Solve LSAP and permute accordingly
            for spk1, spk2 in zip(*linear_sum_assignment(torch.stack(loss_matrix).cpu())):
                permuted_labels[b, :, spk1] = targets[b, :, spk2]
        return permuted_labels



class PyanNetLike(pl.LightningModule):
    def __init__(
        self, 
        net: PyanNet, 
        num_speakers: int, 
        learning_rate: float, 
        test_der_threshold: torch.Tensor,
        noise_path: Union[Text, Path],
        noise_min_snr: float,
        noise_max_snr: float,
        noise_probability: float = 0.9,
        sample_rate: int = 16000,
        num_channels: int = 1,
        summary_writer: Optional[SummaryWriter] = None
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.net = net
        if self.net.hparams.linear["num_layers"] > 0:
            in_features = self.net.hparams.linear["hidden_size"]
        else:
            in_features = self.net.hparams.lstm["hidden_size"] * (
                2 if self.net.hparams.lstm["bidirectional"] else 1
            )
        
        self.num_speakers=num_speakers
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.summary_writer = summary_writer
        self.net.classifier = nn.Linear(in_features, num_speakers)
        self.net.activation = nn.Sigmoid()
        self.pit = PITSpeakerDiarizationLoss()

        
        self.test_der_threshold = OptimalDiarizationErrorRateThreshold()
        
        self.der = OptimalDiarizationErrorRate()

        self.false_alarm = OptimalFalseAlarmRate()
        self.missed_detection = OptimalMissedDetectionRate()
        self.confusion = OptimalSpeakerConfusionRate()
        
        #self.test_der_threshold = test_der_threshold
        self.test_der = OptimalDiarizationErrorRate()
        self.test_false_alarm = OptimalFalseAlarmRate()
        self.test_missed_detection = OptimalMissedDetectionRate()
        self.test_confusion = OptimalSpeakerConfusionRate()
        

        self.noise = AddBackgroundNoise(noise_path, noise_min_snr, noise_max_snr, mode="per_example", p=noise_probability)
    
    @staticmethod
    def from_config(config: Config, summary_writer: Optional[SummaryWriter] = None) -> 'PyanNetLike':
        return PyanNetLike(
            num_speakers=config.num_speakers,
            learning_rate=config.learning_rate,
            noise_path=config.noise,
            noise_min_snr=config.noise_min_snr,
            noise_max_snr=config.noise_max_snr,
            noise_probability=config.noise_probability,
            sample_rate=config.sample_rate,
            summary_writer=summary_writer)



    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    
    
    def configure_optimizers(self):
        optimizer = Adam(lr=self.learning_rate, betas=[0.9, 0.999], eps=1e-08, weight_decay=0, amsgrad=False, params=self.parameters())
        
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
        labels = batch["labels"].to(self.device)
        waveform = self.noise(waveform, self.sample_rate)
        preds = self(waveform)
        permuted_labels = self.pit.permute_targets(preds, labels)
        loss = self.pit(preds, permuted_labels).mean()
        self.log("train_loss", loss)

        return loss
    
    #TODO add noise method, then apply it to the waveform in training_step

    
    def validation_step(self, batch, batch_idx):
        #TODO: check shape of batch
        waveform = batch["mix"].to(self.device)
        labels = batch["labels"].to(self.device)
        preds = self(waveform)
        preds = rearrange(preds, "batch frame speaker -> batch speaker frame")
        labels = rearrange(labels, "batch frame speaker -> batch speaker frame")
        threshold = self.test_der_threshold.compute()
        for metric in [self.der, self.false_alarm, self.missed_detection, self.confusion, self.test_der_threshold]:
            metric(preds, labels)
        self.log_dict(
            {
                "val_der": self.der,
                "val_fa": self.false_alarm,
                "val_miss": self.missed_detection,
                "val_confusion": self.confusion,
                "test_der_threshold":self.test_der_threshold,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )



    def test_step(self, batch, batch_idx):
        waveform = batch["mix"].to(self.device)
        labels = batch["labels"].to(self.device)
        #preds = rearrange(self(waveform), "batch frame speaker -> batch speaker frame")
        preds = self(waveform)
        preds = rearrange(preds, "batch frame speaker -> batch speaker frame")
        labels = rearrange(labels, "batch frame speaker -> batch speaker frame")
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Change self.der to self.der_test etc ... (in order not to accumulate validation der)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        
    def num_output_frames(self, chunk_duration, samp_rate):
        with torch.no_grad():
            out = self(torch.randn(1, 1, int(np.rint(chunk_duration * samp_rate))))
        return out.shape[1]
