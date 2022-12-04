from dataclasses import dataclass
from pathlib import Path
from typing import Union, Text
import yaml
import numpy as np
from typing_extensions import Literal


@dataclass
class Config:
    ami_set: Path
    noise: Path
    ckpt_path : Path
    json : Path
    sample_rate: int = 16000
    num_channels: int = 1
    num_speakers: int = 4
    num_sources: int = 4
    chunk_duration: float = 5.0
    noise_min_snr: float = 5.0
    noise_max_snr: float = 15.0
    noise_probability: float = 0.9
    fft_size: int = 512
    hidden_size: int = 128
    num_layers: int = 4
    dropout: float = 0.5
    ssep_loss_target_channel: Union[Literal["mix_mean", "spec_mean"], int] = "spec_mean"
    learning_rate: float = 0.001
    batch_size: int = 34
    epochs: int = 2
    loss_weight: float = 0.5
    channels_choice: Literal["first", "rand"] = "first"
    step: float = 0.5

    @property
    def num_chunk_samples(self) -> int:
        return int(np.rint(self.chunk_duration * self.sample_rate))

    def resolution(self, num_chunk_frames: int) -> float:
        return self.chunk_duration / num_chunk_frames


def load_config(filepath: Union[Text, Path]) -> Config:
    with open(filepath, "r") as file:
        data = yaml.load(file, yaml.FullLoader)
    data["ami_set"] = data["ami_set"]
    data["noise"] = Path(data["noise"])
    return Config(**data)
