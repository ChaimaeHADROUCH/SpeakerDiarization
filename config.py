
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Text
from typing_extensions import Literal
import yaml
import numpy as np


@dataclass
class Config:
    ami_set: Path
    noise: Path
    chunk_duration: float = 5.0
    noise_min_snr: float = 5.0
    noise_max_snr: float = 15.0
    noise_probability: float = 0.9
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 300
    step: float = 0.5



def load_config(filepath: Union[Text, Path]) -> Config:
    with open(filepath, "r") as file:
        data = yaml.load(file, yaml.FullLoader)
    data["ami_set"] = data["ami_set"]
    data["noise"] = Path(data["noise"])
    return Config(**data)
