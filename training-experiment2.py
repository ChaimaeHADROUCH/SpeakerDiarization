#!/usr/bin/env python
# coding: utf-8

import argparse
import os

from pathlib import Path
from types import MethodType
from typing import List, Optional, Text, Union
import torch
import pytorch_lightning as pl
from pyannote.database import FileFinder ,get_database
from pyannote.database import get_protocol, Protocol

from pyannote.audio.core.model import Model
from pyannote.audio.pipelines.utils import get_model, get_devices
from pyannote.audio.tasks.segmentation.segmentation import Segmentation
from pyannote.database import get_protocol, Protocol
from pyannote.database.util import FileFinder
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from pyannote.audio.core.inference import Inference
from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
from pyannote.audio.utils.signal import binarize
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.torchmetrics import (
    OptimalDiarizationErrorRate,
    OptimalDiarizationErrorRateThreshold,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
)

from torch_audiomentations.utils.file import find_audio_files
from torch_audiomentations import AddBackgroundNoise
from config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path, help="Path to config.yml")
parser.add_argument("--workers", type=int, default=0, help="Number of workers for data loaders")
parser.add_argument("--gpus", type=int, default=1, help="Number of gpus for training")
args = parser.parse_args()


logdir = args.config.parent
config = load_config(args.config)


PROTOCOL = 'AMI.SpeakerDiarization.only_words'

# tell pyannote.database where to find partition, reference, and wav files
# path provided by the PYANNOTE_DATABASE_CONFIG environment variable
os.environ['PYANNOTE_DATABASE_CONFIG'] = config.ami_set + '/database.yml'

# used to automatically find paths to wav files
preprocessors = {"audio": FileFinder()}
# initialize 'only_words' experimental protocol
protocol = get_protocol(PROTOCOL, preprocessors={"audio": FileFinder()})
'''
def get_augmentation(
    mode: str = "per_example",
    p: float = 0.9,
    output_type: Optional[str] = None,
) -> Optional[BaseWaveformTransform]:
    
    return BaseWaveformTransform(
        mode = "per_example",
        p = 0.9,
        output_type = "tensor" )

augmentation = get_augmentation("per_example" , 0.9 , "tensor")
augmentation = augmentation.unfreeze_parameters()
'''
def get_audio_file_paths(paths : List[Union[Path, Text]]) ->  Optional[List[Path]]:
    if not paths:
        return None
    noise_paths = []
    for path in paths:
        noise_paths.extend(find_audio_files(path))
    return noise_paths
'''
noise_paths = get_audio_file_paths([os.path.join(config.noise ,"music/fma/", ""),
                                    os.path.join(config.noise , "music/fma-western-art/", "")  ,
                                    os.path.join(config.noise , "music/hd-classical/", "") , 
                                    os.path.join(config.noise , "music/jamendo/", ""), 
                                    os.path.join(config.noise , "music/rfm/", ""), 
                                    os.path.join(config.noise , "noise/free-sound/", "")  ,
                                    os.path.join(config.noise , "noise/sound-bible/", "")])
'''
noise_paths = get_audio_file_paths([config.noise])
def get_noise_augmentation(
    paths: List[Path],
    min_snr: float,
    max_snr: float,
    noise_probability: float
) -> Optional[BaseWaveformTransform]:
    return AddBackgroundNoise(
        noise_paths,
        min_snr_in_db = config.noise_min_snr,
        max_snr_in_db = config.noise_max_snr,
        mode="per_example",
        p = config.noise_probability,
        output_type = "dict",
    )

augmentation = get_noise_augmentation(noise_paths,config.noise_min_snr,config.noise_max_snr,config.noise_probability)


def get_task(
    protocol: Protocol,
    weight: Optional[Text] = None,
    augmentation: Optional[BaseWaveformTransform] = augmentation,
):

        return Segmentation(
            protocol=protocol,
            duration=config.chunk_duration,
            max_num_speakers=4,
            warm_up= 0.0,
            batch_size= config.batch_size,
            weight=weight,
            #overlap={"probability": 0.5,"snr_min": 0.0,"snr_max": 10.0},
            num_workers=args.workers,
            pin_memory= False,
            augmentation=augmentation,
            loss  = "bce" ,
            vad_loss = "bce",
            metric = [
            OptimalDiarizationErrorRate(),
            OptimalDiarizationErrorRateThreshold(),
            OptimalSpeakerConfusionRate(),
            OptimalMissedDetectionRate(),
            OptimalFalseAlarmRate(),
        ]
        )


# Create a segmentation task and configure the model to train with it
seg_task = get_task( protocol,augmentation=augmentation)

#We initialize one model with the PyanNet architecture, 
#In particular, we increase the default stride of the initial sincnet feature extraction layer to 10.
segmentation = PyanNet(task=seg_task, sincnet={'stride': 10})

monitor, direction = seg_task.val_monitor


SEED = int(os.environ.get("PL_GLOBAL_SEED", "0"))
PATIENCE = 5

def configure_optimizers(self):
    optimizer = Adam(lr=config.learning_rate, betas=[0.9, 0.999], eps=1e-08, weight_decay=0, amsgrad=False, params=self.parameters())
    if monitor is None:
            return optimizer
    
    lr_scheduler = {
        "scheduler": ReduceLROnPlateau(
            optimizer,
            mode=direction,
            factor=0.5,
            patience=4 * PATIENCE,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=2 * PATIENCE,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        ),
        "interval": "epoch",
        "reduce_on_plateau": True,
        "monitor": monitor,
        "strict": True,
    }

    return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


segmentation.configure_optimizers = MethodType( configure_optimizers, segmentation)

    # Fit model
segmentation.setup(stage="fit")


checkpoint_callback = pl.callbacks.ModelCheckpoint(
          dirpath="checkpoints_experiment2/",
          filename="{epoch}" if monitor is None else f"{{epoch}}-{{{monitor}:.6f}}",
          verbose = -1,
          monitor = monitor, 
          mode    = direction, 
          save_top_k = None if monitor is None else 5,
      )
#earlystopping = EarlyStopping(patience=10, monitor= monitor, mode="min")

logger = TensorBoardLogger(
    "checkpoints_experiment2/training_logs",
    name="",
    version="",
    log_graph=False,  # TODO: fixes onnx error with asteroid-filterbanks
)


trainer = pl.Trainer(
    gpus=args.gpus,
    max_epochs=config.epochs,
    num_sanity_val_steps=0,
    logger=logger,
    checkpoint_callback=True,
    callbacks=[checkpoint_callback],
    weights_summary=None,
)


def get_cder(
    model: Model,
    protocol: Protocol,
    subset: Text,
    batch_size: int,
    threshold: float,
    device: torch.device
) -> DiscreteDiarizationErrorRate:
    metric = DiscreteDiarizationErrorRate()
    inference = Inference(
        segmentation,
        skip_aggregation=True,
        device=device,
        step=0.5,
        batch_size=batch_size,
    )
    for file in tqdm(list(getattr(protocol, subset)()), desc=f"Evaluating on {subset}"):
        metric(file["annotation"], binarize(inference(file), onset=threshold, offset=threshold), uem=file["annotated"])
    return metric

if __name__ == '__main__':
    
    #trainer.fit(segmentation)
    
    checkpoint = torch.load('/gpfswork/rech/cjx/uaf42iq/multichannel-dia/PyanNet_v1/PyanNet/checkpoints_experiment2/epoch=88-Segmentation-AMISpeakerDiarizationonly_words-OptimalDiarizationErrorRate=0.208335.ckpt')   
    model = segmentation.load_state_dict(checkpoint['state_dict'])
        
    metric = get_cder(model, protocol, 'development', config.batch_size , 0.5 , get_devices(1)[0] )
 
    cder = abs(metric)
    conf = metric["confusion"] / metric["total"]
    miss = metric["missed detection"] / metric["total"]
    fa = metric["false alarm"] / metric["total"]
    print (cder,conf,miss,fa)
    print(f"{protocol.development} CDER: {100 * cder:.4f}%")

    metric_val = get_cder(model, protocol, 'development', config.batch_size , 0.5600000023841858 , get_devices(1)[0] )
 
    cder_val = abs(metric_val)
    conf = metric_val["confusion"] / metric_val["total"]
    miss = metric_val["missed detection"] / metric_val["total"]
    fa = metric_val["false alarm"] / metric_val["total"]
    print (cder_val,conf,miss,fa)
    print(f"{protocol.development} CDER: {100 * cder_val:.4f}%")


    
    metric_test = get_cder(model , protocol, 'test',config.batch_size , 0.5600000023841858,get_devices(1)[0] )
    cder_test = abs(metric_test)
    conf = metric_test["confusion"] / metric_test["total"]
    miss = metric_test["missed detection"] / metric_test["total"]
    fa = metric_test["false alarm"] / metric_test["total"]
    print (cder_test,conf,miss,fa)
    print(f"{protocol.test} CDER: {100 * cder_test:.4f}%")
     
    
    
