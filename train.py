#!/usr/bin/env python
# coding: utf-8



import argparse
from pathlib import Path

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import csv
from pyannote.audio.models.segmentation import PyanNet
from model import PyanNetLike
from config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path, help="Path to config.yml")
parser.add_argument("--train_workers", type=int, default=4, help="Number of workers for data loaders")
parser.add_argument("--val_workers", type=int, default=0, help="Number of workers for data loaders")
parser.add_argument("--test_workers", type=int, default=0, help="Number of workers for data loaders")
parser.add_argument("--gpus", type=int, default=1, help="Number of gpus for training")
args = parser.parse_args()

logdir = args.config.parent
config = load_config(args.config)

os.environ["PYANNOTE_DATABASE_CONFIG"] = config.ami_set + '/database.yml'

from sampling import SpeechChunkSampler
from speech import ValTestDataset

def train():
    

    
    pn_model = PyanNet(sincnet={'stride': 10})
    
    model = PyanNetLike(pn_model, 
                    config.num_speakers, 
                    config.learning_rate, 
                    torch.Tensor([0.5]),
                    config.noise, 
                    config.noise_min_snr,
                    config.noise_max_snr,
                    config.noise_probability,
                    config.sample_rate,
                    config.num_channels,)
    
   
    
    num_output_frames = model.num_output_frames(config.chunk_duration, config.sample_rate)
    
    # Build speech chunk generator for joint speaker diarization and source separation
    
    train_data = SpeechChunkSampler.from_config(config, num_output_frames)
    
    train_loader = train_data.get_loader(config.batch_size, num_workers = args.train_workers)
   
    
    
    val_data = ValTestDataset(
         config.num_channels,
         config.num_speakers,
         config.resolution(num_output_frames),
         duration=config.chunk_duration,
         step=config.step,
         sample_rate=config.sample_rate,
         dataset="val"
     )
    
    


    val_loader = val_data.get_loader(config.batch_size, num_workers = args.val_workers)
    

    monitor="val_der"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_der",
        dirpath="checkpoints_experiment3/",
        filename= f"{{epoch}}-{{{monitor}:.6f}}",
        verbose = -1, 
        mode="min", 
        save_top_k= 1,          
        
    )
    # The checkpoints and logging files are automatically saved in save_dir
    logger = TensorBoardLogger("checkpoints_experiment3/training_logs", name=None, version='logs')

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=config.epochs,
        num_sanity_val_steps=0,
        logger=logger,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)
    

    test_data = ValTestDataset(
        config.num_channels,
        config.num_speakers,
        config.resolution(num_output_frames),
        duration=config.chunk_duration,
        step=config.step,
        sample_rate=config.sample_rate,
        dataset = "test"

    )
    
    test_loader = test_data.get_loader(config.batch_size, num_workers = args.test_workers)
    trainer.test(model, dataloaders=test_loader)
    '''
    pn_model_val = PyanNet(sincnet={'stride': 10})
    checkpoint = torch.load("/gpfswork/rech/cjx/uaf42iq/multichannel-dia/PyanNetlike/checkpoints_experiment3/epoch=59-val_der=0.210918.ckpt")
    
    model_val = PyanNetLike(pn_model_val, config.num_speakers, config.learning_rate,torch.Tensor([0.5]), config.noise, config.noise_min_snr, config.noise_max_snr, config.noise_probability)
    model_val.load_state_dict(checkpoint['state_dict'])
    trainer.validate(model_val,val_loader)
    '''
if __name__ == '__main__':
    train()
    
    
    
     
