#!/usr/bin/env python
# coding: utf-8



import argparse
from pathlib import Path

import os
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import csv
from pyannote.audio.models.segmentation import PyanNet
from model import StereoNet
from config import load_config

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to config.yml")
    parser.add_argument("--train_workers", type=int, default=0, help="Number of workers for data loaders")
    parser.add_argument("--val_workers", type=int, default=0, help="Number of workers for data loaders")
    parser.add_argument("--test_workers", type=int, default=0, help="Number of workers for data loaders")
    parser.add_argument("--gpus", type=int, default=1, help="Number of gpus for training")
    args = parser.parse_args()


    logdir = args.config.parent
    config = load_config(args.config)

    

    from sampling import SpeechChunkSampler
    from speech import ValTestDataset

    writer = SummaryWriter(logdir)
    
    model = StereoNet.from_config(config, writer)
    
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
        dirpath="checkpoints_sincNet/",
        filename= f"{{epoch}}-{{{monitor}:.6f}}",
        verbose = -1, 
        mode="min", 
        save_top_k= 1,          
        
    )
    # The checkpoints and logging files are automatically saved in save_dir
    logger = TensorBoardLogger("checkpoints_sincNet/training_logs", name=None, version='logs')

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=config.epochs,
        num_sanity_val_steps=0,
        logger=logger,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
    )
    #trainer.fit(model, train_loader, val_loader)
    

    test_data = ValTestDataset(
        config.num_channels,
        config.num_speakers,
        config.resolution(num_output_frames),
        duration=config.chunk_duration,
        step=config.step,
        sample_rate=config.sample_rate,
        dataset = "test"

    )
    
    test_loader = test_data.get_loader(config.batch_size, args.test_workers)
    
    
    checkpoint = torch.load('/home/ubuntu/Multichannel_dia/multichannel-dia/checkpoints_sincNet/epoch=227-val_der=0.228122.ckpt')
    
    model = StereoNet.from_config(config, writer, torch.tensor([0.5399999618530273]))
    model.load_state_dict(checkpoint['state_dict'])
    
    trainer.test(model, test_loader)
if __name__ == '__main__':
    train()
    
    
    
     
