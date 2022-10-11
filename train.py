
#!/usr/bin/env python
# coding: utf-8



import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from config import load_config
from model import StereoNet


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
    num_chunk_frames = model.feature_extractor.num_output_frames(config.num_chunk_samples)

    
    
    # Build speech chunk generator for joint speaker diarization and source separation
    #print("Creating chunk sampler...", end="", flush=True)
    train_data = SpeechChunkSampler.from_config(config, num_chunk_frames)
    train_loader = train_data.get_loader(config.batch_size, args.train_workers)
    #print("Done")
    
    #print("Loading validation dataset...", end="", flush=True)
    val_data = ValTestDataset(
        config.num_channels,
        config.num_speakers,
        config.resolution(num_chunk_frames),
        duration=config.chunk_duration,
        step=config.step,
        sample_rate=config.sample_rate,
        dataset="val"
    )
    val_loader = val_data.get_loader(config.batch_size, args.val_workers)
    #print("Done")
    monitor="val_der"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_der",
        dirpath="checkpoints_exp4/",
        filename= f"{{epoch}}-{{{monitor}:.6f}}",
        verbose = -1, 
        mode="min", 
        save_top_k= 5,          
        
    )
    # The checkpoints and logging files are automatically saved in save_dir
    logger = TensorBoardLogger("checkpoints_exp4/training_logs", name=None, version='logs')
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=config.epochs,
        num_sanity_val_steps=0,
        logger=logger,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
    )
    
    #trainer.fit(model, train_loader, val_loader)
    
    #print("Loading test dataset...", end="", flush=True)
    test_data = ValTestDataset(
        config.num_channels,
        config.num_speakers,
        config.resolution(num_chunk_frames),
        duration=config.chunk_duration,
        step=config.step,
        sample_rate=config.sample_rate,
        dataset = "test"
    )
    test_loader = test_data.get_loader(config.batch_size, args.test_workers)
    #trainer.test(dataloaders=test_loader)
    
    #checkpoint = torch.load(config.ckpt_path , map_location=torch.device('cpu'))
    #model = StereoNet.load_state_dict(checkpoint['state_dict'],test_der_threshold =test_der_threshold )
    #trainer.test(model)
    
    checkpoint = torch.load('/gpfswork/rech/cjx/uaf42iq/multichannel-dia/exp4_v4/Exp4/StereoNet49/checkpoints_exp4/epoch=107-val_der=0.220676.ckpt')
    
    model = StereoNet.from_config(config, writer, torch.Tensor([0.5]))
    model.load_state_dict(checkpoint['state_dict'])
    
    trainer.test(model, test_loader)
    
if __name__ == '__main__':
    train()
