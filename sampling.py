import argparse 
import os
import logging
import random
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Text, Union, Dict, List
from typing_extensions import Literal
from einops import reduce
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch_audiomentations import AddBackgroundNoise
from model import HybridSincNet
from pyannote.core import Annotation, Segment
from pyannote.database import FileFinder, get_protocol
from model import StereoNet
from config import Config
from config import load_config








class SpeechChunkSampler(IterableDataset):
    def __init__(
        self,
#        root: Union[Text, Path],  # AMI dataset directory with database.yml file
        noise: AddBackgroundNoise,
        diarization_resolution: float= 8.0 ,
        sample_rate: int = 16000,  # samples per second
        num_sources: int = 4,
        num_speakers: int = 4,
        chunk_duration: float = 5.0,
        num_channels: int = 1,
        channels_choice: Literal["first", "rand"] = "first"
    ):
#        self.root = root
        self.sample_rate = sample_rate
        self.num_sources = num_sources
        self.num_speakers = num_speakers
        self.noise = noise
        self.chunk_duration = chunk_duration
        self.diarization_resolution = diarization_resolution
        self.num_channels = num_channels
        assert channels_choice in ["first", "rand"]
        self.channels_choice = channels_choice
        self.annotations = []
        self.audio_files = []

        preprocessors = {'audio': FileFinder()}
        only_words = get_protocol('AMI.SpeakerDiarization.only_words', preprocessors=preprocessors)

        for file in only_words.train():
            self.annotations.append(file['annotation'])
            self.audio_files.append(file['audio'])

        msg = f"Found {len(self.annotations)} speaker diarization annotations and audios"
        logging.debug(msg)
        self.total_duration = sum(self.get_annotation_duration(ann) for ann in self.annotations)
        

    @staticmethod
    def from_config(config: Config, num_chunk_frames: int) -> 'SpeechChunkSampler':
        noise = AddBackgroundNoise(
            config.noise,
            config.noise_min_snr,
            config.noise_max_snr,
            mode="per_example",
            p=config.noise_probability
        )
        return SpeechChunkSampler(
#            config.ami_set,
            noise,
            config.resolution(num_chunk_frames),
            config.sample_rate,
            config.num_sources,
            config.num_speakers,
            config.chunk_duration,
            config.num_channels,
            config.channels_choice
        )

    @staticmethod
    def get_annotation_duration(annotation: Annotation) -> float:
        return annotation.get_timeline().extent().end

    @staticmethod
    def shift_annotation(annotation: Annotation, shift_duration: float) -> Annotation:
        new_annotation = Annotation()
        for segment, track, label in annotation.itertracks(yield_label=True):
            segment = Segment(segment.start + shift_duration, segment.end + shift_duration)
            new_annotation[segment, track] = label
        return new_annotation
    
    @staticmethod
    def read_audio_section(filename, start_time, stop_time, samplerate, num_channels, channels_choice):
        track = sf.SoundFile(filename)
    
        can_seek = track.seekable() # True
        if not can_seek:
            raise ValueError("Not compatible with seeking")
    
        if track.samplerate != samplerate:
            raise ValueError("Data samplerate is different from requitred samplerate")
    
        start_frame = int(np.rint(samplerate * start_time))
        frames_to_read = int(np.rint(samplerate * (stop_time - start_time)))
        track.seek(start_frame)
        audio_section = track.read(frames_to_read)
        if audio_section.ndim == 1:
            audio_section = np.expand_dims(audio_section, axis=1)
            
        if channels_choice == "rand":
            raise ValueError("DEBUG FIRST ! This part of code is not debugged yet")
            
            audio_section = np.transpose(audio_section)
            np.random.shuffle(audio_section)
            audio_section = np.transpose(audio_section)
            
        if audio_section.shape[1] < num_channels:
            raise ValueError("Fewer channels in data than required")
        elif audio_section.shape[1] > num_channels:
            audio_section = audio_section[:, :num_channels]
            
        audio_section
            
        return audio_section

    def sample_annotation_and_audio(self, min_duration: float):
        index_permutation = random.sample(range(len(self.annotations)), k=len(self.annotations))
        for index in index_permutation:
            if min_duration < self.get_annotation_duration(self.annotations[index]):
                return self.annotations[index], self.audio_files[index]
        msg = f"No annotation with duration >= {min_duration} exists"
        raise ValueError(msg)

    def __len__(self):
        # Estimation of what would be the number of chunks in an epoch
        return int(self.total_duration // self.chunk_duration)

    def __iter__(self):
        while True:
            # Sample diarization annotation
            annotation, audio_file = self.sample_annotation_and_audio(self.chunk_duration)
    
            # Sample segment (10 ms are added for security)
            chunk_start = random.uniform(0., self.get_annotation_duration(annotation) - self.chunk_duration - 0.01)
            chunk_stop = chunk_start + self.chunk_duration
            chunk_annotation = annotation.crop(Segment(chunk_start, chunk_stop))
            chunk_annotation = self.shift_annotation(chunk_annotation, -chunk_start)
    
            chunk_audio = self.read_audio_section(audio_file, chunk_start, chunk_stop,
                                                  self.sample_rate, self.num_channels, self.channels_choice)
            chunk_audio = torch.from_numpy(chunk_audio).transpose(0, 1)
    
            # discretize chunk annotation
            d_chunk_annotation = chunk_annotation.discretize(
                resolution=self.diarization_resolution,
                duration=self.chunk_duration
                )
            #d_chunk_annotation = torch.from_numpy(d_chunk_annotation.data).transpose(0, 1)
            d_chunk_annotation = torch.from_numpy(d_chunk_annotation.data)

            num_speakers = len(chunk_annotation.labels())
            if num_speakers < self.num_sources:
                diff = self.num_sources - num_speakers
                #d_chunk_annotation = F.pad(d_chunk_annotation, (0, 0, 0, diff), "constant", 0)
                d_chunk_annotation = F.pad(d_chunk_annotation, (0, diff, 0, 0), "constant", 0)
            elif num_speakers > self.num_sources:
                msg = f"Dropping chunk with {num_speakers} speakers. Expected a maximum of {self.num_sources}"
                logging.warning(msg)
                continue

            #d_chunk_annotation = d_chunk_annotation.data.transpose(0, 1)
            
            yield {"mix": chunk_audio.unsqueeze(0), "labels": d_chunk_annotation.unsqueeze(0)}

    def _collate_fn(self, data: List[Dict]) -> Dict:
        # Batch tensors
        original = torch.vstack([sample["mix"] for sample in data])  # shape (batch, channel, samples)
        labels = torch.vstack([sample["labels"] for sample in data])  # shape (batch, frames, sources)
        if original.ndim == 2:
            original, labels = original.unsqueeze(0), labels.unsqueeze(0)

        return {
            "mix": self.noise(original, self.sample_rate).float(),
            "labels": labels
        }

    def get_loader(self, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    

    
