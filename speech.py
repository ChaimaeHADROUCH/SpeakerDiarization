import os
import logging
from pathlib import Path
from typing import Text, Union, List, Dict
from typing_extensions import Literal

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from einops import rearrange
from pyannote.core import Annotation, Segment
from pyannote.database import FileFinder, get_protocol
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset


class ValTestDataset(IterableDataset):
    def __init__(
        self,
#        root: Union[Text, Path],  # AMI dataset directory with database.yml file
        num_channels: int,
        num_sources: int,
        diarization_resolution: float,
        duration: float = 5.0,  # seconds
        step: float = 0.5,  # seconds
        sample_rate: int = 16000,
        dataset: Literal["val", "test"] = "val"
    ):
#        self.root = root
        self.num_channels = num_channels
        self.num_sources = num_sources
        self.diarization_resolution = diarization_resolution
        self.duration = duration
        self.step = step
        self.sample_rate = sample_rate
        self.num_samples = int(np.rint(duration * sample_rate))
        self.step_samples = int(np.rint(step * sample_rate))
        self._num_chunks = None
        assert dataset in ["val", "test"]
        self.dataset = dataset

        self.annotations = []
        self.audio_files = []
        
        preprocessors = {'audio': FileFinder()}
        only_words = get_protocol('AMI.SpeakerDiarization.only_words', preprocessors=preprocessors)

        if self.dataset == "val":
            target_list = only_words.development()
        elif self.dataset == "test":
            target_list = only_words.test()

        for file in target_list:
            self.annotations.append(file['annotation'])
            self.audio_files.append(file['audio'])

        msg = f"Found {len(self.annotations)} speaker diarization annotations and audios"
        logging.debug(msg)

    def _collate_fn(self, data: List[Dict]) -> Dict:
        return {
            "mix": torch.vstack([sample["mix"].float() for sample in data]), # shape (batch, channel, samples)
            "labels": torch.vstack([sample["labels"] for sample in data]) # shape (batch, frames, sources)
            }

    def get_loader(self, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )

    def _load_mix(self, audio_file: Text) -> torch.Tensor:
#        sr, audio = wavfile.read(audio_file)

        track = sf.SoundFile(audio_file)
        sr = track.samplerate
        audio = track.read()
        
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=1)

        if sr != self.sample_rate or audio.shape[1] != self.num_channels:
            raise ValueError("Data samplerate or number of channels are different from requitred")
        
        return torch.from_numpy(audio).transpose(0, 1)

    def __len__(self):
        # TODO account for last chunk padding
        if self._num_chunks is None:
            self._num_chunks = 0
            for i in range(len(self.audio_files)):
                num_samples = sf.SoundFile(self.audio_files[i]).frames
                numerator = num_samples - self.num_samples + self.step_samples
                self._num_chunks += int(numerator // self.step_samples)
        return self._num_chunks

    def __iter__(self):
        for i in range(len(self.annotations)):
            ref = self.annotations[i]
            mix = self._load_mix(self.audio_files[i])  # shape (channels, samples)
            file_duration = mix.shape[1] / self.sample_rate

            mix_chunks = rearrange(
                mix.unfold(1, self.num_samples, self.step_samples),
                "channel chunk sample -> chunk channel sample",
            )
            last_chunk_end_time = (mix_chunks.shape[0] - 1) * self.step + self.duration

            # FIXME pad last chunk so we don't miss it
            #print("Last chunk end time:", last_chunk_end_time)
            #print("File duration:", file_duration)

            for chunk in range(mix_chunks.shape[0]):
                start = chunk * self.step
                chunk_ref = torch.from_numpy(ref.discretize(
                    support=Segment(start, start + self.duration),
                    resolution=self.diarization_resolution,
                    duration=self.duration,
                ).data)
                num_frames, num_sources = chunk_ref.shape
                if num_sources == 0:
                    chunk_ref = torch.zeros(num_frames, self.num_sources)
                elif num_sources < self.num_sources:
                    diff = self.num_sources - num_sources
                    chunk_ref = F.pad(chunk_ref, (0, diff, 0, 0), "constant", 0)
                elif num_sources > self.num_sources:
                    # ?????????????????????????
                    msg = f"Dropping chunk with {num_sources} speakers. Expected a maximum of {self.num_sources}"
                    logging.warning(msg)
                    continue

                yield {
                    "mix": mix_chunks[chunk:chunk+1],
#                    "labels": chunk_ref.transpose(0, 1).unsqueeze(0)
                    "labels": chunk_ref.unsqueeze(0)
                }
