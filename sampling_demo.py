import os
from torch_audiomentations import AddBackgroundNoise
from scipy.io.wavfile import write
import numpy as np
import argparse
import time
from pyannote.core import notebook, SlidingWindowFeature, SlidingWindow
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--ami-set", required=True, type=str, help="Root directory for speech dataset")
parser.add_argument("--noise", required=True, type=str, help="Root directory for noise dataset")
parser.add_argument("--channels", default=1, type=int,
                    help="Number of channels in audio (must match the dataset)")
parser.add_argument("--speakers", default=4, type=int,
                    help="Maximum number of speakers in the room at once (<= IR dataset positions)")
parser.add_argument("-sr", "--sample-rate", default=16000, type=int,
                    help="Sampling rate of the recordings (number of samples per second)")
args = parser.parse_args()
args.sr = args.sample_rate

os.environ["PYANNOTE_DATABASE_CONFIG"] = args.ami_set + '/database.yml'

from sampling import SpeechChunkSampler


chunk_duration = 5.0
target_diarization_frames = 313
resolution = chunk_duration / target_diarization_frames


# Create diffuse noise
#print("Initializing noise...", end="", flush=True)
noise = AddBackgroundNoise(args.noise, min_snr_in_db=5.0, max_snr_in_db=15.0, mode="per_example", p=1)
#print("Done")

# Build speech chunk generator for joint speaker diarization and source separation
#print("Creating audio sampler...", end="", flush=True)
sampler = SpeechChunkSampler(args.sample_rate, noise, args.speakers, chunk_duration, resolution, args.channels)
#print("Done")

# Extract a sample chunk
#print("Loading sample batch...", end="", flush=True)
loader = sampler.get_loader(batch_size=32)
start = time.monotonic()
chunk = next(iter(loader))
end = time.monotonic() - start
#print("Done")
#print(f"Took {end:.2f} seconds")

#print()
#print("Mix:", chunk["mix"].shape)
#print("Annotation:", chunk["labels"].shape)

# Write signals for debugging
for channel in range(args.channels):
    write(
        f"mix_channel{channel}.wav",
        args.sr,
        np.round(chunk["mix"].numpy()[0, channel] * 32767).astype(np.int16)
    )

sw = SlidingWindow(start=0, step=resolution, duration=resolution)
annotation = SlidingWindowFeature(chunk["labels"][0].transpose(0, 1).numpy(), sw)
fig, ax = plt.subplots(figsize=(8, 2))
notebook.plot_feature(annotation, ax=ax)
plt.savefig("annotation.png", bbox_inches="tight")
