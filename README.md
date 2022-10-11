

# Branches
| Branches | Model   | DATA    |Features| STFT's parameters(window_size & stride)| Model name(in notion) |
| :---:   | :---: | :---: |:---: | :---:| :---:|
| PyanNet | PyanNet   | SingleChannel | ---| --- | ---|
| PyanNetlike | PyanNet   | SingleChannel |---|--- |SincNet (PyanNet)|
| StereoNet_Single_Channel | STFT+DataNormalization+LSTM+FeedForwards  | SingleChannel | Magnitudes|window_size=512, stride=256|STFT+DataNormalization|
| StereoNet_Multi_Channel | STFT+DataNormalization+LSTM+FeedForwards   | MultiChannel| Magnitude+IPD|window_size=512, stride=256 |STFT+DataNormalization|
| StereoNet_Multi_Channel_ILD| STFT+DataNormalization+LSTM+FeedForwards   | MultiChannel|Magnitude+IPD+ILD| window_size=512, stride=256|STFT+DataNormalization|
| StereoNet_CNN_SingleChannel | STFT+2CNN+LSTM+FeedForwards   | SingleChannel |Magnitudes| window_size=128, stride=32|HybridSincNet(STFT+2CNN)|
| StereoNet_CNN_MultiChannel | STFT+2CNN+LSTM+FeedForwards   | MultiChannel|Magnitude+IPD|window_size=128, stride=32 |HybridSincNet(STFT+2CNN)|
| StereoNet_CNN_MultiChannel_ILD | STFT+2CNN+LSTM+FeedForwards   | MultiChannel|Magnitude+IPD+ILD| window_size=128, stride=32|HybridSincNet(STFT+2CNN)|
| StereoNet_SincNet | SincNet+LSTM+FeedForwards  |MultiChannel|---|---|SincNet (Multichannel)|

### The difference between PyanNet and PyanNetlike:

PyanNet :
- code to train PyanNet with the pyannote-audio framework, and the loss function is **VAD_loss + BinaryCrossEntropy**, as defined in Pyannote Library.

PyanNetLike :
- code also uses the pyannote-audio framework but with more flexibility, we use an instance of a subclass of IterableDataset `SpeechChunkSampler` for training set, and `ValTestDataset` for evaluation sets that implements the \_\_iter\_\_() protocol, and represents an iterable over data samples, the data samples are the audio chunks of duration 5s that are read randomly from the audio. For loss function, we use just **BinaryCrossEntropy**.
 
# Download Data from S3

| DATA |  Description  | URL on S3    |
| :---:   | :---: | :---: |
| data_AMI_channel1 | First channel of Array1 of AMI dataset   | s3://ava-ai-eu/public_datasets/AMI/data_AMI_channel1/ | 
| data_AMI_Array1 | The 8 channels of Array 1 of AMI datasets   | s3://ava-ai-eu/public_datasets/AMI/data_AMI_Array1/ |
| free-sound | Noise that we added to the training set   | s3://ava-ai-eu/public_datasets/musan/noise/free-sound/ |
| lists | Contains the list of identifiers of the files in the training set. |s3://ava-ai-eu/public_datasets/AMI/lists/|
| only_words  | Contains the reference speaker diarization using RTTM format |s3://ava-ai-eu/public_datasets/AMI/only_words/|
| uems   | Describes the annotated regions using UEM format |s3://ava-ai-eu/public_datasets/AMI/uems/|

# Download pretrained models from S3

| Pretrained model |  The corresponding branch  | URL on S3    |
| :---:   | :---: | :---: |
| PyanNet |   PyanNet | s3://ava-ai-eu/trained_models/PyanNet/| 
| PyanNetlike | PyanNetLike   | s3://ava-ai-eu/trained_models/PyanNetlike/ |
| StereoNet_Single_Channel |  StereoNet_Single_Channel  | s3://ava-ai-eu/trained_models/StereoNet_Single_Channel/ |
| StereoNet_Multi_Channel | StereoNet_Multi_Channel |s3://ava-ai-eu/trained_models/StereoNet_Multi_Channel/|
| StereoNet_Multichannel_ILD | StereoNet_Multi_Channel_ILD|s3://ava-ai-eu/trained_models/StereoNet_Multichannel_ILD/|
| StereoNet_CNN_SingleChannel | StereoNet_CNN_SingleChannel |s3://ava-ai-eu/trained_models/StereoNet_CNN_SingleChannel/|
| StereoNet_CNN_MultiChannel |StereoNet_CNN_MultiChannel |s3://ava-ai-eu/trained_models/StereoNet_CNN_MultiChannel/|
| StereoNet_CNN_MultiChannel_ILD | StereoNet_CNN_MultiChannel_ILD |s3://ava-ai-eu/trained_models/StereoNet_CNN_MultiChannel_ILD/|
| StereoNet_SincNet | StereoNet_SincNet|s3://ava-ai-eu/trained_models/StereoNet_SincNet/|


# Install environment
```python
conda create -n env python=3.9 ipython ipykernel  
pip install git+https://github.com/pyannote/pyannote-audio.git@develop#egg=pyannote-audio  
pip install git+https://github.com/asteroid-team/torch-audiomentations@master#egg=torch_audiomentations  
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch  
```
# Training

**`python train.py config.yml --train_workers 4 --val_workers 4 --test_workers 4 --gpus 1`**  

# Loading the statistics

For the branchs:
- StereoNet_Single_Channel.
- StereoNet_Multi_Channel.
- StereoNet_Multi_Channel_ILD.

We should compute the statistics of normalization first and then start the training.
To do that we run the command:

**`python data_normalization.py config.yml `**

# How to use test_der_threshold to evaluate our model.

- During training the model is evaluated on the validation set at each epoch using **`OptimalDiarizationErrorRate`** and **`OptimalDiarizationErrorRateThreshold`**,
- At the end of the training we can get 3 things: 
    
    - The checkpoint of the best epoch.
    
    - The CDER on validation of the best epoch.
    
    - The optimal threshold of the best epoch: we got it by searching in the logs of Tensorboard the threshold that correspond to the best epoch.
  
- We load the checkpoint in a model variable using our best threshold, and we use the loaded_model to evaluate our model.
- A general example of the evaluation using **`test_der_threshold=0.5600000023841858`**:

```
checkpoint = torch.load("/path/to/best/checkpoints.ckpt")   
model = Model.from_config(config, writer, torch.Tensor([0.5600000023841858]))
model.load_state_dict(checkpoint['state_dict']) 
#trainer.validate(model, val_loader) —> evaluation on validation subset using the optimal threshold
trainer.test(model, test_loader) —> evaluation on test subset using the optimal threshold

```
**Exception:**

In PyanNet model, we proceed the same as in other models(PyanNetlike, StereoNet...), but we don't use **`trainer.validate`** or **`trainer.test`**. We use instead a function **`get_cder`** that works with the same principle.

Here is an example:

*We load the checkpoint in a model variable:*

```
checkpoint = torch.load("/path/to/best/checkpoints.ckpt")   
model = segmentation.load_state_dict(checkpoint['state_dict'])    
```
*To evaluate PyanNet model on validation subset:*
```  
metric_val = get_cder(model, protocol, 'development', config.batch_size , 0.5799999833106995 , get_devices(1)[0] )
 
cder_val = abs(metric_val)
conf = metric_val["confusion"] / metric_val["total"]
miss = metric_val["missed detection"] / metric_val["total"]
fa = metric_val["false alarm"] / metric_val["total"]
print (cder_val,conf,miss,fa)
print(f"{protocol.development} CDER: {100 * cder_val:.4f}%")
```
*To evaluate PyanNet model on test subset:*
```    
metric_test = get_cder(model , protocol, 'test',config.batch_size , 0.5799999833106995,get_devices(1)[0] )
cder_test = abs(metric_test)
conf = metric_test["confusion"] / metric_test["total"]
miss = metric_test["missed detection"] / metric_test["total"]
fa = metric_test["false alarm"] / metric_test["total"]
print (cder_test,conf,miss,fa)
print(f"{protocol.test} CDER: {100 * cder_test:.4f}%")  
```



# Inspecting logs

On AWS:
For example `3.138.175.75` is the IP adress of the machine.  

We run first the commands:  
**`ssh -i .ssh/chaimae-ava.pem -L 6006:localhost:6006 ubuntu@3.138.175.75`**  
**`tensorboard --logdir /home/ubuntu/Multichannel_dia/experiment5/chechpoints_exp5/training_logs --purge_orphaned_data True`**  

The tensorboard will be under the url: http://localhost:6006  

But before opening the url we run the command in another terminal:  
**`ssh -i .ssh/chaimae-ava.pem -L 6006:localhost:6006 -L 7777:localhost:8888 ubuntu@3.138.175.75`**  


# The paths in config.yml

For example, Using AWS machine:
```
ami_set: /home/ubuntu/Multichannel_dia/experiment5 is the path to database.yml file.
noise: /home/ubuntu/Multichannel_dia/datasets/free-sound/ is the path to free-sound noise files.
json: /home/ubuntu/Multichannel_dia/experiment5/stats_stereo.json is the path to json file that contains statistics values that we use to do the normaliation of STFT features.
```
# Data

## AMI Corpus

Some audio files in MultichannelArray of AMI dataset are not correctly formatted or are missing:
- `ES2010d`—> should be corrected.
    - each channel actually contains 2 channels.
    - these 2 channels = duplicate of the same channel.
- `IS1003b + IS1007d`—> should be removed.  
    - they contain just channel 2 of Array1.

## Training Data

Using the `sampling.py`, training chunks are selected randomly from AMI dataset audio files using the `database.yml` configuration file.   
Chunks are selected according to `config.yml` configuration file, with a duration of 5s, and number of speakers 4.  
For Background noise we use the `Musan/noise/free-sound files`.      
For ground Truth, we use `only_words` training annotations defined in `database.yml` configuration file.  

## Validation set

Using the `speech.py`, validation chunks are selected randomly from AMI dataset audio files using the `database.yml `configuration file.    
We don't use noise in validation set.    
For ground truth we use AMI `only_words` development annotations.


