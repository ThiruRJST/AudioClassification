# AudioClassification

The dataset for this model was collected from audioset. The audioset dataset didn't provide exact audio files instead only extracted features. I made a workaround, collected the video-ids from audioset.csv file and used **PYTUBE** to download the video from youtube and converted into WAV files using **FFMPEG**.

The script to download the videos from youtube is ```audio_download.py```

Then MP4 files are converted into WAV files 

Classes and Data Source:
-----------------------

1. Baby Crying - Audioset
2. Conversation - Audioset
3. Doorbell - Audioset
4. Rain - Audioset
5. Overflow - Audioset
6. Object Falling - Manually from YouTube



FFPMEG CONVERSION OF MP4 to WAV:
--------------------------------

```
1. cd into the class folders
2. for file in *.mp4; do ffmpeg -i "$file" -a 2 -f wav '${file:0:-4}'.wav; done
3. for file in *.mp4; do rm -f "$file"; done

```
At the end of this step, you will be left with converted WAV files and MP4 files will be deleted.

**The reasons for using WAV format**
    1. The WAV format doesn't use any lossy compression and the best to use formats for training the model afte FLAC.
    2. torch_audio backends supports WAV formats better.



Deep Learning Model
-------------------

1. The CNN14 model from PANET paper was customized for the given smaller dataset
2. The model's cfg file is provided in CFG folder







## Intern work to classify the audio files into 7 classes

- [x] Data Collection (4 classes: Baby,Conversation,Doorbell,Rain)
- [x] Create Pipeline for training(Preprocessing and Augmentations)
- [x] Create Model YAML file
- [x] Define the Training loop and Validation loop
- [] Create inference file for the model 
