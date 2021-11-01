from tqdm import tqdm
import pandas as pd
import pandas as pd
import torchaudio.transforms as T
from pytube import YouTube
from tqdm.notebook import tqdm
import numpy as np
import glob
from pytube.exceptions import VideoPrivate


class_map = {
    '/t/dd00002':'Baby',
    '/m/03wwcy':'Doorbell',
    '/m/01h8n0':'Conversation',
    '/t/dd00038':'Rain'
}

def download_audio(df:pd.DataFrame):
    miss_vids = []
    root_url = "https://www.youtube.com/watch?v="
    n = len(df)
    for i in tqdm(range(n)):
        uid = df.loc[i,'YTID']
        l = df.loc[i,'labels'].split(',')
        if len(l) == 1:
            p = class_map[l[0]]
        else:
            for j in class_map.keys():
                if j in l:
                    p = class_map[j]
        try:
            yt = YouTube(url=root_url+uid)
            title = yt.title
            stream = yt.streams.filter(only_audio=True)
            print(stream[1])
            miss_vids.append({'path':f"{p}/{str(i)}",'vides-id':uid,'start_time':df.loc[i,'start'],'end_time':df.loc[i,'end'],'class':p})
            print(yt)
        except (Exception):
            print(f"{uid}")
            pass
        
        else:
            stream[1].download(p,filename=str(i)+'.mp4')
    return miss_vids        