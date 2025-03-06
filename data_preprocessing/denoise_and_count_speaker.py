from df.enhance import enhance, init_df, load_audio, save_audio
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Annotation
import os
import pandas as pd
import csv

def denoise_count(i):
    audio_path = 'open/test_wav' + df['path'][i][6:-3] + 'wav'
    audio, sample_rate = torchaudio.load(audio_path)
    enhanced = enhance(model, df_state, audio)

    output_path = 'open/denoise_wav' + df['path'][i][6:-3] + 'wav'
    save_audio(output_path, enhanced, sample_rate)

    try:
        diarization = pipeline(audio_path, min_speakers=0, max_speakers=2)
    except Exception as e:
        return 0
    speakers = set()
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
    num_no = len(speakers)

    try:
        diarization = pipeline(output_path, min_speakers=0, max_speakers=2)
    except Exception as e:
        return 0
    speakers = set()
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
    num_denoise = len(speakers)

    if os.path.exists(output_path):
        os.remove(output_path)

    return num_denoise, num_no

if __name__ == "__main__":
    model, df_state, _, _ = init_df()

    df = pd.read_csv('open/test.csv')

    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token="     ")  #your huggingface token here

    count_denoise = []
    ind = []
    count = []

    for i in range(80):
        num_d, num = denoise_count(i)
        count_denoise.append(num_d)
        count.append(num)
        ind.append(i)

    df_d = pd.DataFrame(data = count_denoise, index = ind, columns = ['count'])
    df = pd.DataFrame(data = count, index = ind, columns = ['count'])

    df_d.to_csv('open/denoise_count.csv')
    df.to_csv('open/just_count.csv')

    print("done")