from pydub import AudioSegment
import os

aud1 = os.path.normcase(r"open/train/ZZYYUTZI.ogg")
aud2 = os.path.normcase(r"open/train/ZSSZEISY.ogg")

audio1 = AudioSegment.from_file(aud1, format="ogg")

audio2 = AudioSegment.from_file(aud2, format="ogg")

if len(audio1) > len(audio2):
    audio2 = audio2 + AudioSegment.silent(duration=(len(audio1) - len(audio2)))
else:
    audio1 = audio1 + AudioSegment.silent(duration=(len(audio2) - len(audio1)))

combined = audio1.overlay(audio2)

merge1 = os.path.normcase(r"open/merge.ogg")

combined.export(merge1, format="ogg")