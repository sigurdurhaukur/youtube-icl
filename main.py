from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from datasets import Dataset
import time
from tqdm.auto import tqdm

# Read the csv file
df = pd.read_csv('./youtube-sl-25-metadata.csv')
df.columns = ['video_id', 'iso639']

info = {"languages": ['is'], "iso639": ['icl']}

# filter by language
df = df[df['iso639'].isin(info['iso639'])]
df = df.reset_index(drop=True)

print("Number of videos: ", len(df))
print(df.head())

batch_size = 5

dataset = {"video_id": [], "transcript": []}
for i in tqdm(range(0, len(df), batch_size)):
    ids = df['video_id'][i:i+batch_size].values.tolist()
    languages = info['languages']

    transcripts = YouTubeTranscriptApi.get_transcripts(ids, languages=languages, preserve_formatting=True)

    for script in transcripts[0].keys():
        dataset['video_id'].append(script)
        dataset['transcript'].append(transcripts[0][script])

dataset = Dataset.from_dict(dataset)
dataset = dataset.shuffle()
dataset = dataset.train_test_split(test_size=0.2)
print(dataset)

dataset.push_to_hub("Sigurdur/icelandic-sign-language")

