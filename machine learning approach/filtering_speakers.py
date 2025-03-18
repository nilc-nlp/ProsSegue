import pandas as pd
import re
import os
import tgt
import chardet

# Function to check if an interval is inside any reference interval
# ALERT: this code considers that interviewee is always speaker 0
def is_spoken_by_interviewee(row):

    interviewee_tiers = [tier for tier in tg_reference.tiers if tier.name.startswith("TB-") and "ponto" not in tier.name and "0" in tier.name]

    # creating the list with all the utterances from the interviewee
    interviewee_utterances = []
    for tier in interviewee_tiers:
       for utterance in tier:
            interviewee_utterances.append((round(utterance.start_time, 2), round(utterance.end_time, 2)))
    interviewee_utterances.sort()
    #print(interviewee_utterances)
    syllable, start_time, end_time = re.match(pattern, row["frame"]).groups()
    start_time, end_time = float(start_time), float(end_time)

    for utterance in interviewee_utterances:
        if utterance[0] <= start_time and end_time <= utterance[1]:
            #print("KEEP THIS ONE", syllable, start_time, end_time)
            return True # keep this syllable
    return False # drop this syllable

def predict_encoding(tg_path):
    '''Predict a file's encoding using chardet'''
    # Open the file as binary data
    with open(tg_path, 'rb') as f:
        # Join binary lines for specified number of lines
        rawdata = b''.join(f.readlines())

    return chardet.detect(rawdata)['encoding']

# Córpus MuPe-Diversidades

common_path = os.getcwd() + "/MuPe-Diversidades/versao-1/" # os.getcwd gets the current folder
estados = ["AL", "BA", "CE", "ES", "GO", "MG", "MS", "PA", "PB", "PE", "PI", "PR", "RJ", "RO", "RS", "SE", "SP"]
numeros = ["1", "2"]

for estado in estados:
  for numero in numeros:
    audio_id = estado+numero
    try:

        tg_reference = common_path + estado + "/" + audio_id + "_OUTPUT_revised.TextGrid"
        tg_reference = tgt.io.read_textgrid(tg_reference, predict_encoding(tg_reference), include_empty_intervals=False)
        prosodic_features = pd.read_csv('ExtractedProsodicFeatures/versao final/'+audio_id+'_prosodic_features.csv')
    except:
        print(audio_id, "nao existe, pulando pro próximo")
        continue
    pattern = r"frame_([^\d_]+)_(\d+\.\d+)_(\d+\.\d+)"

    # Apply the filtering condition
    print(prosodic_features)

    df_filtered = prosodic_features[prosodic_features.apply(is_spoken_by_interviewee, axis=1)]
    print(df_filtered)

    # Step 4: Save the filtered DataFrame to a new CSV file
    df_filtered.to_csv('ExtractedProsodicFeatures/'+audio_id+'_prosodic_features_filtered_speakers.csv', index=False)   