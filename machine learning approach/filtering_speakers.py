# Code to filter speech from interviewers, keeping only speech from interviewees

# Requirements:
# - Textgrids with the reference utterances must be in correct format, with correct names, inside a folder named '/MuPe-Diversidades/versao-1/'
# - CSV files with extracted prosodic features from each interview must be in correct format, with correct names, inside folder named 'ExtractedProsodicFeatures/versao final/'

import pandas as pd
import re
import os
import tgt
import chardet

# Function to check if a syllablee is inside any reference intervals (utterances from interviewee)
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
    syllable, syl_start_time, syl_end_time = re.match(pattern, row["frame"]).groups()
    syl_start_time, syl_end_time = float(syl_start_time), float(syl_end_time)

    # each utterance is a tuple with start and end time e.g. (0.0, 15.2), where utterance[0] is its start time and utterance[1] its end time
    for utterance in interviewee_utterances:
        if utterance[0] <= syl_start_time and syl_end_time <= utterance[1]:
            return True # keep this syllable (the interviewee said it)
    return False # drop this syllable (the interviewer said it)

def predict_encoding(tg_path):
    '''Predict a file's encoding using chardet'''
    # Open the file as binary data
    with open(tg_path, 'rb') as f:
        # Join binary lines for specified number of lines
        rawdata = b''.join(f.readlines())
    return chardet.detect(rawdata)['encoding']

# Corpus MuPe-Diversidades

common_path = os.getcwd() + "/MuPe-Diversidades/versao-1/" # os.getcwd gets the current folder
estados = ["AL", "BA", "CE", "ES", "GO", "MG", "MS", "PA", "PB", "PE", "PI", "PR", "RJ", "RO", "RS", "SE", "SP"]
numeros = ["1", "2"]

for estado in estados:
  for numero in numeros:
    audio_id = estado+numero
    print("Processing",audio_id)
    try:
        tg_reference = common_path + estado + "/" + audio_id + "_OUTPUT_revised.TextGrid"
        tg_reference = tgt.io.read_textgrid(tg_reference, predict_encoding(tg_reference), include_empty_intervals=False)
        prosodic_features = pd.read_csv('ExtractedProsodicFeatures/versao final/'+audio_id+'_prosodic_features.csv')
    except:
        print(audio_id, "doesn't exist, skipping to the next one")
        continue
    pattern = r"frame_([^\d_]+)_(\d+\.\d+)_(\d+\.\d+)"

    # Apply the filtering condition
    df_filtered = prosodic_features[prosodic_features.apply(is_spoken_by_interviewee, axis=1)]
    print(df_filtered)
    print(audio_id,"successfully filtered")

    # Save the filtered DataFrame to a new CSV file
    df_filtered.to_csv('ExtractedProsodicFeatures/'+audio_id+'_prosodic_features_filtered_speakers.csv', index=False)   
