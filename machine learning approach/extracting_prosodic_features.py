# Extraction of prosodic features

# Requirements:
# Corpus MuPe-Diversidades, there must be folders "/MuPe-Diversidades/versao-1/" containing all audios, textgrids with phonetic alignment and refrence textgrids with prosodic segmented utterances, in the same folder as this code


# Related methods
#    Luengo, I., Navas, E., Hernáez, I., & Sánchez, J. (2005). Automatic emotion recognition using prosodic parameters. In Ninth European conference on speech communication and technology.
#    Rao, K. S., Koolagudi, S. G., & Vempada, R. R. (2013). Emotion recognition from speech using global and local prosodic features. International journal of speech technology, 16(2), 143-160.

import os
from os.path import isfile, join
import pandas as pd
import parselmouth
from adapted_feature_extraction_utils import *
import librosa
import numpy
import tgt
import chardet
import statistics


# Função para extrair features prosódicas
def extract_prosody(frame, sr, frame_id,utt_avg_pitch, next_interval_text, next_interval_dur, interval_start_time, interval_end_time, nucleus_end_time, nucleus_start_time,  nucleus_vowel, vowel_stats_dict):

  sound = parselmouth.Sound(values=frame, sampling_frequency=sr)
  df = pd.DataFrame()

  attributes = {}

  intensity_attributes = get_intensity_attributes(sound)[0]
  pitch_attributes, _, avg_pitch = get_pitch_attributes(sound)

  attributes['f0_avgutt_diff'] = abs(avg_pitch - utt_avg_pitch)
  if next_interval_text == "sil": 
    p_dur = next_interval_dur
  else:
    p_dur = 0 
  attributes['p_dur'] = p_dur

  # normalized duration of the nucleus of the syllable (always a vowel)

  if nucleus_vowel != None:
    vowel_type_mean = vowel_stats_dict[nucleus_vowel]["mean"]
    vowel_type_std_dev = vowel_stats_dict[nucleus_vowel]["std_dev"]
    attributes['n_dur'] = ((nucleus_end_time - nucleus_start_time) - vowel_type_mean) / vowel_type_std_dev 
  else: 
    attributes['n_dur'] = 0 # there was no nucleus vowel

  attributes.update(intensity_attributes)
  attributes.update(pitch_attributes)

  for attribute in attributes:
    df.at[0, attribute] = attributes[attribute]
  
  df.at[0, 'frame'] = frame_id
  rearranged_columns = df.columns.tolist()[-1:] + df.columns.tolist()[:-1]
  df = df[rearranged_columns]

  return df


def predict_encoding(tg_path):
    '''Predict a file's encoding using chardet'''
    # Open the file as binary data
    with open(tg_path, 'rb') as f:
        # Join binary lines for specified number of lines
        rawdata = b''.join(f.readlines())
    return chardet.detect(rawdata)['encoding']

# Corpus MuPe-Diversidades

estados = ["AL", "BA", "CE", "ES", "GO", "MG", "MS", "PA", "PB", "PE", "PI", "PR", "RJ", "RO", "RS", "SE", "SP"]
numeros = ["1", "2"]

# Creating list with audios' names
common_path = os.getcwd() + "/MuPe-Diversidades/versao-1/" # os.getcwd gets the current folder
audio_files = []
for estado in estados:
  for numero in numeros:
    audio_id = estado+numero+'.wav'
    if isfile(join(common_path+estado+"/", audio_id)): # checking if the file exists
      audio_files.append(estado+"/"+audio_id)

# Creating list with names, ids, states and paths of audios
audios_list = []
tg_phones_list = []
tg_reference_list = []
for path in audio_files:
  audio_id = path.replace('.wav','').split("/")[1]
  estado = path[0:2]
  file = path[3:]
  audios_list.append([file,audio_id,estado, common_path+path])
  tg_phones = common_path + estado + "/" + audio_id + "_fones.TextGrid"
  tg_reference =  common_path + estado + "/" + audio_id + "_OUTPUT_revised.TextGrid"
  tg_phones_list.append(tg_phones)
  tg_reference_list.append(tg_reference)

for i, inquiry in enumerate(audios_list): # To extract features exclusively from specific interviews, specify here, inside parenthesis, adding for example '[4:5], start=4'
  print("Processing", i,inquiry)

  tg_phone = tg_phones_list[i]
  tg_phone = tgt.io.read_textgrid(tg_phone, predict_encoding(tg_phone), include_empty_intervals=False)
  syllables_tier = tg_phone.get_tier_by_name("silabas-fonemas")
  try: # according to the version of ufpalign, the textgrid's phonemes' and graphemes' tiers could have one of the following two names
    phonemes_tier = tg_phone.get_tier_by_name("fonemeas")
    wordGraphemesTier = tg_phone.get_tier_by_name("palavras-grafemas") 
  except ValueError:
    phonemes_tier = tg_phone.get_tier_by_name("fonemas")
    wordGraphemesTier = tg_phone.get_tier_by_name("grafemas")

  # reading reference textgrid with prosodic segmented utterances to get labels and boundary positions
  tg_reference = tgt.io.read_textgrid(tg_reference_list[i], predict_encoding(tg_reference_list[i]), include_empty_intervals=False)
  print(tg_phones_list[i], tg_reference_list[i]) # to check if I got the correct files

  # Loading the audio file
  audio = audios_list[i] 
  audio_path = audio[3]
  print(audio_path)
  y, sr = librosa.load(audio_path, sr=None)  # sr=None ensures native sampling rate

  # Dividing audio in syllables

  syllable_frames = [ y[int(interval.start_time * sr):int(interval.end_time * sr)] for interval in syllables_tier]
  
  # Generate silent padding for the start and end of each syllable
  #padding_duration = 0.05  # 50ms of silence on each side so pitch extraction will consider edges - if pitch is extracted from the silence padding, it is considered NAN and are supposed to be ignored     
  #num_padding_samples = int(sr * padding_duration) # Convert time to samples
  #silent_padding = numpy.zeros(num_padding_samples)  # Silent array
  #extended_syllable_frames = []
  #for frame in syllable_frames:
  #  extended_frame = numpy.concatenate((silent_padding, frame, silent_padding))
  #  extended_syllable_frames.append(extended_frame)
  # syllable_frames = extended_syllable_frames)
  # print(extended_syllable_frames)
  # quit()

  print("Quantity of syllables:", len(syllable_frames))

  # Here, as syllables have different durations, they'll end up with different numbers of samples, which correspond to columns and become NAN, so I'm padding all syllables to the maximum frame length  
  max_length = max(len(frame) for frame in syllable_frames)
  syllable_frames = [numpy.pad(frame, (0, max_length - len(frame)), mode='constant') for frame in syllable_frames]

  # ADAPT HERE IF USING A DIFFERENT CORPUS, OR IF YOU WISH UTTERANCES FROM ALL SPEAKERS -> UNCOMMENT THIS BLOCK AND COMMENT THE LINE BELOW
  # Merge TB tiers from all speakers -- later on, there is filtering_speakers.py to filter utterances spoken by interviewers
  TB_tiers = [tier for tier in tg_reference.tiers if tier.name.startswith("TB-") and "ponto" not in tier.name]
  all_utterances = []
  for tier in TB_tiers:
    all_utterances.extend(tier.intervals)
  all_utterances.sort(key=lambda interval: interval.start_time)
  # list of all utterances, ordered according to start time

  # Alternatively, only get utterances from interviewee, considering it is always speaker 0:
  #all_utterances = tg_reference.get_tier_by_name("TB-speaker 0") # ADAPT HERE IF USING A DIFFERENT CORPUS, OR IF YOU WISH UTTERANCES FROM ALL SPEAKERS -> CHANGE THIS LINE FOR THE COMMENTED BLOCK ABOVE, HOWEVER I STILL WOULD HAVE TO FILTER SYLLABLES LIKE IN THE FILTERING_SPEAKERS.PY

  # Calculating vowel mean duration for every possible nucleus vowel
  vowel_types = ["a","e","i","o","u","a~","e~","i~","o~","u~", "E","O"]
  vowel_stats_dict = {vowel: {"mean": None, "std_dev": None, "durations": []} for vowel in vowel_types}

  for phone in phonemes_tier:
    if phone.text in vowel_stats_dict:
      vowel_stats_dict[phone.text]["durations"].append(phone.end_time - phone.start_time)

  for vowel_type in vowel_stats_dict:
    vowel_stats_dict[vowel_type]["mean"] = statistics.mean(vowel_stats_dict[vowel_type]["durations"])
    vowel_stats_dict[vowel_type]["std_dev"] = statistics.stdev(vowel_stats_dict[vowel_type]["durations"]) if len(vowel_stats_dict[vowel_type]["durations"]) > 1 else 0
    print("Vowel types mean calculated:",vowel_type,"->", vowel_stats_dict[vowel_type]["mean"], vowel_stats_dict[vowel_type]["std_dev"])
  
  utterance_averages = []
  labels = []
  i_utt = 0

  # Attributing labels to each syllable
  #   (Going through syllables tier, verifying whether each syllable's start time is smaller than the end time of the current utterance, meaning that the current syllable would belong to the current utterance, and label "NB" (no boundary) is attributed. If not, "TB" is attributed and we move to the next syllable and utterance)
  for i_syl, syllable in enumerate(syllables_tier): 
    if syllable.text != "sil":
      if i_syl+1 >= len(syllables_tier) or i_utt >= len(all_utterances) or syllables_tier[i_syl+1].start_time >= round(all_utterances[i_utt].end_time, 2):
        labels.append("TB")
        i_utt += 1
      elif syllables_tier[i_syl+1].text == "sil" and i_syl+2 < len(syllables_tier) and syllables_tier[i_syl+2].start_time >= round(all_utterances[i_utt].end_time, 2):
        labels.append("TB")
        i_utt += 1
      else:
        labels.append("NB")   

  # Calculating pitch average for each utterance
  # Note: here, we don't consider the extended sound with a silent padding at the beginning and at the end as the utterance duration is quite longer than a syllable's duration, but still, a few values could be lost at the beginning and ending of each utterance
  for i_utt, utterance in enumerate(all_utterances):
    utterance_frame = y[int(utterance.start_time * sr):int(utterance.end_time * sr)]
    sound = parselmouth.Sound(values=utterance_frame, sampling_frequency=sr)
    utt_avg = get_utterance_avg_pitch(sound)
    utterance_averages.append(utt_avg)

  print("Length of the list of utterances' pitch averages:", len(utterance_averages))
  print("Length of all utterances:", len(all_utterances))

  all_syllables_prosodic_features = []
  utterance_counter = 0
  labels_counter = 0
  phones_index = 0

  # Extracting prosodic features from each syllable
  for i, (frame, interval) in enumerate(zip(syllable_frames, syllables_tier)): # CHANGED SYLLABLE FRAMES TO EXTENDED SYLLABLE FRAMES TO TEST

    if interval.text != "sil" and utterance_counter <= len(all_utterances): 
      interval.text = interval.text.replace(" ", "")
      start_time = round(interval.start_time, 2)
      end_time = round(interval.end_time, 2)
      frame_id = f"frame_{interval.text}_{start_time}_{end_time}"
      print("frame", frame_id)
      
      # Identifying the nucleus vowel and getting its characteristics

      # Aligning phones' tier with syllables' tier - reaching the first phone of the current syllable
      while phonemes_tier[phones_index].start_time < start_time:
        phones_index += 1
      # Reaching the nucleus vowel of the syllable    
      phones_index_aux = 0
      while (not any(vowel in phonemes_tier[phones_index+phones_index_aux].text.lower() for vowel in "aeiou")) and (phonemes_tier[phones_index+phones_index_aux].end_time <= interval.end_time): # procurando o núcleo da sílaba
        phones_index_aux += 1    
      # Checking if we found the nucleus vowel (if the phone index already passed the end time of the syllable or we reached a silent phone, if so, we go back to the time of the first phone of the syllable and check for 'j' or 'w' as nucleus vowel)
      if phonemes_tier[phones_index+phones_index_aux].start_time >= interval.end_time or phonemes_tier[phones_index+phones_index_aux].text == "sil":
        phones_index_aux = 0
        while not any(vowel in phonemes_tier[phones_index+phones_index_aux].text.lower() for vowel in "jw") and (phonemes_tier[phones_index+phones_index_aux].end_time <= interval.end_time):
          phones_index_aux += 1
        
        if phonemes_tier[phones_index+phones_index_aux].start_time >= interval.end_time or phonemes_tier[phones_index+phones_index_aux].text == "sil":
          nucleus_vowel = None
        elif phonemes_tier[phones_index+phones_index_aux].text == "j":
          nucleus_vowel = "i"
        elif phonemes_tier[phones_index+phones_index_aux].text == "w":
          nucleus_vowel = "u"
      else:  
        nucleus_vowel = phonemes_tier[phones_index+phones_index_aux].text
      phones_index += phones_index_aux
      nucleus_start_time = phonemes_tier[phones_index].start_time
      nucleus_end_time = phonemes_tier[phones_index].end_time

      # Getting characteristics of the next syllable to consider whether there was a pause or not after the current syllable
      if i+1 == len(syllables_tier):
        next_interval_text = "fim"
        next_interval_dur = 0 
      else:
        next_interval_text = syllables_tier[i+1].text
        next_interval_text = syllables_tier[i+1].text.replace(" ", "")
        next_interval_dur = syllables_tier[i+1].end_time - syllables_tier[i+1].start_time

      # Calling the function to extract prosodic features for the current frame
      frame_prosodic_features = extract_prosody(frame, sr, frame_id, utterance_averages[utterance_counter], next_interval_text, next_interval_dur, interval.start_time, interval.end_time, nucleus_end_time, nucleus_start_time, nucleus_vowel, vowel_stats_dict)
      all_syllables_prosodic_features.append(frame_prosodic_features)

      # Updating current utterance
      if labels[labels_counter] == "TB":
        print("proxima i_utt:", utterance_counter, "labels i", labels_counter, labels[labels_counter])
        print(frame_id)
        print(all_utterances[utterance_counter])
        utterance_counter += 1

      labels_counter += 1

  print("Extracted features from all frames!!")

  # Organizing and saving results of the prosodic features extraction and labels at a table
  df_prosodic = pd.concat(all_syllables_prosodic_features).reset_index(drop=True)
  df_prosodic['label'] = labels

  print(df_prosodic) 
  df_prosodic.to_csv('ExtractedProsodicFeatures/'+inquiry[1]+'_prosodic_features.csv',index=False)
  #  df_prosodic.to_csv('ExtractedProsodicFeatures/versao final/'+inquiry[1]+'_prosodic_features.csv',index=False)