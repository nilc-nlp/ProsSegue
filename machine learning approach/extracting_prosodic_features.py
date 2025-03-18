# Pré-processamento com extração de features prosódicas
# Fonte: 

# Primeira etapa: organização dos dados

# Ter uma pasta com todos os áudios dentro .wav

# Segunda etapa:
# Extração das features prosódicas

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
def extract_prosody(frame, sr, frame_id,utt_avg_pitch, next_interval_text, next_interval_dur, interval_start_time, interval_end_time, nucleus_end_time, nucleus_start_time,  nucleus_vowel, vowel_stats_dict): #, phones_per_syl): # em vez de passar o caminho do arquivo (sound_filepath), vou passar o array de janelas de 10ms do áudio 

  #sound = parselmouth.Sound(sound_filepath)
  sound = parselmouth.Sound(values=frame, sampling_frequency=sr)
  df = pd.DataFrame()

  attributes = {}

  intensity_attributes = get_intensity_attributes(sound)[0]
  pitch_attributes, _, avg_pitch = get_pitch_attributes(sound)#[0]

  attributes['f0_avgutt_diff'] = abs(avg_pitch - utt_avg_pitch)
  if next_interval_text == "sil": 
    p_dur = next_interval_dur
  else:
    p_dur = 0 
  attributes['p_dur'] = p_dur

  # normalized duration of the nucleus of the syllable (always a vowel)
  #attributes['syllable_dur'] = interval_end_time - interval_start_time

  if nucleus_vowel != None:
    vowel_type_mean = vowel_stats_dict[nucleus_vowel]["mean"]
    vowel_type_std_dev = vowel_stats_dict[nucleus_vowel]["std_dev"]

    attributes['n_dur'] = ((nucleus_end_time - nucleus_start_time) - vowel_type_mean) / vowel_type_std_dev 
  else: 
    attributes['n_dur'] = 0 # não tinha vogal núcleo na sílaba

  #attributes['n_phones'] = phones_per_syl

  #print("n_phones", phones_per_syl )
  #print(attributes['n_phones'])

  attributes.update(intensity_attributes)
  attributes.update(pitch_attributes)

  #print(attributes)

  for attribute in attributes:
    df.at[0, attribute] = attributes[attribute]
  
  #df['n_phones'] = df['n_phones'].astype(int) # transforma o n_phones em int no dataframe, mas os valores continuam aparecendo com .0 na versão final...
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

# Córpus MuPe-Diversidades

estados = ["AL", "BA", "CE", "ES", "GO", "MG", "MS", "PA", "PB", "PE", "PI", "PR", "RJ", "RO", "RS", "SE", "SP"]
numeros = ["1", "2"]

# Cria lista com nomes dos áudios
common_path = os.getcwd() + "/MuPe-Diversidades/versao-1/" # os.getcwd gets the current folder
audio_files = []
for estado in estados:
  for numero in numeros:
    audio_id = estado+numero+'.wav'
    if isfile(join(common_path+estado+"/", audio_id)): # checando se o arquivo realmente existe
      audio_files.append(estado+"/"+audio_id)

# cria lista com os nomes, ids, estados e caminhos dos áudios dentro da pasta de seu estado
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

for i, inquiry in enumerate(audios_list): # [4:5], start=4
  print("COMEÇANDO EXTRAÇÃO DE FEATURES DO INQUÉRITO:")
  print(i,inquiry)
  vowel_nucleus_not_found = False
# Ler textgrid de alinhamento fonético
  tg_phone = tg_phones_list[i] # INQUÉRITO ATUAL # Textgrid gerado pelo ufpalign
  tg_phone = tgt.io.read_textgrid(tg_phone, predict_encoding(tg_phone), include_empty_intervals=False)
  try:
    phonemes_tier = tg_phone.get_tier_by_name("fonemeas") 
  except ValueError:
    phonemes_tier = tg_phone.get_tier_by_name("fonemas")
  try:
    wordGraphemesTier = tg_phone.get_tier_by_name("palavras-grafemas")
  except ValueError:
    wordGraphemesTier = tg_phone.get_tier_by_name("grafemas") # OS TEXTGRIDS DA NOVA BRANCH VIERAM COM NOMES DE TIERS DIFERENTES
  #try: # NAO PRECISO MAIS DESSA VERIFICAÇÃO SE A CAMADA DE SÍLABAS EXISTE
  syllables_tier = tg_phone.get_tier_by_name("silabas-fonemas")
  #except ValueError:
  #  print("nao tinha camada de silabas, pulando este inquérito")
  #  continue
# ler o textgrid de referência para saber as labels e a posição das fronteiras # INQUÉRITO ATUAL
  tg_reference = tgt.io.read_textgrid(tg_reference_list[i], predict_encoding(tg_reference_list[i]), include_empty_intervals=False)
  print(tg_phones_list[i], tg_reference_list[i]) # to check if I got the correct files

  # Load the audio file
  audio = audios_list[i] # INQUÉRITO ATUAL
  audio_path = audio[3]
  print(audio_path)
  y, sr = librosa.load(audio_path, sr=None)  # sr=None ensures native sampling rate

  # DIVISÃO DO ÁUDIO POR SÍLABAS
  # Convert syllable start and end times to sample indices.

  syllable_frames = [
    y[int(interval.start_time * sr):int(interval.end_time * sr)] for interval in syllables_tier
  ]
  
  print("Qtd sílabas:", len(syllable_frames))

  for i, (frame, interval) in enumerate(zip(syllable_frames, syllables_tier)):
    interval.text = interval.text.replace(" ", "")
    start_time = round(interval.start_time, 2)
    end_time = round(interval.end_time, 2)
    frame_id = f"frame_{interval.text}_{start_time}_{end_time}"
  
    if interval.text != "sil":
      sound = parselmouth.Sound(values=frame, sampling_frequency=sr) # ORIGINAL SOUND

      padding_duration = 0.05  # 50ms of silence on each side so pitch extraction will consider edges - if pitch is extracted from the silence padding, it is considered NAN and are supposed to be ignored
      # Generate silent padding
      num_padding_samples = int(sr * padding_duration)  # Convert time to samples
      silent_padding = numpy.zeros(num_padding_samples)  # Silent array
      extended_frame = numpy.concatenate((silent_padding, frame, silent_padding))
      extended_sound = parselmouth.Sound(values=extended_frame, sampling_frequency=sr)

      intensity_attributes = get_intensity_attributes(extended_sound)[0]
      pitch_attributes, _, avg_pitch = get_pitch_attributes(extended_sound) # CHANGE TO SOUND TO GO BACK TO ORIGINAL

  # AQUI, COMO AS SÍLABAS TEM TAMANHOS DIFERENTES, ELAS TERÃO NÚMERO DE SAMPLES DIFERENTES, OS QUAIS CORRESPONDEM ÀS COLUNAS E VIRAM NaN, FIZ UM PADDING
  # Find the maximum frame length and pad them to the maximum length
  max_length = max(len(frame) for frame in syllable_frames)
  syllable_frames = [
    numpy.pad(frame, (0, max_length - len(frame)), mode='constant') for frame in syllable_frames
  ]
  
  #print("len syllable frames", len(syllable_frames))  
  #df_audio_frames = pd.DataFrame(syllable_frames) # version that doesn't work for SE1 - compare some other inquiry to see if it works well too

  # Create a DataFrame in chunks instead of all at once to solve memory problem
  chunk_size = 400  # Adjust chunk size based on available memory
  df_audio_frames = pd.DataFrame()

  for i in range(0, len(syllable_frames), chunk_size):
    print("i", i)
    chunk = pd.DataFrame(syllable_frames[i:i + chunk_size])
    df_audio_frames = pd.concat([df_audio_frames, chunk], ignore_index=True)

  # Fazer merge de todas as tiers de TB 
  TB_tiers = [tier for tier in tg_reference.tiers if tier.name.startswith("TB-") and "ponto" not in tier.name]

  all_utterances = []

  for tier in TB_tiers:
    all_utterances.extend(tier.intervals)
  all_utterances.sort(key=lambda interval: interval.start_time)
  # lista de todas as falas ordenadas de acordo com o tempo de início menor

  # calcular os tipos de todas as vogais
  vowel_types = ["a","e","i","o","u","a~","e~","i~","o~","u~", "E","O"]
  vowel_stats_dict = {vowel: {"mean": None, "std_dev": None, "durations": []} for vowel in vowel_types}

  for phone in phonemes_tier:
    if phone.text in vowel_stats_dict:
      vowel_stats_dict[phone.text]["durations"].append(phone.end_time - phone.start_time)

  for vowel_type in vowel_stats_dict:
    vowel_stats_dict[vowel_type]["mean"] = statistics.mean(vowel_stats_dict[vowel_type]["durations"])
    vowel_stats_dict[vowel_type]["std_dev"] = statistics.stdev(vowel_stats_dict[vowel_type]["durations"]) if len(vowel_stats_dict[vowel_type]["durations"]) > 1 else 0
  
  print("vowel types mean calculated", vowel_stats_dict[vowel_type]["mean"], vowel_stats_dict[vowel_type]["std_dev"])
  utterance_averages = []
  labels = []
  i_utt = 0

  # Percorrer a camada de sílabas, verificando o tempo de cada sílaba se é menor do que o tempo final da utterance atual. Em caso positivo, declara como "sem fronteira" e passa pra seguinte, em caso negativo declara como "TB" e passa para a fala seguinte e sílaba seguinte.
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

  # lista de pitch average das utterances de acordo com as camadas TB do textgrid de referência
  for i_utt, utterance in enumerate(all_utterances):
    utterance_frame = y[int(utterance.start_time * sr):int(utterance.end_time * sr)]
    sound = parselmouth.Sound(values=utterance_frame, sampling_frequency=sr)
    utt_avg = get_utterance_avg_pitch(sound)
    utterance_averages.append(utt_avg)

  #print("Lista com as médias de pitch das utterances:", utterance_averages)
  print("Qtd de médias calculadas (utterances):", len(utterance_averages))
  print("Length of all utterances", len(all_utterances))

  all_syllables_prosodic_features = []
  utterance_counter = 0
  labels_counter = 0
  phones_index = 0

  for i, (frame, interval) in enumerate(zip(syllable_frames, syllables_tier)):
    #phones_per_syl = 0
    #if utterance_counter >= len(all_utterances):
    #if i >= len(labels): # NAO SEI SE ESSE IF TEM QUE FICAR NO CÓDIGO OU NAO
    #  print("última sílaba", labels[labels_counter-1], i-1, syllables_tier[i-1])
    #  break

    if interval.text != "sil": 
      # Calculate the starting and ending time for the current frame 
      interval.text = interval.text.replace(" ", "")
      start_time = round(interval.start_time, 2)
      end_time = round(interval.end_time, 2)
      frame_id = f"frame_{interval.text}_{start_time}_{end_time}"
      
      # alinhando a camada de fones com a camada de sílabas - chegando ao primeiro fone da sílaba atual
      while phonemes_tier[phones_index].start_time < start_time:
        phones_index += 1
        #phones_per_syl += 1

      # chegando até a vogal núcleo da sílaba    
      phones_index_aux = 0
      while (not any(vowel in phonemes_tier[phones_index+phones_index_aux].text.lower() for vowel in "aeiou")) and (phonemes_tier[phones_index+phones_index_aux].end_time <= interval.end_time): # procurando o núcleo da sílaba
        phones_index_aux += 1
        #phones_per_syl += 1
      
      # checar se a vogal núcleo da sílaba foi encontrada ou se o índice já ultrapassou o tempo da sílaba (se já ultrapassou, voltamos para o fone inicial da sílaba e procuramos por j ou w como vogal núcleo)
      if phonemes_tier[phones_index+phones_index_aux].start_time >= interval.end_time or phonemes_tier[phones_index+phones_index_aux].text == "sil":
        phones_index_aux = 0
        while not any(vowel in phonemes_tier[phones_index+phones_index_aux].text.lower() for vowel in "jw") and (phonemes_tier[phones_index+phones_index_aux].end_time <= interval.end_time):
          phones_index_aux += 1
        
        if phonemes_tier[phones_index+phones_index_aux].start_time >= interval.end_time or phonemes_tier[phones_index+phones_index_aux].text == "sil":
          vogal_nucleo = None
        elif phonemes_tier[phones_index+phones_index_aux].text == "j":
          vogal_nucleo = "i"
        elif phonemes_tier[phones_index+phones_index_aux].text == "w":
          vogal_nucleo = "u"

      else:  
        vogal_nucleo = phonemes_tier[phones_index+phones_index_aux].text

      phones_index += phones_index_aux
      nucleus_start_time = phonemes_tier[phones_index].start_time
      nucleus_end_time = phonemes_tier[phones_index].end_time

      if i+1 == len(syllables_tier):
        next_interval_text = "fim"
        next_interval_dur = 0 
      else:
        next_interval_text =  syllables_tier[i+1].text
        next_interval_text = syllables_tier[i+1].text.replace(" ", "")
        next_interval_dur = syllables_tier[i+1].end_time - syllables_tier[i+1].start_time

      # Call the function to extract prosodic features for the current frame
      frame_prosodic_features = extract_prosody(frame, sr, frame_id, utterance_averages[utterance_counter], next_interval_text, next_interval_dur, interval.start_time, interval.end_time, nucleus_end_time, nucleus_start_time, vogal_nucleo, vowel_stats_dict) #, phones_per_syl) # mean pitch
      all_syllables_prosodic_features.append(frame_prosodic_features)

    # fim da utterance atingido
      if labels[labels_counter] == "TB":
        utterance_counter += 1
      labels_counter += 1

  print("Extracted features from all frames!!")

  # Organiza os resultados da análise prosódica e rótulos em uma tabela
  df_prosodic = pd.concat(all_syllables_prosodic_features).reset_index(drop=True)

  # adiciona os labels ao dataframe
  df_prosodic['label'] = labels

  print(df_prosodic) # mostra a tabela

  # Salva a tabela com as features prosódicas em um csv

  df_prosodic.to_csv('ExtractedProsodicFeatures/versao final/'+inquiry[1]+'_prosodic_features.csv',index=False)

  df_prosodic.label.hist() # faz um gráfico por categoria, acho que terei que comentar