import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def join_inquiries_in_single_dataset():
    estados = ["AL", "BA", "CE", "ES", "GO", "MG", "MS", "PA", "PB", "PE", "PI", "PR", "RJ", "RO", "RS", "SE", "SP"]
    numeros = ["1", "2"]
    all_X = [] # prosodic features from all syllables of all audios
    y = [] # labels from all syllables of all audios
    frames = []
    all_stratification_ids = []
    for estado in estados:
        for numero in numeros:
            audio_id = estado+numero
            # Reading csv file with prosodic features  extracted from each syllable of the original audio
            try:
                df_prosodic = pd.read_csv('ExtractedProsodicFeatures/'+audio_id+'_prosodic_features_filtered_speakers.csv')
                #df_prosodic = pd.read_csv('ExtractedProsodicFeatures/'+audio_id+'_prosodic_features_filtered_speakers.csv')
                current_X = df_prosodic[features]        
                current_X = df_prosodic[features].fillna(0) # Replace NaN values with 0 in X
                current_X = current_X.values.tolist()
                all_X.extend(current_X)
                current_y = df_prosodic.label.to_list() # Label column must be filled with labels, such as "TB", "NB" at .csv file
                y.extend(current_y)
                current_frame = df_prosodic['frame'].to_list() # Label column must be filled with labels, such as "TB", "NB" at .csv file
                frames.extend(current_frame)
                df_prosodic['stratificationID'] = audio_id + "_" + df_prosodic['label'] 
                current_stratification_id = df_prosodic['stratificationID'].to_list()
                all_stratification_ids.extend(current_stratification_id) 
            except:
                print(audio_id, "doesn't exist, skipping to the next one")
                continue

    mupe_diversidades = pd.DataFrame(all_X, columns=features)
    mupe_diversidades["frame"] = frames
    mupe_diversidades["label"] = y
    mupe_diversidades["stratificationID"] = all_stratification_ids
    mupe_diversidades.to_csv('MuPe-Diversidades_corrigido.csv', index=False) 

    X = pd.DataFrame(all_X)
    
    return X, y, all_stratification_ids

features = ['f0_avgutt_diff','p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff']
seed = 42
X, y, all_stratification_ids = join_inquiries_in_single_dataset()
df_prosodic = pd.read_csv('MuPe-Diversidades_corrigido.csv')


#estados = ["AL", "BA", "CE", "ES", "GO", "MG", "MS", "PA", "PB", "PE", "PI", "PR", "RJ", "RO", "RS", "SE", "SP"]
#X = df_prosodic[features]        
#X = df_prosodic[features].fillna(0) # Replace NaN values with 0 in X
#y = df_prosodic['label'] 
#stratification_ids = df_prosodic['stratificationID']

## FILLING DATAFRAME WITH INFO ABOUT REGION, GENDER, AGE, EDUCATION - ONLY NEEDED ONCE

# REGIÕES

# NORDESTE
#('AL1','AL2','BA1','BA2','CE1','CE2','PE1','PE2','PB1','PB2','PI1','SE1') 12 speakers

# NORTE E CENTRO OESTE
#('GO1','GO2','MS1','PA1','PA2','RO1','RO2') 7 speakers

# SUL-SUDESTE
# ('ES1','MG1',MG2','PR1','PR2','RJ1','RJ2','RS1','RS2','SP1','SP2') 11 speakers

# ID
df_prosodic['ID'] = df_prosodic['stratificationID'].apply(lambda s:s.split('_')[0])

# Region of birth (state)
df_prosodic['state'] = df_prosodic['stratificationID'].apply(lambda s:s.split('_')[0][:2])

df_prosodic['region group'] = ''
df_prosodic['age group'] = ''
df_prosodic['education'] = ''
df_prosodic['education group'] = ''

df_prosodic['region group'] = np.where(df_prosodic['state'].str.startswith(('AL','BA','CE','PE','PB','PI','SE')), 'NORDESTE', df_prosodic['region group'])
df_prosodic['region group'] = np.where(df_prosodic['state'].str.startswith(('GO','MS','PA','RO')), 'NORTE/CENTRO-OESTE', df_prosodic['region group'])
df_prosodic['region group'] = np.where(df_prosodic['state'].str.startswith(('ES','MG','PR','RJ','RS','SP')), 'SUL/SUDESTE', df_prosodic['region group'])

# Gender
df_prosodic['gender'] = np.where(df_prosodic['stratificationID'].str.startswith(('AL2','BA2', 'CE1', 'GO1', 'GO2', 'MG1','PA2','PB2', 'PE1','PR1','RJ2', 'RO2','RS1', 'SP1')), 'M', 'F')

# Age groups
# I ('BA2','PE2','RS2','SE1','SP2')
# II ('CE1','ES1','MG2','MS1','PA2','PB2','PI1','PR2','RJ2','RS1')
# III ('AL1','AL2','BA1','CE2','GO1','GO2','MG1','PA1','PE1','PB1','PR1','RJ1','RO1','RO2','SP1')

df_prosodic['age group'] = np.where(df_prosodic['stratificationID'].str.startswith(('BA2','PE2','RS2','SE1','SP2')), 'I', df_prosodic['age group'])
df_prosodic['age group'] = np.where(df_prosodic['stratificationID'].str.startswith(('CE1','ES1','MG2','MS1','PA2','PB2','PI1','PR2','RJ2','RS1')), 'II', df_prosodic['age group'])
df_prosodic['age group'] = np.where(df_prosodic['stratificationID'].str.startswith(('AL1','AL2','BA1','CE2','GO1','GO2','MG1','PA1','PE1','PB1','PR1','RJ1','RO1','RO2','SP1')), 'III', df_prosodic['age group'])

# Education
# NE ('AL1','BA1','MG1','PB1','PB2','RO1','RO2','SP1') 8 speakers

# IES ('AL2','BA2','CE1','CE2','PA1','RJ1') 
# CES ('GO1','PA2') 8 speakers

# IB ('ES1','PE2','RS1','RS2') 8 speakers
# TE ('PE1','PR1','RJ2','SE1')

# CB ('GO2','MG2','MS1','PI1','PR2') 6 speakers
# M ('SP2')

df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('ES1','PE2','RS1','RS2')), 'IB', df_prosodic['education'])
df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('PE1','PR1','RJ2','SE1')), 'TE', df_prosodic['education'])
df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('GO2','MG2','MS1','PI1','PR2')), 'CB', df_prosodic['education'])
df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('AL1','BA1','MG1','PB1','PB2','RO1','RO2','SP1')), 'NE', df_prosodic['education'])
df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('BA2','CE1','CE2','PA1','RJ1')), 'IES', df_prosodic['education'])
df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('GO1','PA2')), 'CES', df_prosodic['education'])
df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('AL2')), 'IE', df_prosodic['education'])
df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('SP2')), 'M', df_prosodic['education'])

df_prosodic['education group'] = np.where(df_prosodic['education'].str.startswith(('NE')), 'I', df_prosodic['education group'])
df_prosodic['education group'] = np.where(df_prosodic['education'].str.startswith(('IES','CES','IE')), 'II', df_prosodic['education group'])
df_prosodic['education group'] = np.where(df_prosodic['education'].str.startswith(('IB','TE')), 'III', df_prosodic['education group'])
df_prosodic['education group'] = np.where(df_prosodic['education'].str.startswith(('CB','M')), 'IV', df_prosodic['education group'])

       
X = df_prosodic[features].fillna(0) # Replace NaN values with 0 in X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=seed, shuffle=True, stratify=all_stratification_ids)

train_indices = X_train.index  # These are from the original DataFrame
test_indices = X_test.index

df_prosodic['set'] = np.where(df_prosodic.index.isin(train_indices), 'TRAIN SET', 'TEST SET')

# SAVING ALL THIS INFO AT THE TABLE
df_prosodic.to_csv('MuPe-Diversidades_corrigido.csv', index=False)