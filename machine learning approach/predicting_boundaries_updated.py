
import pickle
import pandas as pd
import numpy as np
import sys
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from codecarbon import EmissionsTracker
import scipy.stats as stats

#with EmissionsTracker(project_name="RF - all predictions delete this one and the 2 above") as tracker:

# MuPe-Diversidades
## FILLING DATAFRAME WITH INFO ABOUT REGION, GENDER, AGE, EDUCATION - ONLY NEEDED ONCE
def complete_info_mupe_diversidades(df_prosodic):
    # REGIÕES
    estados = ["AL", "BA", "CE", "ES", "GO", "MG", "MS", "PA", "PB", "PE", "PI", "PR", "RJ", "RO", "RS", "SE", "SP"]

    # NORDESTE
    #('AL1','AL2','BA1','BA2','CE1','CE2','PE1','PE2','PB1','PB2','PI1','SE1') 12 speakers

    # NORTE E CENTRO OESTE
    #('GO1','GO2','MS1','PA1','PA2','RO1','RO2') 7 speakers

    # SUL-SUDESTE
    # ('ES1','MG1',MG2','PR1','PR2','RJ1','RJ2','RS1','RS2','SP1','SP2') 11 speakers
    
    # ID - CHOOSE EITHER ID OR STRATIFICATION ID ACCORDING TO YOUR NEED
    #df_prosodic['ID'] = df_prosodic['stratificationID'].apply(lambda s:s.split('_')[0])
    df_prosodic['stratificationID'] = df_prosodic['ID'] + "_" + df_prosodic['label']#.apply(lambda s:s.split('_')[0])

    # Region of birth (state)
    df_prosodic['state'] = df_prosodic['stratificationID'].apply(lambda s:s.split('_')[0][:2])

    df_prosodic['region group'] = df_prosodic['state']
    df_prosodic['region group'] = np.where(df_prosodic['state'].str.startswith(('AL','BA','CE','PE','PB','PI','SE')), 'NORDESTE', df_prosodic['region group'])
    df_prosodic['region group'] = np.where(df_prosodic['state'].str.startswith(('GO','MS','PA','RO')), 'NORTE/CENTRO-OESTE', df_prosodic['region group'])
    df_prosodic['region group'] = np.where(df_prosodic['state'].str.startswith(('ES','MG','PR','RJ','RS','SP')), 'SUL/SUDESTE', df_prosodic['region group'])

    # Gender
    df_prosodic['gender'] = np.where(df_prosodic['stratificationID'].str.startswith(('AL2','BA2', 'CE1', 'GO1', 'GO2', 'MG1','PA2','PB2', 'PE1','PR1','RJ2', 'RO2','RS1', 'SP1')), 'M', 'F')

    # Age groups
    # I ('BA2','PE2','RS2','SE1','SP2')
    # II ('CE1','ES1','MG2','MS1','PA2','PB2','PI1','PR2','RJ2','RS1')
    # III ('AL1','AL2','BA1','CE2','GO1','GO2','MG1','PA1','PE1','PB1','PR1','RJ1','RO1','RO2','SP1')

    df_prosodic['age group'] = df_prosodic['stratificationID']
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

    df_prosodic['education'] = df_prosodic['ID']
    df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('ES1','PE2','RS1','RS2')), 'IB', df_prosodic['education'])
    df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('PE1','PR1','RJ2','SE1')), 'TE', df_prosodic['education'])
    df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('GO2','MG2','MS1','PI1','PR2')), 'CB', df_prosodic['education'])
    df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('AL1','BA1','MG1','PB1','PB2','RO1','RO2','SP1')), 'NE', df_prosodic['education'])
    df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('BA2','CE1','CE2','PA1','RJ1')), 'IES', df_prosodic['education'])
    df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('GO1','PA2')), 'CES', df_prosodic['education'])
    df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('AL2')), 'IE', df_prosodic['education'])
    df_prosodic['education'] = np.where(df_prosodic['stratificationID'].str.startswith(('SP2')), 'M', df_prosodic['education'])

    df_prosodic['education group'] = df_prosodic['education']
    df_prosodic['education group'] = np.where(df_prosodic['education'].str.startswith(('NE')), 'I', df_prosodic['education group'])
    df_prosodic['education group'] = np.where(df_prosodic['education'].str.startswith(('IES','CES','IE')), 'II', df_prosodic['education group'])
    df_prosodic['education group'] = np.where(df_prosodic['education'].str.startswith(('IB','TE')), 'III', df_prosodic['education group'])
    df_prosodic['education group'] = np.where(df_prosodic['education'].str.startswith(('CB','M')), 'IV', df_prosodic['education group'])

    # SAVING ALL THIS INFO AT THE TABLE
    #df_prosodic.to_csv('MuPe-Diversidades_.csv', index=False)
    df_prosodic.to_csv('MuPe-Diversidades_newufpalign_8features_non-filtered.csv', index=False)

    print(df_prosodic['age group'])
    print(df_prosodic['gender'])
    print(df_prosodic['state'])
    print(df_prosodic['education'])
    print(df_prosodic['region group'])
    print(df_prosodic['education group'])
    return df_prosodic

# CHOOSE VERSION OF MODEL YOU WISH TO USE AND COMMENT THE OTHER TWO BLOCKS

# 8 FEATURES OLD UFPALIGN ALL MUPE DIVERSIDADES
#features = ['p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] # WITHOUT F0 AVG UTT DIFF
#scaler = pickle.load(open('ModelsAndScalers/scaler_all_mupe-diversidades_8features.pkl', 'rb'))
#chosen_model = pickle.load(open('ModelsAndScalers/RF_all_mupe-diversidades_8features.pkl', 'rb')) 
#df_prosodic = pd.read_csv('Mupe-DiversidadesCSVs/MuPe-Diversidades_original_8features.csv')

# 8 FEATURES NEW UFPALIGN ALL MUPE DIVERSIDADES
#features = ['p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] # WITHOUT F0 AVG UTT DIFF
#scaler = pickle.load(open('scaler_all_mupe-diversidades_8features_newufpalign.pkl', 'rb'))
#chosen_model = pickle.load(open('RF_all_mupe-diversidades_8features_newufpalign.pkl', 'rb')) 
#df_prosodic = pd.read_csv('MuPe-Diversidades_newufpalign_8features.csv')

# 8 FEATURES NEW UFPALIGN ALL MUPE DIVERSIDADES NON FILTERED
#features = ['p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] # WITHOUT F0 AVG UTT DIFF
#scaler = pickle.load(open('ModelsAndScalers/scaler_all_mupe-diversidades_8features_newufpalign_non-filtered.pkl', 'rb'))
#chosen_model = pickle.load(open('ModelsAndScalers/RF_all_mupe-diversidades_8features_newufpalign_non-filtered.pkl', 'rb')) 
#df_prosodic = pd.read_csv('MuPe-Diversidades_newufpalign_8features_non-filtered.csv')

# 8 FEATURES NEW UFPALIGN ONLY TRAINSET NON FILTERED
#features = ['p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] # WITHOUT F0 AVG UTT DIFF
#scaler = pickle.load(open('ModelsAndScalers/scaler_only-trainset_8features_newufpalign_non-filtered.pkl', 'rb'))
#chosen_model = pickle.load(open('ModelsAndScalers/RF_only-trainset_8features_newufpalign_non-filtered.pkl', 'rb')) 
#df_prosodic = pd.read_csv('Mupe-DiversidadesCSVs/MuPe-Diversidades_newufpalign_8features_non-filtered.csv')

# 8 FEATURES NEW UFPALIGN ONLY TRAINSET FILTERED INTERVIEWER'S SPEECH
#features = ['p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] # WITHOUT F0 AVG UTT DIFF
#scaler = pickle.load(open('ModelsAndScalers/scaler_only-trainset_8features_newufpalign.pkl', 'rb'))
#chosen_model = pickle.load(open('ModelsAndScalers/RF_only-trainset_8features_newufpalign.pkl', 'rb')) 
#df_prosodic = pd.read_csv('Mupe-DiversidadesCSVs/MuPe-Diversidades_newufpalign_8features.csv') 

# 9 FEATURES ORIGINAL F0_AVGUTT_DIFF ONLY TRAINSET OLD UFPALIGN
#features = ['f0_avgutt_diff','p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] 
#scaler = pickle.load(open('scaler_prosodic_only-trainset_mupe-diversidades_9features_originalf0_avg_utt_diff.pkl', 'rb'))
#chosen_model = pickle.load(open('RF_only-trainset_mupe-diversidades_9features_originalf0_avg_utt_diff.pkl', 'rb')) 
#df_prosodic = pd.read_csv('MuPe-Diversidades_original_9features.csv')

# 9 FEATURES NEW VERSION OF FO_AVGUTT_DIFF: F0_AVGUTT_DIFF_2 ALL MUPE DIVERSIDADES NEW UFPALIGN
#features = ['f0_avgutt_diff_2','p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] 
#scaler = pickle.load(open('scaler_prosodic_all_mupe-diversidades_9features_new-f0_avg_utt_diff_2.pkl', 'rb'))
#chosen_model = pickle.load(open('RF_all_mupe-diversidades_9features_new-f0_avg_utt_diff_2.pkl', 'rb')) # _midwordcutadjusted
#df_prosodic = pd.read_csv('Mupe-DiversidadesCSVs/MuPe-Diversidades_NewUfpalign_NewFeature_Corrigido.csv')

# ---
# 9 FEATURES NEW VERSION OF FO_AVGUTT_DIFF: F0_AVGUTT_DIFF_2 ONLY TRAINSET NEW UFPALIGN
#features = ['f0_avgutt_diff_2','p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] 
#scaler = pickle.load(open('ModelsAndScalers/scaler_only-trainset_9features_newufpalign_new-f0_avg_utt_diff_2.pkl', 'rb'))
#chosen_model = pickle.load(open('ModelsAndScalers/RF_only-trainset_9features_newufpalign_new-f0_avg_utt_diff_2.pkl', 'rb')) # _midwordcutadjusted
#df_prosodic = pd.read_csv('Mupe-DiversidadesCSVs/MuPe-Diversidades_NewUfpalign_NewFeature_Corrigido.csv')#'MuPe-Diversidades_newufpalign_9features_newfeature_trainset.csv')

# 9 FEATURES NEW VERSION OF FO_AVGUTT_DIFF: F0_AVGUTT_DIFF_2 - only trainset - non filtered NEW UFPALIGN 
#features = ['f0_avgutt_diff_2','p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] 
#scaler = pickle.load(open('ModelsAndScalers/scaler_only-trainset_9features_newufpalign_new-f0_avg_utt_diff_2_nonfiltered.pkl', 'rb'))
#chosen_model = pickle.load(open('ModelsAndScalers/RF_only-trainset_9features_newufpalign_new-f0_avg_utt_diff_2_nonfiltered.pkl', 'rb')) # _midwordcutadjusted
#df_prosodic = pd.read_csv('Mupe-DiversidadesCSVs/MuPe-Diversidades_newufpalign_9features_newfeature_trainset_nonfiltered.csv')

# MINIMUM CORPUS, 8 FEATURES, only trainset, non filtered, NEW UFPALIGN 
features = ['p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] 
scaler = pickle.load(open('ModelsAndScalers/scaler_MC_8features_newufpalign_nonfiltered.pkl', 'rb'))
chosen_model = pickle.load(open('ModelsAndScalers/RF_MC_8features_newufpalign_nonfiltered.pkl', 'rb')) # _midwordcutadjusted


#################################################################################

# Predicting general results for datasets not used in training

# TO PREDICT RESULTS FOR A SINGLE FILE UNCOMMENT THIS BLOCK AND GIVE IT AS AN ARGUMENT
#if len(sys.argv) < 2:
#    print("Missing audio or textgrid filenames, please write them like this when you run the code: python3 mycode.py myfeatures.csv")
#    sys.exit(1)
 # Adapt here according to the path of your audio and textgrid generated by ufpalign
#csv_path = sys.argv[1] # for example: 'MuPe-Diversidades_newufpalign_8features_only-trainset.csv'
##########

# TO CREATE A CSV FILE CONTAINING FEATURES OF MULTIPLE FILES:
# CHOOSE ONE OF THE FOLLOWING PATHS OR UPDATE IT
#path = "./CM/CM_to_predict/CurrentInquiry" # here all files must be inside folder "CurrentInquiry", which is a folder inside folder "CM_to_predict"
path = "./MupeDiversidades_to_predict"
#path = "./CoralBrasil_to_predict"
features_csv_files = []
for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        features_csv_files.append(full_path) # full_path.split("/")[2]
features_csv_files.sort()
#print(features_csv_files)

# COMPLETING INFO ABOUT MUPE-DIVERSIDADES
# ONLY NEEDED ONCE

#for csv_file_name in features_csv_files:
#    print(csv_file_name)
#    id = csv_file_name.split("/")[2].split("_")[0]
#    print(id)
#    df_prosodic = pd.read_csv(csv_file_name)
#    df_prosodic['ID'] = id
#    df_prosodic.to_csv(csv_file_name, index=False)
#

# IF YOU WISH TO CHECK RESULTS ON GROUPS OF FILES, UNCOMMENT THE FOLLOWING LINE -- or PREDICTING RESULTS ON ENTIRE MUPE-DIVERSIDADES, ONLY NEEDED ONCE, 
df_prosodic = pd.concat(map(pd.read_csv, features_csv_files), ignore_index=True)
print(df_prosodic)

# COMPLETING INFO ABOUT MUPE-DIVERSIDADES - ONLY NEEDED ONCE
df_prosodic = complete_info_mupe_diversidades(df_prosodic)

###########
# ALL CORPUS CM
#df_prosodic.to_csv("CM_corpus_features.csv", index=False)
#csv_path = "CM_corpus_features.csv"
########
#################################################################################################

#df_prosodic.to_csv("CM_corpus_features_testset.csv", index=False) # you either need this line and the preceeding block that reads all separate files inside Current inquiry, or the following line
#df_prosodic = pd.read_csv('CM_corpus_features_testset.csv')
#print(df_prosodic)
# Predicting results on full dataset
X_test = df_prosodic[features]        
X_test = df_prosodic[features].fillna(0) # Replace NaN values with 0 in X
y_test = df_prosodic['label'] 

# CHOOSE - COMMENT IF NOT MUPE DIVERSIDADES
stratification_ids = df_prosodic['stratificationID'] # MUPE-DIVERSIDADES

# CHOOSE COMMENT IF NOT MUPE DIVERSIDADES
# Predicting results for the test set (ONLY IF THERE WAS SEPARATION OF TRAIN SET AND TEST SET)
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, train_size=0.8, test_size=0.2, random_state=seed, shuffle=True, stratify=stratification_ids)

X_test = scaler.fit_transform(X_test)
y_pred = chosen_model.predict(X_test) 

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="TB")  # Specify the positive class
recall = recall_score(y_test, y_pred, pos_label="TB")
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, pos_label="TB")

print(f"F1 Score binary: {f1:.4f}")
print(f"F1 Score macro: {f1_macro:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (TBs only): {precision:.4f}")
print(f"Recall(sensitivity) (TBs only): {recall:.4f}")
print(f"F1 Score micro: {f1_micro:.4f}")

print("Results considering both TBs and NBs (no boundary)")
print(classification_report(y_test, y_pred, target_names=["NB", "TB"]))

# Calculate SER
# Get confusion matrix: [[TN, FP], [FN, TP]]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=["NB", "TB"]).ravel()
print("True negatives: ", tn)
print("True positives: ", tp)
print("False positives: ", fp)
print("False negatives: ", fn)
print("Specificity: ", tn /(tn + fp))
ser = (fp + fn) / (tp + fn)  # Using your formula
print(f"Slot Error Rate (SER): {ser:.4f}")

#############################################################################################3

#print(X)
#print(y)
#print(stratification_ids)

#X_names = X.columns # to preserve features names
#y_name = y.name # to preserve column name 'label'

# ONLY IF THERE WAS SEPARATION OF TRAIN SET AND TEST SET
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=seed, shuffle=True, stratify=stratification_ids)

# Predicting results on specific groups

#train_indices = X_train.index  # These are from the original DataFrame
#test_indices = X_test.index

#print(train_indices)
#print(sorted(train_indices)[:20]) # tested to visualize the first rows contained in trainset 
#print(test_indices)
#print(y_test)

# USED ONLY ONCE TO CREATE A NEW COLUMN INDICATING WHETHER THE SYLLABLE BELONGS TO TEST SET OR TRAIN SET
#print(df_prosodic)
#df_prosodic['set'] = np.where(df_prosodic.index.isin(train_indices), 'TRAIN SET', 'TEST SET')
#print(df_prosodic)
#df_prosodic.to_csv('MuPe-Diversidades.csv', index=False)
#quit()

# MAYBE SHOULD BE COMMENTED, MAYBE NOT - HAVE TO ANALYZE
#df_prosodic['education group'] = np.where(df_prosodic['education'].str.startswith(('CB','M')), 'IV', df_prosodic['education group'])
#org_info_exc['frame'].replace(',', '-', inplace=True)


# Predicting results on specific states

""" # RESULTADOS POR ESTADO - DESCONTINUADO
test_states = df_prosodic.loc[test_indices, 'state'] # test set com informação dos estados
print(test_states)
grouped_states = test_states.groupby(test_states.values)
states_indexes = []
for i,state in enumerate(estados):
    print()
    print("ESTADO ",state)

    states_indexes.append(grouped_states.groups[state])
    #print(states_indexes)
    #print(states_indexes[i])

    current_state_features = df_prosodic.loc[states_indexes[i], features]
    current_state_features = scaler.fit_transform(current_state_features)
    y_test = df_prosodic.loc[states_indexes[i], 'label']

    y_pred = chosen_model.predict(current_state_features) 


# Predicting results per speaker - to enable calculating statistical relevance
print(df_prosodic.columns)
speakers_f1 = []
test_speaker = df_prosodic.loc[test_indices, 'ID'] # test set com informação dos falantes
print(test_speaker)
grouped_individual_speaker = test_speaker.groupby(test_speaker.values)

# debug print

#for index, value in grouped_individual_speaker:
#    print(index)
#    print(value)

speaker_indexes = []
for i,speaker in enumerate(['AL1','AL2','BA1', 'BA2', 'CE1', 'CE2', 'ES1', 'GO1', 'GO2', 'MG1', 'MG2', 'MS1', 'PA1', 'PA2', 'PB1', 'PB2', 'PE1', 'PE2', 'PI1', 'PR1', 'PR2', 'RJ1', 'RJ2', 'RO1', 'RO2', 'RS1', 'RS2', 'SE1', 'SP1', 'SP2']):
    print()
    print("SPEAKER ",speaker)

    speaker_indexes.append(grouped_individual_speaker.groups[speaker])
    print(speaker_indexes[i])

    current_speaker_features = df_prosodic.loc[speaker_indexes[i], features]
    current_speaker_features = scaler.fit_transform(current_speaker_features)
    y_test = df_prosodic.loc[speaker_indexes[i], 'label']

    y_pred = chosen_model.predict(current_speaker_features) 

    f1 = f1_score(y_test, y_pred, pos_label="TB")
    print(f"F1 Score binary: {f1:.4f}")

    speakers_f1.append([speaker,f1])
    print(speakers_f1)
print(speakers_f1)


####################################################################
# Predicting results on specific regions
print(df_prosodic.columns)
test_regions = df_prosodic.loc[test_indices, 'region group'] # test set com informação dos estados
#print(test_regions)
grouped_regions = test_regions.groupby(test_regions.values)

# debug print
#for index, value in grouped_regions:
#    print(index)
#    print(value)

regions_indexes = []
for i,region in enumerate(['NORDESTE','NORTE/CENTRO-OESTE','SUL/SUDESTE']):
    print()
    print("REGIÃO ",region)

    regions_indexes.append(grouped_regions.groups[region])
    #print(regions_indexes)
    #print(regions_indexes[i])

    current_region_features = df_prosodic.loc[regions_indexes[i], features]
    current_region_features = scaler.fit_transform(current_region_features)
    y_test = df_prosodic.loc[regions_indexes[i], 'label']

    y_pred = chosen_model.predict(current_region_features) 


###### RESULTADOS

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="TB")  # Specify the positive class
    recall = recall_score(y_test, y_pred, pos_label="TB")
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, pos_label="TB")

    print(f"F1 Score binary: {f1:.4f}")
    print(f"F1 Score macro: {f1_macro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score micro: {f1_micro:.4f}")

    print(classification_report(y_test, y_pred, target_names=["NB", "TB"]))

    # Calculate SER
    # Get confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=["NB", "TB"]).ravel()
    ser = (fp + fn) / (tp + fn)  # Using your formula
    print(f"Slot Error Rate (SER): {ser:.4f}")

# Calculating statistical relevance
NE_f1 = []
NMW_f1 = []
SSE_f1 = []
for speaker in speakers_f1:
    if speaker[0] in ['AL1','AL2','BA1','BA2','CE1','CE2','PE1','PE2','PB1','PB2','PI1','SE1']:
        print(speaker[0], speaker[1], 'NORDESTE')
        NE_f1.append(speaker[1])
    elif speaker[0] in ['GO1','GO2','MS1','PA1','PA2','RO1','RO2']:
        print(speaker[0], speaker[1], 'NORTE/MIDWEST')
        NMW_f1.append(speaker[1])
    else:
        print(speaker[0], speaker[1], 'SOUTH/SOUTHEAST')
        SSE_f1.append(speaker[1])

print(NE_f1)
print(NMW_f1)
print(SSE_f1)
statistical_relevance_region = stats.f_oneway(NE_f1, NMW_f1,SSE_f1)
print(statistical_relevance_region)


# Predicting results on specific genders

test_gender = df_prosodic.loc[test_indices, 'gender'] # test set com informação dos estados
#print(test_gender)
grouped_genders = test_gender.groupby(test_gender.values)

# debug print
#for index, value in grouped_regions:
#    print(index)
#    print(value)

gender_indexes = []
for i,gender in enumerate(['M','F']):
    print()
    print("GÊNERO ",gender)

    gender_indexes.append(grouped_genders.groups[gender])
    #print(regions_indexes)
    #print(regions_indexes[i])

    current_gender_features = df_prosodic.loc[gender_indexes[i], features]
    current_gender_features = scaler.fit_transform(current_gender_features)
    y_test = df_prosodic.loc[gender_indexes[i], 'label']

    y_pred = chosen_model.predict(current_gender_features) 


###### RESULTADOS

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="TB")  # Specify the positive class
    recall = recall_score(y_test, y_pred, pos_label="TB")
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, pos_label="TB")

    print(f"F1 Score binary: {f1:.4f}")
    print(f"F1 Score macro: {f1_macro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score micro: {f1_micro:.4f}")

    print(classification_report(y_test, y_pred, target_names=["NB", "TB"]))

    # Calculate SER
    # Get confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=["NB", "TB"]).ravel()
    ser = (fp + fn) / (tp + fn)  # Using your formula
    print(f"Slot Error Rate (SER): {ser:.4f}")

# Calculating statistical relevance
Male_f1 = []
Female_f1 = []
for speaker in speakers_f1:
    if speaker[0] in ['AL2','BA2', 'CE1', 'GO1', 'GO2', 'MG1','PA2','PB2', 'PE1','PR1','RJ2', 'RO2','RS1', 'SP1']:
        print(speaker[0], speaker[1], 'Male')
        Male_f1.append(speaker[1])
    else:
        print(speaker[0], speaker[1], 'Female')
        Female_f1.append(speaker[1])
#stats.f_oneway(df['Coluna1'], df['Coluna2'],df['Coluna3'])
print(Male_f1)
print(Female_f1)
statistical_relevance_gender = stats.f_oneway(Male_f1,Female_f1)
statistical_relevance_gender2 = stats.ttest_ind(Male_f1, Female_f1)
print("anova", statistical_relevance_gender)
print("t test", statistical_relevance_gender2)


# Predicting results on specific age groups

test_age_group = df_prosodic.loc[test_indices, 'age group'] # test set com informação dos estados
#print(test_age_group)
grouped_age_groups = test_age_group.groupby(test_age_group.values)

# debug print
#for index, value in grouped_regions:
#    print(index)
#    print(value)

age_group_indexes = []
for i,age_group in enumerate(['I','II','III']):
    print()
    print("AGE GROUP ",age_group)

    age_group_indexes.append(grouped_age_groups.groups[age_group])
    #print(age_group_indexes)
    

    current_age_group_features = df_prosodic.loc[age_group_indexes[i], features]
    current_age_group_features = scaler.fit_transform(current_age_group_features)
    y_test = df_prosodic.loc[age_group_indexes[i], 'label']

    y_pred = chosen_model.predict(current_age_group_features) 


###### RESULTADOS

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="TB")  # Specify the positive class
    recall = recall_score(y_test, y_pred, pos_label="TB")
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, pos_label="TB")

    print(f"F1 Score binary: {f1:.4f}")
    print(f"F1 Score macro: {f1_macro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score micro: {f1_micro:.4f}")

    print(classification_report(y_test, y_pred, target_names=["NB", "TB"]))

    # Calculate SER
    # Get confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=["NB", "TB"]).ravel()
    ser = (fp + fn) / (tp + fn)  # Using your formula
    print(f"Slot Error Rate (SER): {ser:.4f}")

# Calculating statistical relevance
age_group_I_f1 = []
age_group_II_f1 = []
age_group_III_f1 = []
for speaker in speakers_f1:
    if speaker[0] in ['BA2','PE2','RS2','SE1','SP2']:
        print(speaker[0], speaker[1], 'I')
        age_group_I_f1.append(speaker[1])
    elif speaker[0] in ['CE1','ES1','MG2','MS1','PA2','PB2','PI1','PR2','RJ2','RS1']:
        print(speaker[0], speaker[1], 'II')
        age_group_II_f1.append(speaker[1])
    else:
        print(speaker[0], speaker[1], 'III')
        age_group_III_f1.append(speaker[1])

print(age_group_I_f1)
print(age_group_II_f1)
print(age_group_III_f1)
statistical_relevance_age = stats.f_oneway(age_group_I_f1, age_group_II_f1, age_group_III_f1)
print(statistical_relevance_age)


# Predicting results on specific education groups 

test_education_group = df_prosodic.loc[test_indices, 'education group'] # test set com informação dos estados
#print(test_education_group)
grouped_education_groups = test_education_group.groupby(test_education_group.values)

# debug print
#for index, value in grouped_regions:
#    print(index)
#    print(value)

education_group_indexes = []
for i,education_group in enumerate(['I','II','III','IV']):
    print()
    print("EDUCATION GROUP ",education_group)

    education_group_indexes.append(grouped_education_groups.groups[education_group])

    current_education_group_features = df_prosodic.loc[education_group_indexes[i], features]
    current_education_group_features = scaler.fit_transform(current_education_group_features)
    y_test = df_prosodic.loc[education_group_indexes[i], 'label']

    y_pred = chosen_model.predict(current_education_group_features) 


###### RESULTADOS

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="TB")  # Specify the positive class
    recall = recall_score(y_test, y_pred, pos_label="TB")
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, pos_label="TB")

    print(f"F1 Score binary: {f1:.4f}")
    print(f"F1 Score macro: {f1_macro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score micro: {f1_micro:.4f}")

    print(classification_report(y_test, y_pred, target_names=["NB", "TB"]))

    # Calculate SER
    # Get confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=["NB", "TB"]).ravel()
    ser = (fp + fn) / (tp + fn)  # Using your formula
    print(f"Slot Error Rate (SER): {ser:.4f}")

print(education_group_indexes)

# Education
# NE ('AL1','BA1','MG1','PB1','PB2','RO1','RO2','SP1') 8 speakers

# IES ('AL2','BA2','CE1','CE2','PA1','RJ1') 
# CES ('GO1','PA2') 8 speakers

# IB ('ES1','PE2','RS1','RS2') 8 speakers
# TE ('PE1','PR1','RJ2','SE1')

# CB ('GO2','MG2','MS1','PI1','PR2') 6 speakers
# M ('SP2')
# Calculating statistical relevance
education_group_I_f1 = []
education_group_II_f1 = []
education_group_III_f1 = []
education_group_IV_f1 = []
for speaker in speakers_f1:
    if speaker[0] in ['AL1','BA1','MG1','PB1','PB2','RO1','RO2','SP1']:
        print(speaker[0], speaker[1], 'EDUCATION GROUP I')
        education_group_I_f1.append(speaker[1])
    elif speaker[0] in ['AL2','BA2','CE1','CE2','PA1','RJ1','GO1','PA2']:
        print(speaker[0], speaker[1], 'EDUCATION GROUP II')
        education_group_II_f1.append(speaker[1])
    elif speaker[0] in ['ES1','PE2','RS1','RS2','PE1','PR1','RJ2','SE1']:
        print(speaker[0], speaker[1], 'EDUCATION GROUP III')
        education_group_III_f1.append(speaker[1])
    else:
        print(speaker[0], speaker[1], 'education group IV')
        education_group_IV_f1.append(speaker[1])

print(education_group_I_f1)
print(education_group_II_f1)
print(education_group_III_f1)
print(education_group_IV_f1)
statistical_relevance_education = stats.f_oneway(education_group_I_f1, education_group_II_f1, education_group_III_f1, education_group_IV_f1)
print(statistical_relevance_education)

print("################################################")
print("Statistical Relevance overall")
print("REGION")
print(statistical_relevance_region)
print("GENDER")
print(statistical_relevance_gender)
print(statistical_relevance_gender2)
print("AGE")
print(statistical_relevance_age)
print("EDUCATION")
print(statistical_relevance_education)
######################################
"""

###############################################################33

#print(y_pred.tolist()) 
#y_prob = chosen_model.predict_proba(X_test) # prediction probabilities
#print(y_prob.tolist())

# printing comparison among predicted label and true label
#for true_label, predicted_label in zip(y_test, y_pred):
#    print(f"True: {true_label}, Predicted: {predicted_label}")