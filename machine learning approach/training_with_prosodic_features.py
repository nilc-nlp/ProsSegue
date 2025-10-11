
# Training the model using prosodic features

# We use a simple MLP implementation of scikit-learn.
# We recommend that participants explore different classification models, feature selection, data balancing, data augmentation and other techniques such as classifier ensemble.

# Source: https://colab.research.google.com/drive/1hdBMPrfk0-k0RxikBUs113RvNeI3o7j-?authuser=1#scrollTo=t45VreMWBg6W

# Related methods:
#    Luengo, I., Navas, E., Hernáez, I., & Sánchez, J. (2005). Automatic emotion recognition using prosodic parameters. In Ninth European conference on speech communication and technology.
#    Rao, K. S., Koolagudi, S. G., & Vempada, R. R. (2013). Emotion recognition from speech using global and local prosodic features. International journal of speech technology, 16(2), 143-160.

# Linear Discriminant Analysis (LDA)
# Random Forest (RF)
# Basear-se na explicação da Bárbara: https://repositorio.ufmg.br/bitstream/1843/47273/1/tese_deteccao_automatica_de_fronteiras_prosodicas_final.pdf

import pandas as pd
import numpy as np
import pickle
import time
import tracemalloc
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score

from codecarbon import EmissionsTracker


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
            print("Processing",audio_id)
            # Reading csv file with prosodic features  extracted from each syllable of the original audio
            try:
                #df_prosodic = pd.read_csv('ExtractedProsodicFeatures/articleversion_newufpalignversion_filtered/'+audio_id+'_prosodic_features__newufpalignversion_articleversion_filtered_speakers.csv') # DELETE AFTER ALL TESTS
                
                # CHOOSE ONE AND COMMENT THE OTHERS

                # 8 or 9 FEATURES ORIGINAL FEATURE 
                #df_prosodic = pd.read_csv('ExtractedProsodicFeatures/'+audio_id+'_prosodic_features_filtered_speakers.csv')

                # 8 or 9 FEATURES NEW FEATURE NEW UFPALIGN
                #df_prosodic = pd.read_csv('ExtractedProsodicFeaturesNEWUFPALIGNVERSION_newfeature/'+audio_id+'_prosodic_features_filtered_speakers_ufpalignnewversion.csv')
                
                # non filtered files new ufpalign new feature
                df_prosodic = pd.read_csv('ExtractedProsodicFeatures/new_feature_newufpalign_corrigidos/'+audio_id+'_prosodic_features.csv')
                
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

    # CHOOSE ONE OF THE FOLLOWING AND COMMENT THE OTHERS
    
    # Original version - 8 features - OLD UFPALIGN
    #mupe_diversidades.to_csv('MuPe-Diversidades_original_8features.csv', index=False) 

    # 8 features - NEW UFPALIGN
    #mupe_diversidades.to_csv('MuPe-Diversidades_newufpalign_8features.csv', index=False) 

    # 8 features - new ufpalign - trainset
    #mupe_diversidades.to_csv('MuPe-Diversidades_newufpalign_8features_only-trainset.csv', index=False)

    # 8 features - new ufpalign - all mupe diversidades - non filtered
    mupe_diversidades.to_csv('MuPe-Diversidades_newufpalign_8features_non-filtered.csv', index=False)

    # 9 features - new ufpalign - trainset - new feature
    #mupe_diversidades.to_csv('MuPe-Diversidades_newufpalign_9features_newfeature_trainset.csv', index=False) 

    # 9 features - new ufpalign - trainset - new feature
    #mupe_diversidades.to_csv('MuPe-Diversidades_newufpalign_9features_newfeature_trainset_nonfiltered.csv', index=False) 


    # Original version - 9 features
    #mupe_diversidades.to_csv('MuPe-Diversidades_original_9features.csv', index=False) 

    # 9 features new feature
    #mupe_diversidades.to_csv('MuPe-Diversidades_NewUfpalign_NewFeature_Corrigido.csv', index=False) 

    X = pd.DataFrame(all_X)
    
    return X, y, all_stratification_ids


#with EmissionsTracker(project_name="RF - training ") as tracker:

# CHOOSE ONE OF THE FOLLOWING COMBINATION OF FEATURES AND COMMENT THE OTHER TWO

# 8 FEATURES - INDICATED FOR USAGE ON NEW DATASETS
features = ['p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff']

# 9 FEATURES - ORIGINAL f0_avg_utt_diff - STIL 2025 MUPE DIVERSIDADES REPORTED RESULTS VERSION
#features = ['f0_avgutt_diff','p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff']

# 9 FEATURES - NEW f0_avg_utt_diff_2 - UPDATED VERSION, INDICATED FOR USAGE ON NEW DATASETS, RESULTS NOT YET EVALUATED
#features = ['f0_avgutt_diff_2','p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff']

# CHOOSE ONE OF THE FOLLOWING BLOCKS TO GET EXCTRACTED FEATURES INFO FROM MUPE DIVERSIDADES AND EITHER COMMENT OR (RECOMMENDED) DELETE THE OTHER 

X, y, all_stratifications_ids = join_inquiries_in_single_dataset() 
#print(all_stratifications_ids)

# If your dataset is completely contained in a single csv file, adapt the following line for the name and path of your file
#df_prosodic = pd.read_csv('MuPe-Diversidades.csv') # ADAPT NAME HERE
#X = df_prosodic[features]
#y = df_prosodic['label'].to_list()

# ANALYZING STRATIFICATION TO GUARANTEE DIVERSITY AND BALANCE OF CLASSES
print("Stratification ids total count")
classes, counts = np.unique(all_stratifications_ids, return_counts=True)
print(dict(zip(classes, counts)))

scaler = StandardScaler()
X = scaler.fit_transform(X) # While some classifiers need this step, gradient boosting and decision tree are not affected by this, but it can safely be applied to all
seed = 42

# CHECKING IF STRATIFICATION WORKED
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True, stratify=all_stratifications_ids) # stratify=y 
# checking stratification # CAREFUL, here all_stratification_ids column is attributed to y so we can see what is inside
#X_train, X_test, y_train, y_test = train_test_split(X, all_stratifications_ids, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True, stratify=all_stratifications_ids) # stratify=y 
#print("Train set stratification count")
#classes, counts = np.unique(y_train, return_counts=True)
#print(dict(zip(classes, counts)))
#print("Test set stratification count")
#classes, counts = np.unique(y_test, return_counts=True)
#print(dict(zip(classes, counts)))

# CHOOSE ONE OF THE FOLLOWING TWO DATASETS TO TRAIN THE MODEL (EITHER THE ENTIRE MUPE-DIVERSIDADES (lines 122 and 123), OR THE SELECTED TRAIN SET (line 119)) AND COMMENT THE OTHER 

# Training with 80% of the dataset
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True, stratify=all_stratifications_ids)  
#print(X_test)
#print(y_test) 
#print("Y test set - TB total count:", y_test.count('TB'))
#print("Y test set - NB total count:", y_test.count('NB'))

# Training with all data to make model available for users
X_train = X
y_train = y

print("Y train set - TB total count:", y_train.count('TB'))
print("Y train set - NB total count:", y_train.count('NB'))

# Model Training

print("Training Random Forest")

# Commented metrics measure time and memory used during training

#tracemalloc.start()
#start_time = time.time()
chosen_model = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=100, class_weight={'NB': 1.0, 'TB': 30},  random_state=seed)
chosen_model.fit(X_train,y_train)
#chosen_model.fit(X,y) # ONLY USED THIS TO MAKE AVAILABLE MODEL TRAINED WITH ENTIRE MUPE-DIVERSIDADES
#memory_usage = tracemalloc.take_snapshot()
#current, peak = tracemalloc.get_traced_memory()
#tracemalloc.stop()
#end_time = time.time()
#execution_time = end_time - start_time
#peak_memory = peak / 1024 / 1024
#print("Execution Time:", round(execution_time,2), "seconds")
#print(f"Current memory usage:{current / 1024 / 1024:.1f} MB")
#print(f"Peak memory usage: {peak_memory:.1f} MB")
#tracemalloc.reset_peak()
#tracemalloc.clear_traces()
#print("Random Forest trained")

importances = chosen_model.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [features[i] for i in indices]
print(names)
print(np.sort(importances)[::-1])

# Plots the graph- TEST THIS
""" plt.figure(figsize=(10, 6))
plt.title(str(chosen_model)+" - Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show() """

# Salva o objeto scaler que foi utilizado para padronizar os dados para garantir que a mesma transformação seja aplicada a novos dados durante a previsão.

# CHOOSE THE RIGHT SCALER AND MODEL BELOW AND COMMENT ALL OTHERS

# 8 FEATURES - NEW UFPALIGN - TRAINSET - INDICATED FOR USAGE ON NEW DATASETS
#with open('scaler_only-trainset_8features_newufpalign.pkl', 'wb') as fid_scaler:
#    pickle.dump(scaler,fid_scaler)
#with open('RF_only-trainset_8features_newufpalign.pkl', 'wb') as fid_model:
#    pickle.dump(chosen_model,fid_model)

# 8 FEATURES - NEW UFPALIGN - ALL MUPE DIVERSIDADES - NON FILTERED - OFFICIAL VERSION - INDICATED FOR USAGE ON NEW DATASETS
with open('scaler_all_mupe-diversidades_8features_newufpalign_non-filtered.pkl', 'wb') as fid_scaler:
    pickle.dump(scaler,fid_scaler)
with open('RF_all_mupe-diversidades_8features_newufpalign_non-filtered.pkl', 'wb') as fid_model:
    pickle.dump(chosen_model,fid_model)

# 9 FEATURES - NEW UFPALIGN - TRAINSET - INDICATED FOR USAGE ON NEW DATASETS
#with open('scaler_only-trainset_9features_newufpalign_new-f0_avg_utt_diff_2.pkl', 'wb') as fid_scaler:
#    pickle.dump(scaler,fid_scaler)
#with open('RF_only-trainset_9features_newufpalign_new-f0_avg_utt_diff_2.pkl', 'wb') as fid_model:
#    pickle.dump(chosen_model,fid_model)

# 9 FEATURES - NEW UFPALIGN - TRAINSET - INDICATED FOR USAGE ON NEW DATASETS
#with open('scaler_only-trainset_9features_newufpalign_new-f0_avg_utt_diff_2_nonfiltered.pkl', 'wb') as fid_scaler:
#    pickle.dump(scaler,fid_scaler)
#with open('RF_only-trainset_9features_newufpalign_new-f0_avg_utt_diff_2_nonfiltered.pkl', 'wb') as fid_model:
#    pickle.dump(chosen_model,fid_model)

# 8 FEATURES - OLD UFPALIGN - INDICATED FOR USAGE ON NEW DATASETS
#with open('scaler_all_mupe-diversidades_8features.pkl', 'wb') as fid_scaler:
#    pickle.dump(scaler,fid_scaler)
#with open('RF_all_mupe-diversidades_8features.pkl', 'wb') as fid_model:
#    pickle.dump(chosen_model,fid_model)

# 8 FEATURES - NEW UFPALIGN - INDICATED FOR USAGE ON NEW DATASETS
#with open('scaler_all_mupe-diversidades_8features_newufpalign.pkl', 'wb') as fid_scaler:
#    pickle.dump(scaler,fid_scaler)
#with open('RF_all_mupe-diversidades_8features_newufpalign.pkl', 'wb') as fid_model:
#    pickle.dump(chosen_model,fid_model)

# 9 FEATURES - ORIGINAL f0_avg_utt_diff - STIL 2025 MUPE DIVERSIDADES REPORTED RESULTS VERSION
#with open('scaler_prosodic_only-trainset_mupe-diversidades_9features_originalf0_avg_utt_diff.pkl', 'wb') as fid_scaler:
#    pickle.dump(scaler,fid_scaler)
#with open('RF_only-trainset_mupe-diversidades_9features_originalf0_avg_utt_diff.pkl', 'wb') as fid_model:
#    pickle.dump(chosen_model,fid_model)

# 9 FEATURES - NEW f0_avg_utt_diff_2 - UPDATED VERSION, INDICATED FOR USAGE ON NEW DATASETS, RESULTS NOT YET EVALUATED
#with open('scaler_prosodic_all_mupe-diversidades_9features_new-f0_avg_utt_diff_2.pkl', 'wb') as fid_scaler:
#    pickle.dump(scaler,fid_scaler)
#with open('RF_all_mupe-diversidades_9features_new-f0_avg_utt_diff_2.pkl', 'wb') as fid_model:
#    pickle.dump(chosen_model,fid_model)

# DELETE THESE AFTER ALL TESTS AND PROCESSES
#with open('scaler_prosodic_all_mupe-diversidades.pkl', 'wb') as fid_scaler:
#with open('scaler_prosodic.pkl', 'wb') as fid_scaler:
#with open('scaler_prosodic_newufpalignversion_articleversion_9features_allmupediversidades.pkl', 'wb') as fid_scaler:
#    pickle.dump(scaler,fid_scaler)
#with open('RandomForest_model_all_mupe-diversidades.pkl', 'wb') as fid_model:
#with open('RandomForest_model.pkl', 'wb') as fid_model:
#with open('RandomForest_model_newufpalignversion_articleversion_9features_allmupediversidades.pkl', 'wb') as fid_model:
#    pickle.dump(chosen_model,fid_model)