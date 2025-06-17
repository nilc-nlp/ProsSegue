
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
#import eli5
#from eli5.sklearn import PermutationImportance

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.neural_network import MLPClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score

#from sklearn.utils.class_weight import compute_sample_weight
#weights = y_train.map({'NB': 0.5, 'TB': 3.0})
#model.fit(X_train, y_train, sample_weight=weights) # how to use sample weights at training time
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
            #print("Processing",audio_id)
            # Reading csv file with prosodic features  extracted from each syllable of the original audio
            try:
                df_prosodic = pd.read_csv('ExtractedProsodicFeatures/'+audio_id+'_prosodic_features_filtered_speakers.csv')

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
    mupe_diversidades.to_csv('MuPe-Diversidades.csv', index=False) 

    X = pd.DataFrame(all_X)
    
    return X, y, all_stratification_ids


#with EmissionsTracker(project_name="RF - training ") as tracker:

# USE THE FOLLOWING COMMAND ONLY IF YOU HAVE LABELS AND WISH TO TRAIN WITH ALL 9 FEATURES
#features = ['f0_avgutt_diff','p_dur','n_dur','e_range','e_maxavg_diff','e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff']
# USE THE FOLLOWING COMMAND ONLY IF YOU WISH TO PREDICT PROSODIC SEGMENTATION (YOU'LL USE ONLY 8 FEATURES)

features = ['p_dur','n_dur','e_range','e_maxavg_diff',
            'e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff']


X, y, all_stratifications_ids = join_inquiries_in_single_dataset()

#print(all_stratifications_ids)

# If your dataset is completely contained in a single csv file, adapt the following line for the name and path of your file
#df_prosodic = pd.read_csv('MuPe-Diversidades.csv')
#X = df_prosodic[features]
#y = df_prosodic['label'].to_list()

#print("Stratification ids total count")
#classes, counts = np.unique(all_stratifications_ids, return_counts=True)
#print(dict(zip(classes, counts)))

scaler = StandardScaler()
X = scaler.fit_transform(X) # While some classifiers need this step, gradient boosting and decision tree are not affected by this, but it can safely be applied to all

seed = 42

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True, stratify=all_stratifications_ids) # stratify=y 

# checking stratification # CAREFUL, here all_stratification_ids column is attributed to y so we can see what is inside
#X_train, X_test, y_train, y_test = train_test_split(X, all_stratifications_ids, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True, stratify=all_stratifications_ids) # stratify=y 

#print("Train set stratification count")
#classes, counts = np.unique(y_train, return_counts=True)
#print(dict(zip(classes, counts)))

#print("Test set stratification count")
#classes, counts = np.unique(y_test, return_counts=True)
#print(dict(zip(classes, counts)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True, stratify=all_stratifications_ids) # stratify=y 

# Training with all data to make model available for users
X_train = X
y_train = y

""" print("Y train set - TB total count:", y_train.count('TB'))
print("Y train set - NB total count:", y_train.count('NB'))

print("Y test set - TB total count:", y_test.count('TB'))
print("Y test set - NB total count:", y_test.count('NB'))
"""
#print(X_test)
#print(y_test)


# Salva o objeto scaler que foi utilizado para padronizar os dados para garantir que a mesma transformação seja aplicada a novos dados durante a previsão.

#with open('scaler_prosodic_all_mupe-diversidades.pkl', 'wb') as fid_scaler:
with open('scaler_prosodic.pkl', 'wb') as fid_scaler:
    pickle.dump(scaler,fid_scaler)

# Model Training

print("Training Random Forest")
#tracemalloc.start()
#start_time = time.time()
chosen_model = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=100, class_weight={'NB': 1.0, 'TB': 30},  random_state=seed)
#chosen_model.fit(X_train,y_train)
chosen_model.fit(X,y) # ONLY USED THIS TO MAKE AVAILABLE MODEL TRAINED WITH ENTIRE MUPE-DIVERSIDADES
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

""" plt.figure(figsize=(10, 6))
plt.title(str(chosen_model)+" - Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show() """

with open('RandomForest_model_all_mupe-diversidades.pkl', 'wb') as fid_model:
#with open('RandomForest_model.pkl', 'wb') as fid_model:
    pickle.dump(chosen_model,fid_model)