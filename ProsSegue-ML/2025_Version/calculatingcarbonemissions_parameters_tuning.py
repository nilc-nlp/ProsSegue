# ML Testing different classifiers performance to choose most adequate ML approach to train model

# ML HYPERPARAMETERS TUNING

import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
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

features = ['f0_avgutt_diff','p_dur','n_dur','e_range','e_maxavg_diff',
            'e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff']


X, y, all_stratifications_ids = join_inquiries_in_single_dataset()


seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True, stratify=all_stratifications_ids) # stratify=y 


print("Y train set - TB total count:", y_train.count('TB'))
print("Y train set - NB total count:", y_train.count('NB'))

print("Y test set - TB total count:", y_test.count('TB'))
print("Y test set - NB total count:", y_test.count('NB'))

#seeds = [42,17,79]
#### Do not comment what came before this point
####################################################################################################

# LDA
""" 
#for index,seed in enumerate(seeds):

 with EmissionsTracker(project_name="LDA - seed "+str(seed)+" - parameters tuning only train data class balanced") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    param_grid = {
        'solver': ['svd', 'lsqr'],
        'shrinkage': [None, 'auto']  # Only applicable to 'lsqr'
    }

    grid = GridSearchCV(LinearDiscriminantAnalysis(), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1) #f1_macro

    grid_result = grid.fit(X_train, y_train)
    
print("Best estimator", grid_result.best_estimator_)
print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
for rank, (params, score) in enumerate(sorted_results, 1):
    print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}")


# MLP

#for index,seed in enumerate(seeds):

with EmissionsTracker(project_name="MLP - seed "+str(seed)+" - parameters tuning only train data class balanced") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    param_grid = {
        'hidden_layer_sizes': [(25,), (50,), (50, 25)],
        'activation': ['logistic', 'relu'],
        'alpha': [0.0001, 0.001],  # L2 penalty
        'learning_rate_init': [0.001, 0.01],
    }

    grid = GridSearchCV(MLPClassifier(max_iter=500, random_state=seed, solver='adam'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1)

    grid_result = grid.fit(X_train, y_train)
    
print("Best estimator", grid_result.best_estimator_)
print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
for rank, (params, score) in enumerate(sorted_results, 1):
    print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}")

# Random Forest

#for index,seed in enumerate(seeds):

with EmissionsTracker(project_name="Random Forest - seed "+str(seed)+" - parameters tuning only train data class balanced") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=seed, class_weight='balanced'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1) # , class_weight='balanced'

    grid_result = grid.fit(X_train, y_train)

print("Best estimator", grid_result.best_estimator_)
print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
for rank, (params, score) in enumerate(sorted_results, 1):
    print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}")



# Logistic Regression

#for index,seed in enumerate(seeds):

with EmissionsTracker(project_name="Logistic Regression - seed "+str(seed)+" - parameters tuning only train data class balanced") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    param_grid = {
        'C': [0.01, 0.1, 1, 10],   # Regularization strength
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l1','l2'],         # l1 only with saga
    }

    grid = GridSearchCV(LogisticRegression(max_iter=500, random_state=seed, class_weight='balanced'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1) # class_weight='balanced'

    grid_result = grid.fit(X_train, y_train)

print("Best estimator", grid_result.best_estimator_)
print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
for rank, (params, score) in enumerate(sorted_results, 1):
    print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}")

# Gradient boosting

#for index,seed in enumerate(seeds):

with EmissionsTracker(project_name="Gradient Boosting - seed "+str(seed)+" - parameters tuning only train data class balanced") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }

    grid = GridSearchCV(GradientBoostingClassifier(random_state=seed), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1)

    grid_result = grid.fit(X_train, y_train)

print("Best estimator", grid_result.best_estimator_)
print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
for rank, (params, score) in enumerate(sorted_results, 1):
    print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}")


# Decision tree

#for index,seed in enumerate(seeds):

with EmissionsTracker(project_name="Decision tree - seed "+str(seed)+" - parameters tuning only train data class balanced") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 5]
    }

    grid = GridSearchCV(DecisionTreeClassifier(random_state=seed,class_weight='balanced'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1) #f1_macro' # , class_weight='balanced'

    grid_result = grid.fit(X_train, y_train)

print("Best estimator", grid_result.best_estimator_)
print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
for rank, (params, score) in enumerate(sorted_results, 1):
    print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}")

# SVC - Careful - if testing different kernels in param_grid, like ['rbf', 'linear', 'poly'], the code runs for days

#for index,seed in enumerate(seeds):

with EmissionsTracker(project_name="SVC - seed "+str(seed)+" - parameters tuning only train data class balanced") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.001]
    }

    grid = GridSearchCV(SVC(probability=True, random_state=seed, kernel='rbf', class_weight='balanced'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1) 

    grid_result = grid.fit(X_train, y_train)

print("Best estimator", grid_result.best_estimator_)
print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
for rank, (params, score) in enumerate(sorted_results, 1):
    print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}") """
 

######################################################################################################

###### To calculate class weights tuning emissions uncomment the following block and comment everything that came before


# Class weights

param_grid = {
    'class_weight': [
        'balanced',
        {'NB': 1, 'TB': 1},
        {'NB': 1, 'TB': 2},
        {'NB': 1, 'TB': 3},
        {'NB': 1, 'TB': 5},
        {'NB': 1, 'TB': 10},
        {'NB': 1.0, 'TB': 15},
        {'NB': 1.0, 'TB': 30},
        {'NB': 1.0, 'TB': 35},
        {'NB': 1.0, 'TB': 40},
        {'NB': 1.0, 'TB': 50}
        
    ]
}


# LDA doesn't accept class weights parameter


# MLP doesn't accept class weights parameter


# Random Forest

#for index,seed in enumerate(seeds):

with EmissionsTracker(project_name="Random Forest - seed "+str(seed)+" class weights only train data") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    grid = GridSearchCV(RandomForestClassifier(random_state=seed, max_depth=20, n_estimators=100, min_samples_split=5), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1) # , class_weight='balanced'

    grid_result = grid.fit(X_train, y_train)

    print("Best estimator", grid_result.best_estimator_)
    print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
    results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    for rank, (params, score) in enumerate(sorted_results, 1):
        print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}")



# Logistic Regression

#for index,seed in enumerate(seeds):

with EmissionsTracker(project_name="Logistic Regression - seed "+str(seed)+" class weights only train data") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    grid = GridSearchCV(LogisticRegression(max_iter=500, random_state=seed, C=0.01, penalty='l1', solver='saga'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1) # class_weight='balanced'

    grid_result = grid.fit(X_train, y_train)

    print("Best estimator", grid_result.best_estimator_)
    print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
    results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    for rank, (params, score) in enumerate(sorted_results, 1):
        print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}")

# Gradient boosting doesn't accept class weights

# Decision tree

#for index,seed in enumerate(seeds):

with EmissionsTracker(project_name="Decision tree - seed "+str(seed)+" class weights only train data") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    grid = GridSearchCV(DecisionTreeClassifier(random_state=seed, max_depth=20, min_samples_leaf=1, min_samples_split=2), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1) #f1_macro' # , class_weight='balanced'

    grid_result = grid.fit(X_train, y_train)

    print("Best estimator", grid_result.best_estimator_)
    print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
    results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    for rank, (params, score) in enumerate(sorted_results, 1):
        print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}")

# SVC 

#for index,seed in enumerate(seeds):

with EmissionsTracker(project_name="SVC - seed "+str(seed)+" class weights only train data") as tracker:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    f1_score_binary = make_scorer(f1_score, average='binary', pos_label='TB')

    grid = GridSearchCV(SVC(probability=True, random_state=seed, kernel='rbf', C=0.1, gamma=0.001), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring=f1_score_binary, n_jobs=-1) 

    grid_result = grid.fit(X_train, y_train)

    print("Best estimator", grid_result.best_estimator_)
    print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
    results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    for rank, (params, score) in enumerate(sorted_results, 1):
        print(f"Rank {rank}: F1 = {score:.4f} | Params = {params}")
