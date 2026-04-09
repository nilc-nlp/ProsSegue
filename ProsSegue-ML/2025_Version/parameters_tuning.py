# ML Testing different classifiers performance to choose most adequate ML approach to train model

# ML HYPERPARAMETERS TUNING

import time
import tracemalloc
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split

features = ['f0_avgutt_diff','p_dur','n_dur','e_range','e_maxavg_diff',
            'e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff']

estados = ["AL", "BA", "CE", "ES", "GO", "MG", "MS", "PA", "PB", "PE", "PI", "PR", "RJ", "RO", "RS", "SE", "SP"]
numeros = ["1", "2"]
all_X = [] # prosodic features from all syllables of all audios
y = [] # labels from all syllables of all audios
for estado in estados:
  for numero in numeros:
    audio_id = estado+numero
    #print("Processing",audio_id)
    # Reading csv file with prosodic features  extracted from each syllable of the original audio

    try:
        df_prosodic = pd.read_csv('/home/giovana/Documentos/Mestrado/ProsSeguePastaLocal/ExtractedProsodicFeatures/'+audio_id+'_prosodic_features_filtered_speakers.csv')

        current_X = df_prosodic[features]        
        current_X = df_prosodic[features].fillna(0) # Replace NaN values with 0 in X
        current_X = current_X.values.tolist()
        all_X.extend(current_X)

        current_y = df_prosodic.label.to_list() # Label column must be filled with labels, such as "TB", "NB" at .csv file
        y.extend(current_y)

    except:
        print(audio_id, "doesn't exist, skipping to the next one")
        continue

X = pd.DataFrame(all_X)

# Uncomment the block with the corresponding classifier if you wish to compare different values for the parameters

# SVC - Careful - if testing different kernels in param_grid, like ['rbf', 'linear', 'poly'], the code runs for days

#param_grid = {
#    'C': [0.1, 1, 10],
#    'gamma': ['scale', 0.01, 0.001]
#}

# Gradient boosting

#param_grid = {
#    'n_estimators': [100, 200],
#    'learning_rate': [0.05, 0.1],
#    'max_depth': [3, 5],
#    'subsample': [0.8, 1.0]
#}

# MLP

#param_grid = {
#    'hidden_layer_sizes': [(25,), (50,), (50, 25)],
#    'activation': ['logistic', 'relu'],
#    'alpha': [0.0001, 0.001],  # L2 penalty
#    'learning_rate_init': [0.001, 0.01],
#}

# Logistic Regression
#param_grid = {
#    'C': [0.01, 0.1, 1, 10],   # Regularization strength
#    'solver': ['lbfgs', 'saga'],
#    'penalty': ['l1','l2'],         # l1 only with saga
#}

# Random Forest
#param_grid = {
#    'n_estimators': [100, 200],
#    'max_depth': [None, 10, 20],
#    'min_samples_split': [2, 5]
#}

# Decision tree
#param_grid = {
#    'max_depth': [None, 10, 20],
#    'min_samples_split': [2, 5],
#    'min_samples_leaf': [1, 5]
#}

# LDA
#param_grid = {
#    'solver': ['svd', 'lsqr'],
#    'shrinkage': [None, 'auto']  # Only applicable to 'lsqr'
#}

scaler = StandardScaler()
X = scaler.fit_transform(X) # While some classifiers need this step, gradient boosting and decision tree are not affected by this, but it can safely be applied to all

# Uncomment the following block to tune parameters for a certain classifier
#seed = 42
#f1_macro = make_scorer(f1_score, average='macro')

# Uncomment the line with the corresponding classifier if you wish to compare different values for the parameters
#grid = GridSearchCV(SVC(probability=True, random_state=seed, class_weight='balanced', kernel='rbf'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring='f1_macro', n_jobs=-1)
#grid = GridSearchCV(MLPClassifier(max_iter=500, random_state=seed, solver='adam'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring='f1_macro', n_jobs=-1)
#grid = GridSearchCV(GradientBoostingClassifier(random_state=seed), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring='f1_macro', n_jobs=-1)
#grid = GridSearchCV(LogisticRegression(max_iter=500, random_state=seed, class_weight='balanced'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring='f1_macro', n_jobs=-1)
#grid = GridSearchCV(RandomForestClassifier(random_state=seed, class_weight='balanced'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring='f1_macro', n_jobs=-1)
#grid = GridSearchCV(DecisionTreeClassifier(random_state=seed, class_weight='balanced'), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring='f1_macro', n_jobs=-1)
#grid = GridSearchCV(LinearDiscriminantAnalysis(), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), verbose=10, scoring='f1_macro', n_jobs=-1)

# Uncomment the following block to tune parameters for a certain classifier

#grid_result = grid.fit(X, y)
#print("Best estimator", grid_result.best_estimator_)
#print("Best: %f using %s" %  (grid_result.best_score_, grid_result.best_params_), "\n")
#results = zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score'])
#sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
#for rank, (params, score) in enumerate(sorted_results, 1):
#    print(f"Rank {rank}: F1-macro = {score:.4f} | Params = {params}")


# Configuration of different models to test which one gets better results 

classifier_statistics = {
    'LinearDiscriminantAnalysis': {'f1': 0, 'peak_memory_used': 0, 'execution_time': 0},
    #'MLPClassifier_25':{'f1': 0, 'peak_memory_used': 0, 'execution_time': 0},
    'MLPClassifier':{'f1': 0, 'peak_memory_used': 0, 'execution_time': 0},
    'RandomForestClassifier':{'f1': 0, 'peak_memory_used': 0, 'execution_time': 0},
    'LogisticRegression':{'f1': 0, 'peak_memory_used': 0, 'execution_time': 0},
    'GradientBoostingClassifier':{'f1': 0, 'peak_memory_used': 0, 'execution_time': 0},
    'DecisionTreeClassifier':{'f1': 0, 'peak_memory_used': 0, 'execution_time': 0},
    'SVC':{'f1': 0, 'peak_memory_used': 0, 'execution_time': 0}
}

seeds = [42, 2025, 170017]

for index,seed in enumerate(seeds):
    print("Test ",index+1, "- seed ", seed, "\n")

    classifiers = [
        LinearDiscriminantAnalysis(), # 74,7% # linear discriminant analysis does not involve randomness
        #MLPClassifier(hidden_layer_sizes=(25,), activation='logistic', solver='adam', max_iter=200, random_state=seed), # 73,9%,  75,3% #NN classifier as described in the paper, other characteristics are set up by default
        MLPClassifier(hidden_layer_sizes=(50, 25), activation='logistic', solver='adam', max_iter=200, random_state=seed, alpha=0.0001, learning_rate_init=0.001), # MLP that outperformed others with gridsearchCV
        RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=100, random_state=seed), # 73%, 73%
        LogisticRegression(C=0.01, penalty='l1',solver='saga',random_state=seed, max_iter=500, class_weight='balanced'), # fails to converge, but reaches a value around 68%, 68% # seed only matters if solver is either saga or liblinear, default solver seems to be lbfgs
        GradientBoostingClassifier(learning_rate=0.05, max_depth=5, n_estimators=100, subsample= 1.0, random_state=seed), # 73,4%, 73,6%
        DecisionTreeClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=2, random_state=seed, class_weight='balanced'), # 68%,68%
        SVC(kernel='rbf',C=0.1,gamma=0.001,class_weight='balanced',random_state=seed), # seed only matters if probability=True, default is probability=False
    ]

    # Testing dataset with all audios for different classifiers 
    for classifier in classifiers:
        print('Running ',classifier)
        kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        tracemalloc.start()
        start_time = time.time()
        # Here, I tested 3 different types of f1_score to evaluate how they behave. The most adequate option is f1 binary, as the others may be misleading
        #scores=cross_val_score(classifier, X, y, cv=kf, scoring='f1_macro') 
        #scores=cross_val_score(classifier, X, y, cv=kf, scoring='f1_micro') 
        scores=cross_val_score(classifier, X, y, cv=kf, scoring=make_scorer(f1_score, average='binary', pos_label='TB'))
        memory_usage = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        execution_time = end_time - start_time
        peak_memory = peak / 1024 / 1024
        print("Execution Time:", round(execution_time,2), "seconds")
        print(f"Current memory usage:{current / 1024 / 1024:.1f} MB")
        print(f"Peak memory usage: {peak_memory:.1f} MB")
        f1_avg = np.mean(scores)
        print('F1 score:', scores, 'f1 mean:', round(f1_avg,3))
        print('-------')
        classifier_statistics[str(classifier).split('(')[0]]['f1'] += f1_avg
        classifier_statistics[str(classifier).split('(')[0]]['peak_memory_used'] += peak_memory
        classifier_statistics[str(classifier).split('(')[0]]['execution_time'] += execution_time

        tracemalloc.reset_peak()
tracemalloc.clear_traces() 

classifier_statistics = {
    clf: {k: v / 3 for k, v in stats.items()}
    for clf, stats in classifier_statistics.items()
}

#print(classifier_statistics)

for clf_name, stats in classifier_statistics.items():
    f1 = round(stats['f1'], 3)
    mem = round(stats['peak_memory_used'], 1)
    time = round(stats['execution_time'], 1)
    
    print(f"{clf_name}: F1 = {f1:.3f}, Memory = {mem:.1f} MB, Time = {time:.1f} s")