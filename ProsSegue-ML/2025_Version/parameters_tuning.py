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

seed=42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True, stratify=all_stratifications_ids) # stratify=y 


print("Y train set - TB total count:", y_train.count('TB'))
print("Y train set - NB total count:", y_train.count('NB'))

print("Y test set - TB total count:", y_test.count('TB'))
print("Y test set - NB total count:", y_test.count('NB'))

seeds = [42, 17, 79]

#classes, counts = np.unique(y, return_counts=True)
#print(dict(zip(classes, counts)))


# Configuration of different models to test which one gets better results 


classifier_statistics = {
    'LinearDiscriminantAnalysis': {'f1_binary': [], 'f1_macro': [], 'accuracy': [], 'peak_memory_used': []}, # {'f1_binary': {'avg': 0, 'stddev': 0}, 'f1_macro': {'avg': 0, 'stddev': 0}, 'accuracy': {'avg': 0, 'stddev': 0}, 'peak_memory_used': {'avg': 0, 'stddev': 0}},#, 'execution_time': {'avg': 0, 'stddev': 0},}, # we'll get duration from carboncode
    'MLPClassifier': {'f1_binary': [], 'f1_macro': [], 'accuracy': [], 'peak_memory_used': []},
    'RandomForestClassifier': {'f1_binary': [], 'f1_macro': [], 'accuracy': [], 'peak_memory_used': []},
    'LogisticRegression': {'f1_binary': [], 'f1_macro': [], 'accuracy': [], 'peak_memory_used': []},
    'GradientBoostingClassifier': {'f1_binary': [], 'f1_macro': [], 'accuracy': [], 'peak_memory_used': []},
    'DecisionTreeClassifier': {'f1_binary': [], 'f1_macro': [], 'accuracy': [], 'peak_memory_used': []},
    'SVC': {'f1_binary': [], 'f1_macro': [], 'accuracy': [], 'peak_memory_used': []},
}

seeds = [42, 17, 79]

""" for index,seed in enumerate(seeds):
    print("Test ",index+1, "- seed ", seed, "\n")

    classifiers = [
        LinearDiscriminantAnalysis(), # 74,7% # linear discriminant analysis does not involve randomness
        MLPClassifier(hidden_layer_sizes=(50, 25), activation='logistic', solver='adam', max_iter=500, random_state=seed, alpha=0.0001, learning_rate_init=0.001), # MLP that outperformed others with gridsearchCV
        RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=100, class_weight={'NB': 1.0, 'TB': 40},  random_state=seed), 
        LogisticRegression(C=0.01, penalty='l1',solver='saga',random_state=seed, max_iter=500, class_weight={'NB': 1, 'TB': 5}), # seed only matters if solver is either saga or liblinear, default solver seems to be lbfgs
        GradientBoostingClassifier(learning_rate=0.05, max_depth=5, n_estimators=100, subsample= 1.0, random_state=seed), 
        DecisionTreeClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=2, random_state=seed, class_weight='balanced'), 
        SVC(kernel='rbf',C=0.1,gamma=0.001,probability=True,class_weight={'NB': 1, 'TB': 10},random_state=seed), # seed only matters if probability=True, default is probability=False
    ]

    # Testing dataset with all audios for different classifiers 
    for classifier in classifiers:
        print('Running ',classifier)
        
        kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }
        tracemalloc.start()
        #start_time = time.time()
        
        #with EmissionsTracker(project_name=str(classifier)+"_seed"+str(seed)) as tracker:

        # Using multiple scoring (tab or untab the following line according to whether you wish to emissions tracker)
        scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

        memory_usage = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        #end_time = time.time()
        #execution_time = end_time - start_time
        peak_memory = peak / 1024 / 1024
        #print("Execution Time:", round(execution_time,2), "seconds")
        print(f"Current memory usage:{current / 1024 / 1024:.1f} MB")
        print(f"Peak memory usage: {peak_memory:.1f} MB")
        print("Scores:")
        for key,value in scores.items():
           print(key,":",value, "Mean:", round(np.mean(value), 2))
        #print()
        
        # Updating scores calculations
        classifier_statistics[str(classifier).split('(')[0]]['f1_binary'].append(np.mean(scores['test_f1_score_binary']))
        classifier_statistics[str(classifier).split('(')[0]]['f1_macro'].append(np.mean(scores['test_f1_macro']))
        classifier_statistics[str(classifier).split('(')[0]]['accuracy'].append(np.mean(scores['test_accuracy']))
        classifier_statistics[str(classifier).split('(')[0]]['peak_memory_used'].append(peak_memory)


        tracemalloc.reset_peak()
    
        
tracemalloc.clear_traces() 

for clf, scores in classifier_statistics.items():
   print()
   print(clf)
   for score in scores:
      print(score + ' average:', round(np.average(scores[score]),3))
      print(score + ' standard deviation:', round(np.std(scores[score]),5)) """



#############################################

for index,seed in enumerate(seeds):

    print('Running ',"LDA ", seed)
    tracemalloc.start()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    classifier = LinearDiscriminantAnalysis(solver='lsqr')
    kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

    scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

    memory_usage = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory = peak / 1024 / 1024
    print(f"Current memory usage:{current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak_memory:.1f} MB")
    print("Scores:")
    for key,value in scores.items():
        print(key,":",value, "Mean:", round(np.mean(value), 2))
    print()
    
    classifier_statistics[str(classifier).split('(')[0]]['f1_binary'].append(np.mean(scores['test_f1_score_binary']))
    classifier_statistics[str(classifier).split('(')[0]]['f1_macro'].append(np.mean(scores['test_f1_macro']))
    classifier_statistics[str(classifier).split('(')[0]]['accuracy'].append(np.mean(scores['test_accuracy']))
    classifier_statistics[str(classifier).split('(')[0]]['peak_memory_used'].append(peak_memory)

    tracemalloc.reset_peak()

for index,seed in enumerate(seeds):

    print('Running ',"MLP ", seed)
    tracemalloc.start()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    classifier = MLPClassifier(hidden_layer_sizes=(50, 25), activation='logistic', solver='adam', max_iter=500, random_state=seed, alpha=0.001, learning_rate_init=0.001) # ALPHA WAS 0.0001
    kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

    scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

    memory_usage = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory = peak / 1024 / 1024
    print(f"Current memory usage:{current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak_memory:.1f} MB")
    print("Scores:")
    for key,value in scores.items():
        print(key,":",value, "Mean:", round(np.mean(value), 2))
    print()
    
    classifier_statistics[str(classifier).split('(')[0]]['f1_binary'].append(np.mean(scores['test_f1_score_binary']))
    classifier_statistics[str(classifier).split('(')[0]]['f1_macro'].append(np.mean(scores['test_f1_macro']))
    classifier_statistics[str(classifier).split('(')[0]]['accuracy'].append(np.mean(scores['test_accuracy']))
    classifier_statistics[str(classifier).split('(')[0]]['peak_memory_used'].append(peak_memory)

    tracemalloc.reset_peak()

for index,seed in enumerate(seeds):

    print('Running ',"Random Forest ", seed)
    tracemalloc.start()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    classifier = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=100, class_weight={'NB': 1.0, 'TB': 30},  random_state=seed) # max depth 20
    kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

    scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

    memory_usage = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory = peak / 1024 / 1024
    print(f"Current memory usage:{current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak_memory:.1f} MB")
    print("Scores:")
    for key,value in scores.items():
        print(key,":",value, "Mean:", round(np.mean(value), 2))
    print()
    
    classifier_statistics[str(classifier).split('(')[0]]['f1_binary'].append(np.mean(scores['test_f1_score_binary']))
    classifier_statistics[str(classifier).split('(')[0]]['f1_macro'].append(np.mean(scores['test_f1_macro']))
    classifier_statistics[str(classifier).split('(')[0]]['accuracy'].append(np.mean(scores['test_accuracy']))
    classifier_statistics[str(classifier).split('(')[0]]['peak_memory_used'].append(peak_memory)

    tracemalloc.reset_peak()

for index,seed in enumerate(seeds):

    print('Running ',"Logistic Regression ", seed)
    tracemalloc.start()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    classifier = LogisticRegression(C=0.01, penalty='l1',solver='saga',random_state=seed, max_iter=500, class_weight={'NB': 1, 'TB': 10}) #consider running it again with more iterations cause it did not converge, was l1, C=0.01
    kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

    scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

    memory_usage = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory = peak / 1024 / 1024
    print(f"Current memory usage:{current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak_memory:.1f} MB")
    print("Scores:")
    for key,value in scores.items():
        print(key,":",value, "Mean:", round(np.mean(value), 2))
    print()
    
    classifier_statistics[str(classifier).split('(')[0]]['f1_binary'].append(np.mean(scores['test_f1_score_binary']))
    classifier_statistics[str(classifier).split('(')[0]]['f1_macro'].append(np.mean(scores['test_f1_macro']))
    classifier_statistics[str(classifier).split('(')[0]]['accuracy'].append(np.mean(scores['test_accuracy']))
    classifier_statistics[str(classifier).split('(')[0]]['peak_memory_used'].append(peak_memory)

    tracemalloc.reset_peak()

for index,seed in enumerate(seeds):

    print('Running ',"Gradient Boosting ", seed)
    tracemalloc.start()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    classifier = GradientBoostingClassifier(learning_rate=0.05, max_depth=3, n_estimators=200, subsample= 1.0, random_state=seed)# 100, 5
    kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

    scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

    memory_usage = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory = peak / 1024 / 1024
    print(f"Current memory usage:{current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak_memory:.1f} MB")
    print("Scores:")
    for key,value in scores.items():
        print(key,":",value, "Mean:", round(np.mean(value), 2))
    print()
    
    classifier_statistics[str(classifier).split('(')[0]]['f1_binary'].append(np.mean(scores['test_f1_score_binary']))
    classifier_statistics[str(classifier).split('(')[0]]['f1_macro'].append(np.mean(scores['test_f1_macro']))
    classifier_statistics[str(classifier).split('(')[0]]['accuracy'].append(np.mean(scores['test_accuracy']))
    classifier_statistics[str(classifier).split('(')[0]]['peak_memory_used'].append(peak_memory)

    tracemalloc.reset_peak()

for index,seed in enumerate(seeds):

    print('Running ',"Decision Tree ", seed)
    tracemalloc.start()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    classifier = DecisionTreeClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=2, random_state=seed, class_weight={'NB': 1, 'TB': 30}) #max depth 20, min samples leaf 1
    kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

    scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

    memory_usage = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory = peak / 1024 / 1024
    print(f"Current memory usage:{current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak_memory:.1f} MB")
    print("Scores:")
    for key,value in scores.items():
        print(key,":",value, "Mean:", round(np.mean(value), 2))
    print()
    
    classifier_statistics[str(classifier).split('(')[0]]['f1_binary'].append(np.mean(scores['test_f1_score_binary']))
    classifier_statistics[str(classifier).split('(')[0]]['f1_macro'].append(np.mean(scores['test_f1_macro']))
    classifier_statistics[str(classifier).split('(')[0]]['accuracy'].append(np.mean(scores['test_accuracy']))
    classifier_statistics[str(classifier).split('(')[0]]['peak_memory_used'].append(peak_memory)

    tracemalloc.reset_peak()

for index,seed in enumerate(seeds):

    print('Running ',"SVC ", seed)
    tracemalloc.start()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 

    classifier = SVC(kernel='rbf',C=0.1,gamma=0.001,probability=True,class_weight={'NB': 1, 'TB': 10},random_state=seed) # gamma 0.001
    kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

    scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

    memory_usage = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory = peak / 1024 / 1024
    print(f"Current memory usage:{current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak_memory:.1f} MB")
    print("Scores:")
    for key,value in scores.items():
        print(key,":",value, "Mean:", round(np.mean(value), 2))
    print()
    
    classifier_statistics[str(classifier).split('(')[0]]['f1_binary'].append(np.mean(scores['test_f1_score_binary']))
    classifier_statistics[str(classifier).split('(')[0]]['f1_macro'].append(np.mean(scores['test_f1_macro']))
    classifier_statistics[str(classifier).split('(')[0]]['accuracy'].append(np.mean(scores['test_accuracy']))
    classifier_statistics[str(classifier).split('(')[0]]['peak_memory_used'].append(peak_memory)

    tracemalloc.reset_peak()
        
tracemalloc.clear_traces() 

for clf, scores in classifier_statistics.items():
   print()
   print(clf)
   for score in scores:
      print(score + ' average:', round(np.average(scores[score]),3))
      print(score + ' standard deviation:', round(np.std(scores[score]),5))