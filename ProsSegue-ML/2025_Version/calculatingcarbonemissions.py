# ML Testing different classifiers performance to choose most adequate ML approach to train model

# ML HYPERPARAMETERS TUNING

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
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

seed=42

X, y, all_stratifications_ids = join_inquiries_in_single_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=seed, shuffle=True, stratify=all_stratifications_ids) # stratify=y 


print("Y train set - TB total count:", y_train.count('TB'))
print("Y train set - NB total count:", y_train.count('NB'))

print("Y test set - TB total count:", y_test.count('TB'))
print("Y test set - NB total count:", y_test.count('NB'))

seeds = [42, 17, 79]


for index,seed in enumerate(seeds):

    with EmissionsTracker(project_name="LDA - seed "+str(seed)+"K-FOLD only train data") as tracker:

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 

        classifier = LinearDiscriminantAnalysis(solver='lsqr')
        kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

        scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

for index,seed in enumerate(seeds):

    with EmissionsTracker(project_name="MLP - seed "+str(seed)+"K-FOLD only train data") as tracker:

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 

        classifier = MLPClassifier(hidden_layer_sizes=(50, 25), activation='logistic', solver='adam', max_iter=500, random_state=seed, alpha=0.001, learning_rate_init=0.001)
        kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

        scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)


for index,seed in enumerate(seeds):

    with EmissionsTracker(project_name="Random Forest - seed "+str(seed)+"K-FOLD only train data") as tracker:

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 

        classifier = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=100, class_weight={'NB': 1.0, 'TB': 30},  random_state=seed)
        kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

        scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

for index,seed in enumerate(seeds):

    with EmissionsTracker(project_name="Logistic Regression - seed "+str(seed)+"K-FOLD only train data") as tracker:

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 

        classifier = LogisticRegression(C=0.01, penalty='l1',solver='saga',random_state=seed, max_iter=500, class_weight={'NB': 1, 'TB': 10})
        kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

        scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

for index,seed in enumerate(seeds):

    with EmissionsTracker(project_name="Gradient Boosting - seed "+str(seed)+"K-FOLD only train data") as tracker:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 

        classifier = GradientBoostingClassifier(learning_rate=0.05, max_depth=3, n_estimators=200, subsample= 1.0, random_state=seed)
        kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

        scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

for index,seed in enumerate(seeds):

    with EmissionsTracker(project_name="Decision Tree - seed "+str(seed)+"K-FOLD only train data") as tracker:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 

        classifier = DecisionTreeClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=2, random_state=seed, class_weight={'NB': 1, 'TB': 30})
        kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

        scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)

for index,seed in enumerate(seeds):

    with EmissionsTracker(project_name="SVC - seed "+str(seed)+"K-FOLD only train data") as tracker:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 

        classifier = SVC(kernel='rbf',C=0.1,gamma=0.001,probability=True,class_weight={'NB': 1, 'TB': 10},random_state=seed)
        kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        multiple_scoring = {'f1_score_binary': make_scorer(f1_score, average='binary', pos_label='TB'), 'accuracy': 'accuracy', 'f1_macro' : 'f1_macro', 'f1_micro': 'f1_micro', 'precision': make_scorer(precision_score, average='binary', pos_label='TB'), 'recall': make_scorer(recall_score, average='binary', pos_label='TB') }

        scores = cross_validate(classifier, X_train, y_train, cv=kf, scoring=multiple_scoring,return_train_score=True)