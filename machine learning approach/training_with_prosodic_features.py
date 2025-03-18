
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

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tgt
from pydub import AudioSegment

# IMPORTS SUGGESTED FOR MODELS BY CHATGPT
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# modelos usados pelo Bruno
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



# Aqui leio um csv correspondente às features prosódicas de cada janela de 10ms do áudio analisado
df_prosodic = pd.read_csv('ExtractedProsodicFeatures/CE1_prosodic_features.csv')#.sort_values(by='sound_filepath') # desnecessário no caso do córpus MuPe local, que já está em ordem alfabética

features = ['f0_avgutt_diff','p_dur','n_dur','e_range','e_maxavg_diff',
            'e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff'] # 'n_phones' poderia ser incluído mas o paper não utiliza essa feature pra segmentação prosódica aparentemente
#features = ['local_jitter', 'local_shimmer', 'min_intensity',
#      'relative_min_intensity_time', 'max_intensity',
#       'relative_max_intensity_time', 'mean_intensity', 'stddev_intensity',
#       'q1_intensity', 'median_intensity', 'q3_intensity', 'voiced_fraction',
#       'min_pitch', 'relative_min_pitch_time', 'max_pitch',
#       'relative_max_pitch_time', 'mean_pitch', 'stddev_pitch', 'q1_pitch',
#       'q3_pitch', 'mean_absolute_pitch_slope',
#       'pitch_slope_without_octave_jumps', 'min_hnr', 'relative_min_hnr_time',
#       'max_hnr', 'relative_max_hnr_time', 'mean_hnr', 'stddev_hnr', 'min_gne',
#       'max_gne', 'mean_gne', 'stddev_gne', 'sum_gne', 'band_energy',
#       'band_density', 'band_energy_difference', 'band_density_difference',
#       'center_of_gravity_spectrum', 'stddev_spectrum', 'skewness_spectrum',
#       'kurtosis_spectrum', 'central_moment_spectrum', 'f1_mean', 'f2_mean',
#       'f3_mean', 'f4_mean', 'f1_median', 'f2_median', 'f3_median',
#       'f4_median', 'formant_dispersion', 'average_formant', 'mff',
#       'fitch_vtl', 'delta_f', 'vtl_delta_f']

X = df_prosodic[features]
#df_prosodic['id'] = df_prosodic['label'].str[-3:] # extraindo só o id da coluna label pq está com o caminho todo do arquivo
y = df_prosodic.label.to_list() # Aqui a coluna label precisa estar preenchida com info do tipo TB ou - para cada janela de 10ms do áudio - feito!

# Replace NaN values with 0 in X
X = df_prosodic[features].fillna(0)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#sc = StandardScaler() 
#X = sc.fit_transform(X) # isso aqui faz uma transformação de média 0 e variância 1 que não sei se devia ser aplicada aqui a tudo... concluímos que não precisa
#print(X)

# Configuração de diferentes modelos para ver qual se sai melhor 

#classifiers = [MLPClassifier(activation='logistic',random_state=1,max_iter=3000), # este código era o código base do prof, apagar dps pq já adaptei embaixo
#               MLPClassifier(activation='tanh',random_state=1,max_iter=3000),
#               MLPClassifier(activation='relu',random_state=1,max_iter=3000),
#               DummyClassifier(strategy="most_frequent",random_state=1),
#               DummyClassifier(strategy="stratified",random_state=1),
#               DummyClassifier(strategy="uniform",random_state=1)]


# 3 classificadores descritos no paper, além desses há mais um intitulado NN +delex LM
classifiers = [
    LinearDiscriminantAnalysis(), 
    #GaussianMixture(n_components=18, covariance_type='full', random_state=42),  # GMM classifier - more complex, requires some more code to be investigated cause it is not a classifier
    MLPClassifier(hidden_layer_sizes=(25,), activation='logistic', solver='adam', max_iter=200),  # NN classifier as described in the paper, other characteristics are set up by default
    MLPClassifier(),
    RandomForestClassifier(n_estimators=100, random_state=42),
    RandomForestClassifier(),
    LogisticRegression(),
    SVC(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier()

    # Aumentar grupo de classificadores - trazer os métodos explorados no trabalho do Bruno: árvore de decisão,gradient boosting, svm, logistic regression
]

# Testar com inquéritos diversos - juntar todo o conjunto do mupe diversidades para testar tudo aqui
for classifier in classifiers:
    print('Running ',classifier)
    kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    scores=cross_val_score(classifier, X, y, cv=kf, scoring='f1_macro') 
    print(scores,'f1_macro=',np.mean(scores))
    print('-------')



# Aqui preciso selecionar o modelo que teve o melhor f1-score e ajustar as próximas linhas de acordo
# Para treiná-los, só preciso usar o fit em cada um dos modelos

# Model Training
print("Training LDA")
chosen_model = LinearDiscriminantAnalysis()
chosen_model.fit(X_train,y_train)


print("LDA trained")
# Salva o modelos treinado para não precisar treiná-lo novamente posteriormente
with open('model.prosodic.pkl', 'wb') as fid_model:
    pickle.dump(chosen_model,fid_model)

y_pred = chosen_model.predict(X_test)  # predicting
#print(y_pred.tolist()) 
#y_prob = chosen_model.predict_proba(X_test) # prediction probabilities
#print(y_prob.tolist())

# printing comparison among predicted label and true label
for true_label, predicted_label in zip(y_test, y_pred):
    print(f"True: {true_label}, Predicted: {predicted_label}")
# Utilizar só se eu também utilizar o scaler
# Salva o objeto scaler que foi utilizado para padronizar os dados para garantir que a mesma transformação seja aplicada a novos dados durante a previsão.
#with open('scaler.prosodic.pkl', 'wb') as fid_scaler:
#    pickle.dump(sc,fid_scaler)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="TB")  # Specify the positive class
recall = recall_score(y_test, y_pred, pos_label="TB")
f1 = f1_score(y_test, y_pred, pos_label="TB")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred, target_names=["NB", "TB"]))


# Calculate SER
# Get confusion matrix: [[TN, FP], [FN, TP]]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=["NB", "TB"]).ravel()
ser = (fp + fn) / (tp + fn)  # Using your formula
print(f"Slot Error Rate (SER): {ser:.4f}")


#print("flag 2")

# EXTRA - APPROACH 4 FROM THE PAPER, AS ORIENTED BY CHATGPT:
# (ACHO MELHOR PRIMEIRO EU TREINAR OS MODELOS ACIMA, QUE TIVERAM MELHORES RESULTADOS NO PAPER E SÃO MENOS COMPLEXOS, E DEPOIS EU VOLTO AQUI)

#from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import train_test_split
#import numpy as np
#import nltk
#from nltk import ngrams
#from collections import Counter

# Example: Neural Network for Prosody Classification
#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Train Neural Network
#nn_classifier = MLPClassifier(hidden_layer_sizes=(25,), activation='logistic', max_iter=1000)
#nn_classifier.fit(X_train, y_train)

# Get Posterior Probabilities (probabilities of prosodic boundary events)
#posterior_probabilities = nn_classifier.predict_proba(X_test)

# Create Sausage Lattice (example)
#def create_lattice(probs, num_syllables):
#    lattice = {}
#    for i in range(num_syllables):
#        lattice[i] = {
#            'boundary': probs[i][1],  # Boundary probability
#            'non_boundary': probs[i][0]  # Non-boundary probability
#        }
#    return lattice

#lattice = create_lattice(posterior_probabilities, len(X_test))

# Build a 4-gram LM (you can use `nltk` or `KenLM` for this)
#def build_lm(sequence):
#    trigrams = ngrams(sequence, 4)
#    trigram_counts = Counter(trigrams)
#    return trigram_counts

# Example of how you'd combine your lattice with the LM (simplified)
#def rescore_lattice_with_lm(lattice, lm):
#    for syllable in lattice:
        # You would apply your LM scores to adjust the lattice arcs here
#        pass  # Implementation depends on LM details

# Decode the best prosodic boundary sequence
# This could be a Viterbi algorithm or other sequence decoding methods
