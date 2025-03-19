
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
            'e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff']

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
# Salva o modelo treinado para não precisar treiná-lo novamente posteriormente
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