
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

models = ['MLP_model.pkl', 'LDA_model.pkl', 'SVC_model.pkl']

features = ['f0_avgutt_diff','p_dur','n_dur','e_range','e_maxavg_diff',
            'e_avgmin_diff','f0_range','f0_maxavg_diff','f0_avgmin_diff']

scaler = pickle.load(open('scaler.prosodic.pkl', 'rb'))

for model in models:

    chosen_model = pickle.load(open(model, 'rb'))

    seed = 42

    df_prosodic = pd.read_csv('MuPe-Diversidades.csv')

    X = df_prosodic[features]        
    X = df_prosodic[features].fillna(0) # Replace NaN values with 0 in X
    y = df_prosodic['label'] 

    print(X)
    print(y)
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y)

    y_pred = chosen_model.predict(X_test)  # predicting

    #print(y_pred.tolist()) 
    #y_prob = chosen_model.predict_proba(X_test) # prediction probabilities
    #print(y_prob.tolist())

    # printing comparison among predicted label and true label
    #for true_label, predicted_label in zip(y_test, y_pred):
    #    print(f"True: {true_label}, Predicted: {predicted_label}")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="TB")  # Specify the positive class
    recall = recall_score(y_test, y_pred, pos_label="TB")
    f1_macro = f1_score(y_test, y_pred, pos_label="TB", average='macro')
    f1_micro = f1_score(y_test, y_pred, pos_label="TB", average='micro')
    f1 = f1_score(y_test, y_pred, pos_label="TB")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score macro: {f1_macro:.4f}")
    print(f"F1 Score micro: {f1_micro:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["NB", "TB"]))

    # Calculate SER
    # Get confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=["NB", "TB"]).ravel()
    ser = (fp + fn) / (tp + fn)  # Using your formula
    print(f"Slot Error Rate (SER): {ser:.4f}")
