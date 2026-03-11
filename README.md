# ProsSegue: Segmentadores Prosódicos

This repository is the result of a Master's Project investigating prosodic segmentation for Brazilian Portuguese.

Inside folder "baseline approach", there is the adaptation of the method described in [Biron,2021] (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250969) for automatic  prosodic segmentation of spontaneous speech in Brazilian Portuguese (BP). It is a method based on heuristics that uses the duration of pauses and the difference in speech rate to determine the location of prosodic boundaries.

Inside folder "machine learning approach", lies an approach inspired by the work described in [Ananthakrishnan, 2008] (https://ieeexplore.ieee.org/abstract/document/4358088). It relies on nine acoustic features, extracted at a syllable-level, that use information about energy, fundamental frequency, duration of pauses and of nucleus vowels, to train a Random Forest classifier that determines whether there is or there is not a prosodic boundary after each syllable.  

## Acknowledgments

First of all, we would like to thank the annotators of the TaRSila project who were tireless in reviewing the automatic transcriptions, training and testing the models for various speech processing systems. This work was carried out at the Center for Artificial Intelligence (C4AI-USP), with support from the São Paulo Research Foundation (FAPESP grant nº 2019/07665-4) and from the IBM Corporation. We also thank the support of the Center of Excellence in Artificial Intelligence (CEIA) funded by the Goiás State Foundation (FAPEG grant no. 201910267000527), the São Paulo University Support Foundation (FUSP) and the National Council for Scientific and Technological Development (CNPq-PQ scholarship, process 304961/2021-3). This project was also supported by the Ministry of Science, Technology and Innovation, with resources from Law nº 8,248, of October 23, 1991, within the scope of PPI-SOFTEX, coordinated by Softex and published Residência no TIC 13, DOU 01245.010222/2022-44.

## Para citar este trabalho

### Baseline approach

> Giovana Meloni Craveiro, Vinicius Gonçalves Santos, Gabriel Jose Pellisser Dalalana, Flaviane R. Fernandes Svartman, and Sandra Maria Aluísio. 2024. Simple and Fast Automatic Prosodic Segmentation of Brazilian Portuguese Spontaneous Speech. In Proceedings of the 16th International Conference on Computational Processing of Portuguese - Vol. 1, pages 32–44, Santiago de Compostela, Galicia/Spain. Association for Computational Lingustics.

> https://aclanthology.org/2024.propor-1.4/

### Machine learning approach

> CRAVEIRO, Giovana M.; ALVES, Caroline A.; SVARTMAN, Flaviane; ALUÍSIO, Sandra M.. Machine Learning Classifiers with Acoustic Features for Prosodic Segmentation in Brazilian Portuguese: A Comprehensive Evaluation. In: SIMPÓSIO BRASILEIRO DE TECNOLOGIA DA INFORMAÇÃO E DA LINGUAGEM HUMANA (STIL), 16. , 2025, Fortaleza/CE. Anais [...]. Porto Alegre: Sociedade Brasileira de Computação, 2025 . p. 113-124. DOI: https://doi.org/10.5753/stil.2025.37818. 

> https://sol.sbc.org.br/index.php/stil/article/view/37818

<img width="547" height="502" alt="image" src="https://github.com/user-attachments/assets/f247816a-e49e-4190-9ed7-d1dffb70104f" />


## Libraries used

### Forced aligner:
- kaldi
  
### Prosodic segmentation:
- python 3.10.12
- chardet 5.2.0
- tgt 1.4.4
- ufpalign
- re
- os

# Instructions

For both approaches, it is necessary to start by using the forced phonetic aligner UFPAlign, which requires an audio and its transcription as input. As output, it provides a phonetically aligned transcription, which contains timestamps of the beginning and ending of each phone, syllable, and word. We will use that file as input for both approaches.

## Alinhador fonético forçado (Forced Aligner)

>Batista, C., Dias, A.L. & Neto, N. Free resources for forced phonetic alignment in Brazilian Portuguese based on Kaldi toolkit. EURASIP J. Adv. Signal Process. 2022, 11 (2022). https://doi.org/10.1186/s13634-022-00844-9
>

To use UFPAlign (exclusively in Linux), go to their github repository (https://github.com/falabrasil/ufpalign/) and follow the download instructions. In order to successfully configure it in my computer, I followed these steps:

0 - After installing kaldi, and successfully running kaldi's example 

1 - Clone UFPAlign github repository into "kaldi/egs" 

>WARNING: The following instructions may be obsolete after UFPAlign m2m-aligner update

2 - Download the file path.sh from this address: (https://github.com/falabrasil/kaldi-br/blob/master/fb-ufpalign/path.sh),

3 - Move it to kaldi/egs/ufpalign

4 - Modify the line that contains the path to folder "ufpalign" with the personalized path from your machine

5 - Go to the command line and run `source path.sh`.

6 - Then, the example command from UFPAlign's github will work with a single modification, like this: `bash ufpalign.sh demo/ex.wav demo/ex.txt mono`

7 - Great! Now you can create a folder inside folder 'ufpalign', which contains your .wav audio file (mono and with 16 kHz), its transcription .txt file, and run a command like `bash ufpalign.sh yourFolder/yourFile.wav yourFolder/yourFile.txt mono`

Every time you open a new terminal window, you should again run `source path.sh` before running your actual UFPAlign command.

In case UFPAlign fails to process your files, there are a few things you can try:

1- Guarantee your audio file is 16 kHz, and mono

2- Guarantee your transcription file does not contain double spaces, or any punctuation - UFPAlign will indicate words chars with errors at the command line

3- Use parameter `--no-bypass true`

4- Use parameters `--beam 40 --retry-beam 100` (and gradually increase them, or try different values)

5- Maybe if the audio contains too much noise or is too long, UFPAlign may still have problems with it, so maybe try cutting it or enhancing the audio

## Instructions - Baseline Approach

The following image illustrates the pipeline adopted in this work:

![pipeline](https://github.com/nilc-nlp/ProsSegue/assets/29277600/a378f1f3-cbdf-4c2a-83e3-c644dad1ad05)

To use the prosodic segmentation code on a certain audio segment, it is necessary to feed it with: 

1 - a .txt file containing the transcription of the audio, in which every utterance is separated by speaker

2 - a .TextGrid file containing the timestamps (beginning and ending) of each phone in a layer called "fonemeas" and of each word in a layer called "palavras-grafemas", as generated by UFPAlign (https://github.com/falabrasil/ufpalign/). To obtain it, we suggest using UFPAlign.

### Segmentador prosódico (Prosodic Segmentation)

To run the prosodic segmentator at your local machine with your personal data, you need to

0- Download file "segmentador_biron.py" 

1- In the same folder that you place it, create a folder called "Data", inside it create a folder for each inquiry. Name the folders with the name of the inquiries + "_segmentado".
Each inquiry folder must contain a folder for each part, which contains the .TextGrid with timestamps and .txt diarized transcription file for that part.

2- Inside the code you must personalize line `inq = "SP_DID_242"`, with the name of your inquiry, then personalize line `segments_quantity = 4` with the number of parts that you have and you should be relatively good to go. There might be additional path errors once you run for different inquiries, but the filenames or paths that must be corrected will be indicated for you at the terminal.

3- Then run:
`python3 segmentador_byron.py`

### Data - Baseline Approach

There are 5 inquiries available (SP_D2_012,SP_D2_255,SP_D2_360,SP_DID_242,SP_EF_156). 

![image](https://github.com/nilc-nlp/ProsSegue/assets/29277600/24f9b44d-8cd3-44dc-9c99-67fb0994e4cd)


Each inquiry was divided into segments of around 10 minutes and thus, its folder contain a folder for each of its segments. The names of the files always start with the name of the inquiry and indicate the number of the segment if applicable.

![image](https://github.com/nilc-nlp/ProsSegue/assets/29277600/323b352e-4d7a-4c87-8354-ae3c34cc0fd7)


Each segment folder contains:

![image](https://github.com/nilc-nlp/ProsSegue/assets/29277600/741ff08b-7223-4b5a-b693-fc98589d5c0b)

- the audio file *(e.g. SP_D2_012_1.wav)*
- the transcription file *(e.g. SP_D2_012_1_clipped.txt)*
- the .TextGrid file generated by UFPAlign *(e.g. SP_D2_012_clipped_1.TextGrid)*
- a readMe file containing the command used to generate the .TextGird file with UFPAlign *(it contains specific parameters)*
- the .txt file that contains each utterance divided by speaker. Its name ends with *locutores*. *(e.g. SP_D2_012_clipped_1_locutores.txt)*
- (possibly) a .txt file in which there is a line for every word and its respective speaker. This one is an auxiliary file that the prosodic segmentation code generates to help its process and is identified by its ending *locutores_palavras*. *(e.g. SP_D2_012_clipped_1_locutores_palavras.txt)*
- (possibly) the output .TextGrid file in which the utterances are prosodically segmented. Its name ends with *OUTPUT*. *(e.g. SP_D2_012_clipped_1_OUTPUT.TextGrid)*
  

Outside of the segment folders, there are also files that reference all of the segments. There are:

![image](https://github.com/nilc-nlp/ProsSegue/assets/29277600/f5be2c1d-513e-4667-af82-7ed98bac145c)


- the manually segmented .TextGrid file used as reference *(e.g. SP_D2_255.TextGrid)*
- a .txt file containing the full transcription of the inquiry *(e.g. SP_D2_012.txt)*
- a .TextGrid file in which all of the partial .TextGrid files from the segments were united into a single file *(e.g. SP_D2_012_concatenated.TextGrid)*
- a .txt file containing the utterances from all the segments by speaker *(e.g. SP_D2_012_locutores.txt)*
- a .txt file containing the utterances from all the segments by speaker, each word in a new line *(e.g. SP_D2_012_locutores_palavras.txt)*
- the output .TextGrid file in which the utterances are prosodically segmented. It corresponds to the whole inquiry and its name ends with "OUTPUT". *(e.g. SP_D2_255_OUTPUT.TextGrid)*
  
- (if applicable) the output prosodically segmented file obtained using only the first heuristic *(e.g. SP_D2_255_OUTPUT_ONLY_H1.TextGrid)*
- (if applicable) the output prosodically segmented file obtained using only the first and the second heuristic *(e.g. SP_D2_255_OUTPUT_ONLY_H1_H2.TextGrid)*
- (if applicable) the output prosodically segmented file obtained using only the silences' heuristic *(e.g. SP_D2_255_OUTPUT_ONLY_SIL.TextGrid)*
- (if applicable) a .csv file containing metrics obtained using all of the parameters *(e.g. SP_D2_255_metrics.csv)*
- (if applicable) a .csv file containing metrics obtained using only the first heuristic *(e.g. SP_D2_255_metrics_ONLY_H1.csv)*
- (if applicable) a .csv file containing metrics obtained using only the first and the second heuristic *(e.g. SP_D2_255_metrics_ONLY_H1_H2.csv)*
- (if applicable) a .csv file containing metrics obtained using only the silences' heuristic *(e.g. SP_D2_255_metrics_ONLY_SIL.csv)*

Note: The files indicated with "v2" in the inquiry SP_DID_242 indicated the files that were generated and obtained after a manual revision of the transcription of its audio.

Note: Some fields are marked with "(possibly)" because in some cases the segments were processed individually, and then concatenated before processed, and in some cases they were only processed after concatenation.

Note: Some fields are marked with "(if applicable)" because the files only exist for the inquiries that contain a reference TextGrid

Note: In SP_D2_012, the indication of speakers was confusing (doc., doc.f, doc.m, inf., inf.f, inf.m) and when standardizing names of speakers we were not sure how many speakers there actually were, so we experimented uniting the utterances of some speakers that seemed to be the same person (files indicated by "3loc","4loc","6loc"). The expert who manually segmented the inquiry chose the version with 3 speakers ("3loc") to use at the article.

## Instructions - Machine Learning Approach

The complete pipeline of the machine learning approach, which includes training, is as follows:

<img width="1355" height="357" alt="image" src="https://github.com/user-attachments/assets/fde91761-ed57-44c6-a668-7f28e50501e3" />


The pipeline of usage of the trained model for the machine learning approach is as follows:

<img width="1036" height="290" alt="image" src="https://github.com/user-attachments/assets/e41eaa1a-b8df-40bd-8e01-5015b7f18984" />

To use the prosodic segmentation code on a certain audio segment, it is necessary to go through 3 stages. The first one is the forced phonetic alignment UFPAlign described above. The second stage is the extraction of features, to which you will need the audio and the textgrid provided by UFPAlign. For the third stage, you will need the textgrid provided by UFPAlign and the csv file that you just obtained at stage 2, which contains all the extracted acoustic features of each syllable of your audio.

### Steps to perform the prosodic segmentation 

0 - Clone this repository
0.5 - Have your audios and transcriptions ready, then for one at a time, go through the following steps:

#### First stage

1 - Run UFPAlign with an audio and its transcription as input (more details above) to obtain a textgrid with timestamps of phones, syllables and words.

#### Second Stage

3 - Once you obtain a textgrid from UFPAlign, place it (and your audio) inside folder "machine learning approach" from this repository

4 - Run the following command, adapting the example to the name of your files: 

```python3 extracting_prosodic_features.py example.wav example.TextGrid```

#### Third stage

5 - Run the following command, adapting the example to the name of your files (the csv file you just obtained with the extracted features and the textgrid from UFPAlign): 

```python3 generate_textgrid_with_prosodic_segmented_utterances.py exemplo.csv exemplo.TextGrid ``` 

### Alternative versions (different models)

The current version of the code uses 8 features and the model was trained using MuPe-Diversidades' files processed in the m2m aligner version of UFPAlign ( a recent update), so it is recommended.

#### STIL 2025 

This version refers to the version of the approach described in the paper published in STIL 2025, where feature fo_avg_utt_diff relied on previous annotation of prosodic boundaries, and therefore, a model with 8 features is made available for usage. Although the model and scaler used to obtain the results reported in the paper are in the repository (RF_only-trainset_mupe-diversidades_9features_originalf0_avg_utt_diff.pkl;scaler_prosodic_only-trainset_mupe-diversidades_9features_originalf0_avg_utt_diff.pkl), you'll only be able to use the model intitled "RF_all_mupe-diversidades_8features.pkl" and scaler "scaler_all_mupe-diversidades_8features.pkl". 

In step 4, run the following command instead of the one mentioned before:

```python3 extracting_prosodic_features_articleversion.py example.wav example.TextGrid```

Then, inside code generate_textgrid_with_prosodic_segmented_utterances.py find comment "# 8 FEATURES OLD UFPALIGN" and uncomment the block of code below it, and comment all other alternatives near it. 

If you wish to replicate results from the article, you also need to find the appropriate blocks of code to comment/uncomment inside the code "generate_textgrid_with_prosodic_segmented_utterances.py".

#### 9 features model with f0_avg_utt_diff_2

We tested a new version of feature f0_avg_utt_diff: f0_avg_utt_diff_2. This version contains 9 features, including f0_avg_utt_diff_2, which instead of relying on previous annotation of prosodic boundaries, uses silences indicated by UFPAlign as "prosodic boundaries' references". So it measures the average F0 among all the text contained between silences, which is considered an utterance. The result is a feature that measures the difference between the average F0 of the syllable and the average F0 of said "utterance". This version of the code was developed after UFPAlign's m2m-aligner update, however, tests indicated that the version with 8 features performed better.

### Steps to train a model with your own dataset

Soon to be completed
