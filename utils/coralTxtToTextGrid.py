import sys
import tgt 
import chardet 
import re
import os

clean_vocab ='ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû\-\'\n\ \/' # adicionei barra aqui para não remover as marcações de TBs

def predict_encoding(tg_path):
    '''Predict a file's encoding using chardet'''
    # Open the file as binary data
    with open(tg_path, 'rb') as f:
        # Join binary lines for specified number of lines
        rawdata = b''.join(f.readlines())

    return chardet.detect(rawdata)['encoding']

def clean_text(new_text):
    if len(new_text) > 0 and new_text[0] == ' ':
        new_text = new_text[1:]

    # Substitui caracter proibido "ũ" por "um"
    new_text = re.sub("ũ", "um", new_text)

    # Remove texto entre parênteses duplos
    new_text = re.sub("\(\([^)]*\)\)", "", new_text)

    # Remove texto entre parênteses duplos
    new_text = re.sub("\*[^)]*\:", "", new_text)

    # Remove texto entre parênteses duplos e "..." (caso o transcritor tenha esquecido de fechar os parênteses)
    new_text = re.sub("\(\([^(\.\.\.)]*\.\.\.", "", new_text)

    # Troca :: por espaço (pode causar erro quebrando palavras ou não. ex: "eh::assim" ou "u::ma pessoa")
    new_text = re.sub("::", " ", new_text)

    # Troca / por espaço
    #new_text = re.sub("/", "", new_text)
    
    # Troca - por espaço
    new_text = re.sub("-", " ", new_text)

    # Troca ` por '
    new_text = new_text.replace("`","'")

    # se não há texto, só pontuação, retornamos a string vazia ""
    if not re.search('[A-Za-z0-9áàâãéèêíóôõúçÁÀÂÃÉÈÍÓÔÕÚÇ]', new_text):
        return ""

    # Formata conforme o vocabulário limpo
    new_text = re.sub("[^{}]".format(clean_vocab), "", new_text)

    # Remove múltiplos espaços
    new_text = re.sub("[ ]+", " ", new_text)

    new_text = re.sub("(?<![A-Z])\.", "", new_text)
    new_text = re.sub("\n[ ]+", "\n", new_text)
    new_text = re.sub("\n{3, 6}", "\n\n", new_text)
    new_text = re.sub("[ ]+", " ", new_text)

    # Substitui ehhhhhh por eh e afins
    new_text = re.sub("h+", "h", new_text)

    new_text = re.sub("'", "", new_text)

    new_text = re.sub(' +', ' ', new_text)
    new_text = new_text.replace("\n ", "\n")

    new_text = new_text.lower()

    if len(new_text.split("\n")) > 0:
        new_text = os.linesep.join([s for s in new_text.splitlines() if s])
    return new_text

def align_files(tg_phones, reference_file_name):

    tg_phones = tgt.io.read_textgrid(tg_phones, predict_encoding(tg_phones), include_empty_intervals=True)
    graphemes_tier = tg_phones.get_tier_by_name("grafemas")

    final_textgrid = tgt.core.TextGrid()
    reference_file = open(reference_file_name, "r")
    content = reference_file.read()
    content = clean_text(content)
    print(content)
    words_list = content.split(" ")
    reference_file.close()
    print(content)
    print(words_list)

    # lê arquivo e cria lista de palavras cleaned e // e percorre a lista 
    words_index = 0
 
    # Creating TBs tier
    new_tier = tgt.core.IntervalTier(start_time=0, end_time=graphemes_tier.end_time, name="TBs-L1", objects=None)
    current_TB = ""
    current_TB_start_time = 0
    for interval in graphemes_tier.intervals:
        #print(interval)
        if interval.text == "<eps>" or words_index >= len(words_list):
            print("eps", interval.text)
            continue
        if words_list[words_index] == "/" or words_list[words_index] == "" or words_list[words_index] == "//":
            print("pulando caracter:", words_list[words_index])
            words_index += 1
            print(words_list[words_index])
        if interval.text == words_list[words_index] and words_index+1 < len(words_list) and words_list[words_index+1] == "//": # fim da TB
            print(interval.text,"==",words_list[words_index], "fim da TB")
            current_TB += " " + words_list[words_index]
            new_interval = tgt.core.Interval(start_time=current_TB_start_time,end_time=interval.end_time, text=current_TB) 
            print(new_interval)
            new_tier.add_interval(new_interval)
            current_TB_start_time = interval.end_time
            current_TB = ""
            words_index += 1
        elif interval.text == words_list[words_index]: 
            print(interval.text,"==",words_list[words_index])
            if current_TB == "":
                current_TB += words_list[words_index]
            else:
                current_TB += " " + words_list[words_index]
            words_index += 1

    final_textgrid.add_tier(new_tier)
    print(new_tier)

    print("##################################################################")

    # Creating NTBs tier - Obs.: separations indicated with "[/]" were ignored
    new_tier = tgt.core.IntervalTier(start_time=0, end_time=graphemes_tier.end_time, name="NTBs-L1", objects=None)
    words_index = 0
    current_TB = ""
    current_TB_start_time = 0
    for interval in graphemes_tier.intervals:
        print(interval)
        if interval.text == "<eps>" or words_index >= len(words_list):
            print("eps", interval.text)
            continue
        if words_list[words_index] == "/" or words_list[words_index] == "" or words_list[words_index] == "//":
            print("pulando caracter:", words_list[words_index])
            words_index += 1
            print(words_list[words_index])
        if interval.text == words_list[words_index] and words_index+1 < len(words_list) and (words_list[words_index+1] == "/" or words_list[words_index+1] == "//"): # fim da NTB
            print(interval.text,"==",words_list[words_index], "fim da NTB")
            current_TB += " " + words_list[words_index]
            new_interval = tgt.core.Interval(start_time=current_TB_start_time,end_time=interval.end_time, text=current_TB) 
            print(new_interval)
            new_tier.add_interval(new_interval)
            current_TB_start_time = interval.end_time
            current_TB = ""
            words_index += 1
        elif interval.text == words_list[words_index]: 
            print(interval.text,"==",words_list[words_index])
            if current_TB == "":
                current_TB += words_list[words_index]
            else:
                current_TB += " " + words_list[words_index]
            words_index += 1

    final_textgrid.add_tier(new_tier)
    print(new_tier)

    return final_textgrid



if len(sys.argv) < 3:
    print("Missing audio or textgrid filenames, please write them like this when you run the code: python3 coralTxtToTextGrid.py reference_txt my_textgrid_generated_by_ufpalign.TextGrid")
    sys.exit(1)

tg_reference = sys.argv[1]
tg_phones = sys.argv[2]
id = tg_reference.split(".")[0]

final_textgrid_name = id + "-reference.TextGrid"

print("Nome dos arquivos passados por você:", tg_reference, tg_phones)
print("Nome do arquivo txt que vou gerar com a transcrição", final_textgrid_name)

final_textgrid = align_files(tg_phones, tg_reference)

tgt.io.write_to_file(final_textgrid, final_textgrid_name, format='long', encoding='utf-8') 