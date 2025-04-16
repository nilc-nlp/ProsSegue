import re
import os
import tgt
import chardet
import sys

clean_vocab ='ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû\-\'\n\ '

def clean_text(new_text):
    if len(new_text) > 0 and new_text[0] == ' ':
        new_text = new_text[1:]

    # Remove texto entre parênteses duplos
    new_text = re.sub("\(\([^)]*\)\)", "", new_text)

    # Remove texto entre parênteses duplos e "..." (caso o transcritor tenha esquecido de fechar os parênteses)
    new_text = re.sub("\(\([^(\.\.\.)]*\.\.\.", "", new_text)

    # Troca :: por espaço (pode causar erro quebrando palavras ou não. ex: "eh::assim" ou "u::ma pessoa")
    new_text = re.sub("::", " ", new_text)

    # Troca / por espaço
    new_text = re.sub("/", " ", new_text)
    
    # Troca - por espaço
    new_text = re.sub("-", "", new_text)

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

    if len(new_text.split("\n")) > 0:
        new_text = os.linesep.join([s for s in new_text.splitlines() if s])
    return new_text

def predict_encoding(tg_path):
    '''Predict a file's encoding using chardet'''
    # Open the file as binary data
    with open(tg_path, 'rb') as f:
        # Join binary lines for specified number of lines
        rawdata = b''.join(f.readlines())

    return chardet.detect(rawdata)['encoding']

def textgridToCleanTxt(reference_tg, output_txt_path):
    full_transcription = ""
    reference_tg = tgt.io.read_textgrid(reference_tg, predict_encoding(reference_tg), include_empty_intervals=True)

    tier_names = reference_tg.get_tier_names()
    print("Camadas presentes no textgrid:", tier_names)
    #tier_names_punct = [tier_name for tier_name in tier_names if "ponto" in tier_name] # VERSAO 3, JUNTANDO COM PONTUAÇÃO
    tier_names = [tier_name for tier_name in tier_names if tier_name.startswith("TB") and "ponto" not in tier_name and "normal" not in tier_name] # VERSAO 3 E TB, JUNTANDO COM PONTUAÇÃO 
    print("Camadas que incluirei na transcrição:", tier_names)
    locutores = ["<"+tier_name.split("-")[1]+">" for tier_name in tier_names]
    print("Locutores:", locutores)

    reference_tiers = {}
    index_intervals = {} # vetor de índices para percorrer cada camada
    aux_intervals_start_time = {}
    aux_intervals_text = {}
    curr_loc = ""
    
    for tier_name in tier_names:
        reference_tiers[tier_name] = reference_tg.get_tier_by_name(tier_name)
        index_intervals[tier_name] = 0 # vetor de índices para percorrer cada camada
        aux_intervals_start_time[tier_name] = reference_tiers[tier_name].intervals[0].start_time # cria vetor com o start time do intervalo
        aux_intervals_text[tier_name] = reference_tiers[tier_name].intervals[0].text
        
    while (len(index_intervals) > 0):  # todas as camadas não tiverem chegado ao fim continuamos       
        curr_tier_name = min(aux_intervals_start_time, key=lambda tier_name: aux_intervals_start_time[tier_name]) # busca o indice do menor tempo de início dentro do vetor
        curr_numbered_index = tier_names.index(curr_tier_name)
        next_text = aux_intervals_text[curr_tier_name]
        
        next_text = clean_text(next_text).lower() # Pré-processar fala
        # Caso o usuário queira desativar o pré-processamento mas não queira incluir falas compostas apenas por "..." basta adicionar a seguinte condição no if a seguir "and next_text != ".."""
        if(curr_loc != locutores[curr_numbered_index] and next_text != ".."):
            curr_loc = locutores[curr_numbered_index]
        
        if(next_text != ""):
            full_transcription += next_text + " "
        
        # ATUALIZAR SÓ O ÍNDICE DA CAMADA QUE ACABAMOS DE USAR O TEXTO
        # se o intervalo atual for diferente do último intervalo da camada atual, atualizamos normal
        if (reference_tiers[curr_tier_name].intervals[index_intervals[curr_tier_name]] != reference_tiers[curr_tier_name].intervals[-1]):
            index_intervals[curr_tier_name] += 1
            aux_intervals_start_time[curr_tier_name] = reference_tiers[curr_tier_name].intervals[index_intervals[curr_tier_name]].start_time 
            aux_intervals_text[curr_tier_name] = reference_tiers[curr_tier_name].intervals[index_intervals[curr_tier_name]].text
        else:
            del index_intervals[curr_tier_name]
            del aux_intervals_start_time[curr_tier_name]  #removing the index of the tier that ended
            del aux_intervals_text[curr_tier_name] 
            print("camada", curr_tier_name, "acabou")
        full_transcription = clean_text(full_transcription)
    with open(output_txt_path, 'w+') as f: 
        f.write(full_transcription)
    print("Finalizado com sucesso!")
    print("A transcrição limpa em txt estará na mesma pasta do textgrid usado como entrada")

if len(sys.argv) < 2:
    print("Missing textgrid filename, please write it like this when you run the code: python3 mycode.py mytextgrid.TextGrid")
    sys.exit(1)
  # Adapt here according to the path of your audio and textgrid generated by ufpalign
CM_textgrid = sys.argv[1]
final_txt_name = CM_textgrid.split('.')[0] + ".txt"

print("Nome do textgrid passado por você:", CM_textgrid)
print("Nome do arquivo txt que vou gerar com a transcrição", final_txt_name)

textgridToCleanTxt(CM_textgrid, final_txt_name)