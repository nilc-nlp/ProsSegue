import tgt
import chardet
import re
import os

clean_vocab ='ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû\-\'\n\ '

def predict_encoding(tg_path):
    '''Predict a file's encoding using chardet'''
    # Open the file as binary data
    with open(tg_path, 'rb') as f:
        # Join binary lines for specified number of lines
        rawdata = b''.join(f.readlines())

    return chardet.detect(rawdata)['encoding']

def clean_text(new_text):

    # Remove texto entre parênteses duplos
    new_text = re.sub("\(\([^)]*\)\)", "", new_text)

    # Remove texto entre parênteses duplos e "..." (caso o transcritor tenha esquecido de fechar os parênteses)
    new_text = re.sub("\(\([^(\.\.\.)]*\.\.\.", "", new_text)

    # Troca :: por espaço (pode causar erro quebrando palavras ou não. ex: "eh::assim" ou "u::ma pessoa")
    new_text = re.sub("::", "", new_text)

    # Troca / por espaço (pode causar erro quebrando palavras ou não. ex: ?
    new_text = re.sub("/", "", new_text)

    # Troca :: por espaço (pode causar erro quebrando palavras ou não. ex: "ja-mais" e "cr-u" ou "bumba-meu-boi" e "dá-nos")
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

    if len(new_text) > 0 and new_text[0] == ' ':
        new_text = new_text[1:]

    if len(new_text.split("\n")) > 0:
        new_text = os.linesep.join([s for s in new_text.splitlines() if s])
    return new_text
    
def textgridToTxt(reference_tg, output_txt_path):
    reference_tg = tgt.io.read_textgrid(reference_tg, predict_encoding(reference_tg), include_empty_intervals=True)

    tier_names = reference_tg.get_tier_names()
    print(tier_names)
    #tier_names_punct = [tier_name for tier_name in tier_names if "ponto" in tier_name]
    #tier_names_punct = [tier_name for tier_name in tier_names if "ponto" in tier_name] # VERSAO 3, JUNTANDO COM PONTUAÇÃO
    tier_names = [tier_name for tier_name in tier_names if tier_name.startswith("TB") and "ponto" not in tier_name] # VERSAO 3 E TB, JUNTANDO COM PONTUAÇÃO 
    #tier_names = [tier_name for tier_name in tier_names if "NTB" in tier_name] #or "ponto" in tier_name] # VERSAO 1 (original), VERSAO 2 (corrigida, mas sem pontuação)
    #print(tier_names_punct)
    #tier_names = tier_names + tier_names_punct
    print(tier_names)
    #print(tier_names_punct)
    locutores = ["<"+tier_name.split("-")[1]+">" for tier_name in tier_names]
    print(locutores)

    reference_tiers = {}
    index_intervals = {} # vetor de índices para percorrer cada camada
    aux_intervals_start_time = {}
    aux_intervals_text = {}
    curr_loc = ""

    # VERSAO 3, COM PONTUACAO (só esse bloco de 6 linhas)
    #aux_intervals_end_time = {}
    #reference_tiers_punct = {}
    #index_intervals_punct = {} # vetor de índices para percorrer cada camada
    #aux_intervals_start_time_punct = {}
    #aux_intervals_end_time_punct = {}
    #aux_intervals_text_punct = {}
    
    for tier_name in tier_names:
        print(tier_name)
        reference_tiers[tier_name] = reference_tg.get_tier_by_name(tier_name)
        index_intervals[tier_name] = 0 # vetor de índices para percorrer cada camada
        aux_intervals_start_time[tier_name] = reference_tiers[tier_name].intervals[0].start_time # cria vetor com o start time do intervalo
        aux_intervals_text[tier_name] = reference_tiers[tier_name].intervals[0].text
    #    aux_intervals_end_time[tier_name] = reference_tiers[tier_name].intervals[0].end_time # VERSAO 3
        
    # VERSAO 3, COM PONTUAÇÃO (só esse bloco)
    #for tier_name in tier_names_punct:
    #    print(tier_name)
    #    reference_tiers_punct[tier_name] = reference_tg.get_tier_by_name(tier_name)
    #    index_intervals_punct[tier_name] = 0 # vetor de índices para percorrer cada camada
    #    aux_intervals_start_time_punct[tier_name] = reference_tiers_punct[tier_name].intervals[0].start_time # cria vetor com o start time do intervalo
    #    aux_intervals_end_time_punct[tier_name] = reference_tiers_punct[tier_name].intervals[0].end_time
    #    aux_intervals_text_punct[tier_name] = reference_tiers_punct[tier_name].intervals[0].text
    
    with open(output_txt_path, 'w+') as f: 
        # PARA REMOVER CABEÇALHO BASTA COMENTAR AS DUAS LINHAS A SEGUIR
        cabecalho = "Locutores: " + " ".join(locutores) + "\n\n"
        f.write(cabecalho)

        while (len(index_intervals) > 0):  # todas as camadas não tiverem chegado ao fim continuamos       
            curr_tier_name = min(aux_intervals_start_time, key=lambda tier_name: aux_intervals_start_time[tier_name]) # busca o indice do menor tempo de início dentro do vetor
            curr_numbered_index = tier_names.index(curr_tier_name)
            print("indice:", curr_numbered_index, curr_tier_name, "vetor todo", aux_intervals_start_time) # é pra ser igual um dos names
            next_text = aux_intervals_text[curr_tier_name]
            print("texto:", next_text)
            
            #next_text = clean_text(next_text).lower() # Pré-processar fala
            # Caso o usuário queira desativar o pré-processamento mas não queira incluir falas compostas apenas por "..." basta adicionar a seguinte condição no if a seguir "and next_text != ".."""
            if(curr_loc != locutores[curr_numbered_index]):
                curr_loc = locutores[curr_numbered_index]
            
            if(next_text != ""):
                f.write(curr_loc)
                #f.write(locutores[curr_numbered_index])
                f.write(next_text) 
            
                # VERSAO 3
            #    curr_loc_punct_tier = curr_tier_name + "-ponto"
            #    while (reference_tiers_punct[curr_loc_punct_tier].intervals[index_intervals_punct[curr_loc_punct_tier]].start_time < aux_intervals_start_time[curr_tier_name]):
            #        index_intervals_punct[curr_loc_punct_tier] += 1
            #        print(reference_tiers_punct[curr_loc_punct_tier].intervals[index_intervals_punct[curr_loc_punct_tier]], reference_tiers_punct[curr_loc_punct_tier].intervals[index_intervals_punct[curr_loc_punct_tier]].end_time)
            #    if( aux_intervals_end_time[curr_tier_name] == reference_tiers_punct[curr_loc_punct_tier].intervals[index_intervals_punct[curr_loc_punct_tier]].end_time and reference_tiers_punct[curr_loc_punct_tier].intervals[index_intervals_punct[curr_loc_punct_tier]].text != ""):
            #        f.write(reference_tiers_punct[curr_loc_punct_tier].intervals[index_intervals_punct[curr_loc_punct_tier]].text)
            
                f.write("\n")
            
            # ATUALIZAR SÓ O ÍNDICE DA CAMADA QUE ACABAMOS DE USAR O TEXTO
            # se o intervalo atual for diferente do último intervalo da camada atual, atualizamos normal
            if (reference_tiers[curr_tier_name].intervals[index_intervals[curr_tier_name]] != reference_tiers[curr_tier_name].intervals[-1]):
                index_intervals[curr_tier_name] += 1
                aux_intervals_start_time[curr_tier_name] = reference_tiers[curr_tier_name].intervals[index_intervals[curr_tier_name]].start_time 
            #    aux_intervals_end_time[curr_tier_name] = reference_tiers[curr_tier_name].intervals[index_intervals[curr_tier_name]].end_time # VERSAO 3 
                aux_intervals_text[curr_tier_name] = reference_tiers[curr_tier_name].intervals[index_intervals[curr_tier_name]].text
                #print("atualizados:", aux_intervals_start_time)
                #print(aux_intervals_text) 
            else:
                del index_intervals[curr_tier_name]
                del aux_intervals_start_time[curr_tier_name]  #removing the index of the tier that ended
                del aux_intervals_text[curr_tier_name] 
            #    del aux_intervals_end_time[curr_tier_name] # VERSAO 3
                print("camada", curr_tier_name, "acabou")

# Organizando caminhos

# Inquérito selecionado
#inq = "SP_D2_012"

#inq = "SP_D2_062"
#inq = "SP_D2_255"
#inq = "SP_D2_333"
#inq = "SP_D2_343"
#inq = "SP_D2_360"
#inq = "SP_D2_396"
#inq = "SP_DID_018"
#inq = "SP_DID_137"
#inq = "SP_DID_161"
#inq = "SP_DID_208"
#inq = "SP_DID_234"
#inq = "SP_DID_235"
#inq = "SP_DID_242"
#inq = "SP_DID_250"
#inq = "SP_DID_251"
#inq = "SP_EF_124"
#inq = "SP_EF_153"
#inq = "SP_EF_156"
#inq = "SP_EF_377"
#inq = "SP_EF_388"
#inq = "SP_EF_405"


## CATNA

#inq = "SP_DID_001_parte_2"
#inq = "SP_D2_008_parte_2"                
#inq = "SP_DID_009_completo"
#inq = "SP_D2_010_parte_1"
#inq = "SP_D2_010_parte_3"
#inq = "SP_D2_023_parte_1"
#inq = "SP_DID_011_completo"
inq = "SP_D2_012"
#inq = "SP_DID_043_completo"
#inq = "SP_DID_044_parte"
#inq = "SP_DID_068_completo"
#inq = "SP_DID_089_completo"
#inq = "SP_DID_090_completo"
#inq = "SP_D2_055"
#inq = "SP_D2_095"
#inq = "SP_D2_109"
#inq = "SP_DID_002"
#inq = "SP_DID_013"
#inq = "SP_DID_017"
#inq = "SP_DID_030"
#inq = "SP_DID_053"
#inq = "SP_DID_114"
#inq = "SP_DID_121"
#inq = "SP_DID_070"
#inq = "SP_DID_016"
#inq = "teste"


reference_tg = "ReferenceTextgrids/" + inq + ".TextGrid" # verificar qual dessas opções consta no nome do arquivo e comentar a outra
#reference_tg = "ReferenceTextgrids/" + inq + ".textgrid"
#output_txt_path = "RevisedTxts/" + inq + "_revised.txt" # VERSAO 1, SEM CORREÇÃO
#output_txt_path = "RevisedTxts/" + inq + "_revised2.txt" # VERSAO 2, CORRIGIDA SEM PONTUAÇÃO
output_txt_path = "RevisedTxts/" + inq + "_revised_TB.txt" # VERSAO 2, CORRIGIDA SEM PONTUAÇÃO USANDO CAMADA TB
#output_txt_path = "RevisedTxts/" + inq + "_revised_punct.txt"
textgridToTxt(reference_tg, output_txt_path)