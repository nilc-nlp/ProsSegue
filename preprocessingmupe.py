import re
import os

clean_vocab ='ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû\-\'\n\ '

def clean_text(new_text):

    # arruma nome do locutor
    #new_text = re.sub(r'SPEAKER (\d+):\s*', r'SPEAKER \1;', new_text)

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

    # Formata conforme o vocabulário limpo (mas remove ; dos locutores)
    #new_text = re.sub("[^{}]".format(clean_vocab), "", new_text)

    new_text = re.sub("(?<![A-Z])\.", "", new_text)
    new_text = re.sub("\n[ ]+", "\n", new_text)
    new_text = re.sub("\n{3, 6}", "\n\n", new_text)
    new_text = re.sub("[ ]+", " ", new_text)

    # Substitui ehhhhhh por eh e afins
    new_text = re.sub("h+", "h", new_text)

    new_text = re.sub("'", "", new_text)

    new_text = re.sub(' +', ' ', new_text)
    new_text = new_text.replace("\n ", "\n")

    # Remove tudo dentro de colchetes
    new_text = re.sub(r'\[[^\]]*\]', '', new_text)

    if len(new_text) > 0 and new_text[0] == ' ':
        new_text = new_text[1:]

    # remove pulos de linha duplos
    if len(new_text.split("\n")) > 0:
        new_text = os.linesep.join([s for s in new_text.splitlines() if s])

    # Remove space at the beginning of lines
    new_text = os.linesep.join([line.lstrip() for line in new_text.splitlines()])

    return new_text

def generate_locs_file(new_text):
    # arruma nome do locutor
    new_text = re.sub(r'SPEAKER (\d+):\s*', r'SPEAKER \1;', new_text)

    # remove tudo dentro de parenteses
    new_text = re.sub(r'\([^)]*\)', '', new_text)

    # remove pulos de linha separando falas de um mesmo falante
    i = 0
    lines = new_text.split('\n')
    cleaned_lines = []
    while i < len(lines):
        if re.search(r'SPEAKER \d+;', lines[i]):
            cleaned_lines.append(lines[i])
        else:
            cleaned_lines[-1] += ' ' + lines[i]
        i += 1
    new_text = '\n'.join(cleaned_lines)

    # transforma em minúsculas 
    new_text = new_text.lower()

    # remove todas as pontuações
    new_text = re.sub(r'[^\w\s;]', '', new_text)

    # Remove múltiplos espaços
    new_text = re.sub("[ ]+", " ", new_text)

    return new_text


# Gera arquivo de transcrição contínua a partir do arquivo de locutores
def remover_locutores(new_text):

    # remove indicação de locutores
    new_text = re.sub(r'speaker (\d+);', ' ', new_text)

    # remove pulos de linha
    new_text = re.sub(r'\n', ' ', new_text)

    # Remove múltiplos espaços
    new_text = re.sub("[ ]+", " ", new_text)

    # Remove character representing the beginning of the file (if present) or space at the beginning of the file
    if new_text and (new_text[0] == ' ' or new_text[0] == '\ufeff'):
        new_text = new_text[1:]


    return new_text

#inq = "PA_Masc"
#inq = "PA_Fem"

estado = "PA"

#genero = "_Masc"
genero = "_Fem"

corpus = "MUPE/"

inq = estado + genero

clean_transcription_path = corpus + inq + "_transcricao_limpa.txt"
locs_path = corpus + inq + "_locutores.txt"
transcricao_continua_path = corpus + inq + "_transcricao_continua.txt"

with open(corpus+inq+'_corrigido.txt', 'r') as file:
    text = file.read()

corrected_text = clean_text(text)

with open(clean_transcription_path, 'w') as file:
    file.write(corrected_text)

corrected_text = generate_locs_file(corrected_text)

with open(locs_path, 'w') as file:
    file.write(corrected_text)

corrected_text = remover_locutores(corrected_text)

with open(transcricao_continua_path, 'w') as file:
    file.write(corrected_text)
