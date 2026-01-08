import sys
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

tg_reference_name = sys.argv[1]

tg_reference = tgt.io.read_textgrid(tg_reference_name, predict_encoding(tg_reference_name), include_empty_intervals=False)
final_textgrid = tgt.core.TextGrid()


TB_tiers = [tier for tier in tg_reference.tiers if tier.name.startswith("TB-") and "ponto" not in tier.name]
print("Length of TB tiers: ", len(TB_tiers))

i = 0
positions = []
for tier in tg_reference:
    print("adicionando tier " + tier.name, i)
    final_textgrid.add_tier(tier)
    if tier.name.startswith("TB-") and "ponto" not in tier.name:
        positions.append(i+len(positions))
    i += 1
    
i = 0
for tier in TB_tiers:
    new_tier = tgt.core.IntervalTier(start_time=0, end_time=tier.end_time, name=tier.name+"-normal", objects=None)

    for interval in tier.intervals:
        next_text = interval.text
        print("Dirty text: " + next_text)
        next_text = clean_text(next_text).lower()
        print("Cleaned text: " + next_text)
        new_interval = tgt.core.Interval(start_time=interval.start_time,end_time=interval.end_time, text=next_text)
        new_tier.add_interval(new_interval)
    print("inserting new tier at position:", positions[i]+1)
    final_textgrid.insert_tier(new_tier, positions[i]+1)
    i += 1
    print(new_tier)
    print()

tgt.io.write_to_file(final_textgrid, tg_reference_name.split(".")[0]+"_normalized.TextGrid", format='long', encoding='utf-8')
print("Textgrid with normalized tiers of the speakers created successfully!")