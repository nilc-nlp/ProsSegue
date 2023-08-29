# Código do segmentador automático de Biron

# Importar bibliotecas
import re
import os
import chardet
import tgt
import textgrids#remover overlaps de textgrids
import nltk
import mytextgrid
from praatio import textgrid
from pympi.Praat import TextGrid as pympiTextGrid

clean_vocab ='ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû\-\'\n\ '

class AutomaticSegmentation:
    def __init__(self, path, locs_file):
        self.path = path
        self.locs_file = locs_file
        self.text_align = ""
        self.alignment_tg = ""

    def remove_overlaps(self,input_tg):

        with open(input_tg, 'r') as tg_file:
            tg = tg_file.read()

        #filtrando para buscar pelos xmax e xmin só na camada de fonemas
        tg_aux = tg[tg.find('intervals: size'):tg.find('item[2]:')] 
        # coletando todas as ocorrências de xmax e xmin
        xmax_indexes = [m.start() for m in re.finditer('xmax', tg_aux)]
        xmin_indexes = [m.start() for m in re.finditer('xmin', tg_aux)]

        #para cada tempo de fim, verifica se o tempo de início consecutivo é menor. Se for o caso, corrige o tempo de fim para que se iguale ao tempo de início
        for enum, xmax in enumerate(xmax_indexes[:-1]):
            xmax_line = tg_aux[xmax:xmax+15] # coleta aproximadamente a linha do xmax
            xmin = xmin_indexes[enum+1]
            xmin_line = tg_aux[xmin:xmin+15]
            xmin_value = xmin_line.split()
            xmax_value = xmax_line.split() # transforma a linha em uma lista separada por espaços
            xmin_value = xmin_value[2] 
            xmax_value = xmax_value[2] # busca o índice que corresponde ao valor do xmax

            try:
                xmin_value = float(xmin_value)
                xmax_value = float(xmax_value)
                
                if xmin_value < xmax_value:
                    print("OVERLAP")
                    tg = tg.replace("xmax = "+str(xmax_value), "xmax = "+str(xmin_value),1)

            except:
                continue
        with open(input_tg, 'w') as tg_file:
            tg_file.write(tg)


    # palavras_por_locutor
    def clean_text(self, new_text):
        if len(new_text) > 0 and new_text[0] == ' ':
            new_text = new_text[1:]

        #new_text = re.sub("ininteligível", "", new_text)
        #new_text = re.sub("inint", "", new_text)
        #new_text = re.sub("inint\.", "", new_text)

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

    # Gera dois novos arquivos com as falas e os turnos
    def generate_words_file(self, locs_file, locs_words_file):
        with open(locs_file, 'r') as lf:
            linhas = lf.readlines()
        locs_list = []
        for i, l in enumerate(linhas):
            if(l.find(';') != -1):
                loc = l.split(';')[0].lower()
                loc = re.sub('[-.,!;\s]', '', loc) # limpando locutor de caracteres especiais e espaços
                if loc not in locs_list:
                    #print("loc appended", loc)
                    locs_list.append(loc)

        with open(locs_file.replace("_locutores.txt", ".txt"), 'w', encoding='utf-8') as nlf2:
            with open(locs_words_file, 'w', encoding='utf-8') as nlf:
                for l in linhas:
                    if(l.find(';') != -1):
                        loc = l.split(";")[0]
                        loc = l.split(';')[0].lower()
                        loc = re.sub('[-.,!;\s]', '', loc) # limpando locutor de caracteres especiais e espaços
                        phrase_elements = l.split(" ")
                        # se houver um locutor no meio da frase, este loop identifica e remove a menção do locutor com o ; 
                        for index, element in enumerate(phrase_elements):
                            if(element.find(';') != -1): # se houver um ; na palavra, então é um erro de locutor concatenado à primeira palavra da próxima frase
                                phrase_elements[index] = element.split(";")[1] # este elemento é atualizado só com a palavra, sem o locutor
                                l = " ".join(phrase_elements) # e a frase é reconstruída sem o locutor
                        # padronizando locutores para doc1, doc2... e l1,l2,...
                        loc = re.sub(r'inf(\d+)', r'l\1', loc)
                        loc = re.sub(r'doc(\d+)', r'doc\1', loc)
                        loc = re.sub(r'inff', r'l1', loc)
                        loc = re.sub(r'infm', r'l2', loc)
                        loc = re.sub(r'docf', r'doc2', loc)
                        loc = re.sub('inf$', 'l1', loc)
                        loc = re.sub('doc$', 'doc1', loc)
                        for loc_from_list in locs_list:
                            if re.search(loc_from_list+"[\s.;,-].*", l,re.IGNORECASE):  #remove locutor duplicado no inicio do texto    
                                l = l[len(loc_from_list)+1:]
                    l = l.lower()
                    l = self.clean_text(l)
                    
                    for lp in l.split():
                        if lp.isnumeric():
                            continue
                        nlf.write(loc+';'+lp+"\n") # reescreve o .txt do texto
                        nlf2.write(lp+" ") # escreve o arquivo de palavras por locutor
        self.text_align = locs_file.replace(".txt", "_palavras_align.txt")

    # função para concatenar todos os textgrids da entrada gerados pelo ufpalign e escrever um textgrid único correspondente a todos os segmentos do inquérito
    def concatenate_textgrids(self, alignment_tg_list, final_tg_name):
        
        first_tg = tgt.io.read_textgrid(alignment_tg_list[0], self.predict_encoding(alignment_tg_list[0]))
        final_textgrid = tgt.core.TextGrid()
        initial_time = 0

        # passaremos por cada camada do textgrid final só uma vez, adicionando todos os intervalos de cada textgrid correspondente àquela camada com os tempos ajustados
        for tier in first_tg.tiers:
            new_tier = tgt.core.IntervalTier(start_time=tier.start_time + initial_time, end_time=tier.end_time + initial_time, name=tier.name, objects=None)
            # para o textgrid correspondente a cada segmento:
            for textgrid in alignment_tg_list:
                tg = tgt.io.read_textgrid(textgrid, self.predict_encoding(textgrid))
                original_tier = tg.get_tier_by_name(tier.name)
                # para cada intervalo da camada atual no textgrid atual, criamos um novo intervalo com os tempos ajustados e o adicionamos à camada que criamos no textgrid final
                for interval in original_tier.intervals:
                    new_interval = tgt.core.Interval(start_time=interval.start_time + initial_time,end_time=interval.end_time + initial_time, text=interval.text)
                    new_tier.add_interval(new_interval)
                # terminamos de percorrer o textgrid de um segmento, então adicionamos o tempo final do textgrid percorrido à variável que será somada com os tempos do próximo textgrid
                initial_time += tg.end_time
            # quando terminamos de percorrer todos os textgrids naquela camada, adicionamos a camada ao textgrid final
            final_textgrid.add_tier(new_tier)
            initial_time = 0

        # escrevendo o textgrid concatenado em um novo arquivo
        tgt.io.write_to_file(final_textgrid, final_tg_name, format='long', encoding='utf-8') 

    # função para concatenar os arquivos de locutores
    def concatenate_locs_file(self, locs_files_list, final_locs_file_path):

        concatenated_locs_text = ""
        for locs_file in locs_files_list:
            with open(locs_file, "r", encoding='utf-8-sig') as f:
                concatenated_locs_text += f.read() + "\n"

        # escrever conteúdo concatenado no arquivo final        
        with open(final_locs_file_path, "w") as final_locs_file:
            final_locs_file.write(concatenated_locs_text)

    def predict_encoding(self, tg_path):
        '''Predict a file's encoding using chardet'''
        # Open the file as binary data
        with open(tg_path, 'rb') as f:
            # Join binary lines for specified number of lines
            rawdata = b''.join(f.readlines())

        return chardet.detect(rawdata)['encoding']

    def calculate_average_phone_duration(self, window_phones):
        s = 0
        for wp in window_phones:
            s += wp[1]
        return s / len(window_phones)

    def dsr_threshold_1(self, windows, delta1):
        # primeiro encontramos o maior e menor speech rates no turno (speech rate = média dos fonemas da janela referente à palavra)
        max_speech_rate_diff = 0
        last_sr = 0
        max_speech_rate = windows[0][1] #primeiro speech rate do turno
        min_speech_rate = windows[0][1]
        for word in windows: # word é uma lista com os fonemas de cada palavra, word[1] é o speech rate (a média dos fonemas da janela)
            if word[1] > max_speech_rate:
              max_speech_rate = word[1]
            if word[1] < min_speech_rate:
              min_speech_rate = word[1]
            last_sr = word[1]

        max_speech_rate_diff = max_speech_rate - min_speech_rate
        last_sr = 0
        dsrs_1 = []
        dsr_windows_1 = []

        # se a diferença entre os speech rates das janelas consecutivas é > delta1 da maior diferença entre speech rates,
        #  identificamos como DSR.

        # tempo da última fronteira
        last_boundary = 0 

        for word in windows:
            if abs(word[1] - last_sr) > delta1 * max_speech_rate_diff:
                print("DSR 1!", word)
                boundary = tgt.core.Interval(start_time=last_boundary, end_time=word[0][0][2])
                print(boundary)
                last_boundary = word[0][0][2]
                dsrs_1.append(word[0][0][2])
                dsr_windows_1.append(word)
            last_sr = word[1]

        return dsrs_1, dsr_windows_1

    def dsr_threshold_2(self, dsr_windows_1, delta2, interval_size, windows, min_words_h2):
        # Giovana que fez
        filtered_windows = []
        previous_dsr = 0
        for dsr in dsr_windows_1:
            aux_windows = [word for word in windows if word[0][0][2] < dsr[0][0][2] and word[0][0][2] > previous_dsr]
            if len(aux_windows) > 0:
              stretch_duration = aux_windows[-1][0][-1][3] - aux_windows[0][0][0][2]
              previous_dsr = dsr[0][0][2]
              if len(aux_windows) > min_words_h2 and stretch_duration > interval_size:
                filtered_windows.append(aux_windows)


        # caso de borda do último dsr, que não é englobado pelo loop anterior
        dsr = windows[-1][0][-1][3]
        aux_windows = [word for word in windows if word[0][0][2] < dsr and word[0][0][2] > previous_dsr]
        if len(aux_windows) > 0:
          stretch_duration = aux_windows[-1][0][-1][3] - aux_windows[0][0][0][2]
          if len(aux_windows) > min_words_h2 and stretch_duration > interval_size:
            filtered_windows.append(aux_windows)

        dsrs_2 = []
        last_sr = 0
        last_boundary = 0

        for stretch in filtered_windows:
          max_speech_rate_diff = 0
          last_sr = 0
          max_speech_rate = stretch[0][1] #primeiro speech rate do turno
          min_speech_rate = stretch[0][1]
          for word in stretch: # word é uma lista com os fonemas de cada palavra, word[1] é o speech rate (a média dos fonemas da janela)
              if word[1] > max_speech_rate:
                max_speech_rate = word[1]
              if word[1] < min_speech_rate:
                min_speech_rate = word[1]
              last_sr = word[1]

          max_speech_rate_diff = max_speech_rate - min_speech_rate
          for word in stretch:
              if abs(word[1] - last_sr) > (delta2 * (max_speech_rate - min_speech_rate)):
                  print("DSR 2!", word)
                  boundary = tgt.core.Interval(start_time=last_boundary, end_time=word[0][0][2])
                  print(boundary)
                  last_boundary = word[0][0][2]
                  dsrs_2.append(word[0][0][2])
              last_sr = word[1]
        return dsrs_2

    def print_silences(self, sil_timestamps, silence_threshold, sil_boundaries):
        silences = []
        sil_avg = 0

        for s in sil_timestamps:
            if s[1] - s[0] > silence_threshold:
                boundary = tgt.core.Interval(start_time=s[0], end_time=s[1], text="...")
                sil_boundaries.append(boundary)
                print("DSR SIL!", boundary)
                sil_avg += abs(s[1]- s[0])
                silences = silences + [s[0],s[1]]

        sil_avg = sil_avg/len(sil_timestamps)
        print("Quantidade de fronteiras de silêncio:",len(sil_timestamps))
        print("Média da duração das fronteiras de silêncios", sil_avg, "\n")
        return silences

    def fill_boundaries_tier(self, timestamps, boundaries_tier, sil_boundaries):
        timestamps.sort()
        #print("timestamps:", timestamps)

        last_ts = timestamps[0]
        for ts in timestamps[1:]:
            if (tgt.core.Interval(start_time=last_ts, end_time=ts, text="...") in sil_boundaries):
                interval_text = "..."
            else:
                interval_text = ""
            boundary = tgt.core.Interval(start_time=last_ts, end_time=ts, text=interval_text)
            try:
              boundaries_tier.add_interval(boundary)
            except:
              print("overlap")
            last_ts = ts
            #print("boundary:", boundary)

        #print("Boundaries tier",boundaries_tier)

    def find_boundaries(self, locs_file, tg_file, annot_tg, output_tg_file, window_size, delta1, delta2, silence_threshold, interval_size, min_words_h2):        
        input_tg = tgt.io.read_textgrid(tg_file, self.predict_encoding(tg_file), include_empty_intervals=False)
        output_tg = tgt.core.TextGrid(output_tg_file)
        annot_tg = tgt.core.TextGrid(annot_tg)

        boundaries_tier = tgt.core.IntervalTier(start_time=input_tg.start_time, end_time=input_tg.end_time, name="fronteiras_metodo", objects=None)

        # lemos as palavras no arquivo com locutores e geramos essas duas listas:
            # sentences: guarda todas as palavras do texto
            # g2p_words: guarda as palavras convertidas para fonemas
        with open(locs_file, 'r') as lf:
            locs_and_words = lf.readlines()
            #print(locs_and_words)
            phonemesTier = input_tg.get_tier_by_name("fonemeas") 
            wordGraphemesTier = input_tg.get_tier_by_name("palavras-grafemas")
            sentences = ""
            for lw in locs_and_words:
                w = lw.split(';')[1]
                sentences += ' ' + w
            sentences = sentences.split()

            # criando g2p_words, a lista de palavras compostas por fonemas para comparação
            g2p_words = []
            auxGrapheme = ""
            index = 0
            phone = phonemesTier[index]
            still_mounting_grapheme = True
            
            for grapheme in wordGraphemesTier:
              still_mounting_grapheme = True
              while still_mounting_grapheme:
                if phone.text != "sil":
                  auxGrapheme += phone.text
                  # obs.: achei um erro em que o tempo final do grafema não correspondia ao tempo final do último fonema da palavra, então acrescentei essa condição de o tempo do fonema ser maior ou igual
                  if phone.end_time >= grapheme.end_time: # fim da palavra, pula para a próxima
                    g2p_words.append(auxGrapheme)
                    auxGrapheme = ""
                    still_mounting_grapheme = False
                index += 1
                
                if index < len(phonemesTier):
                  phone = phonemesTier[index]
                
            for pwi, pw in enumerate(g2p_words):
                # substituimos fonemas 'w' por 'v' (e 'y' por 'i') pois o alinhador joga fora (é uma solução tosca, mas o que dá pra fazer sem alterar o alinhador)
                g2p_words[pwi] = g2p_words[pwi].replace("w", "v")
                g2p_words[pwi] = g2p_words[pwi].replace("y", "i")
        #print("sentences", sentences)
        #print("palavras convertidas para fonemas", g2p_words, len(g2p_words))
        
        # SO FAR SO GOOD

        tier = input_tg.get_tier_by_name("fonemeas")
        # índice para iterar pelas palavras convertidas via g2p
        i = 0
        curr_turn = ""
        # Formato dos items da lista window_phones (texto[0], duração[1], tempo de início[2], tempo de fim[3], locutor[4])
        window_phones = []
        tier_names = []
        windows = []
        all_timestamps = [input_tg.start_time, input_tg.end_time]
        timestamps_dsrs_1 = []
        timestamps_dsrs_2 = []
        constr = ""
        curr_word = g2p_words[0]
        curr_word_grapheme = sentences[0]
        curr_loc = locs_and_words[0].split(';')[0]
        curr_window = [tier.intervals[0].start_time, tier.intervals[0].start_time + window_size]
        turn_index = 0
        last_turn_start = 0
        
        # estrutura que guarda, em ordem, [0] cada palavra do texto, [1] o texto do turno atual até dada palavra, [2] o locutor desse turno,
        #  [3] o início do tempo da palavra, [4] o início do tempo do turno e [5] o final do tempo do turno até a palavra (final do tempo da palavra)
        turn_until_word = []
        for index in range(len(sentences)):
            turn_until_word.append(["", "", "", 0.0, 0.0, 0.0])
        turn_until_word[0] = [curr_word_grapheme, curr_word_grapheme, curr_loc, 0.0, 0.0, 0.0]
        #print(turn_until_word)

        # Se o primeiro fonema é um silêncio, o início da palavra, da janela e o início do tempo do turno é na verdade o tempo de fim deste silêncio
        if(tier.intervals[0].text == "sil"):
            turn_until_word[0][3] = tier.intervals[0].end_time
            turn_until_word[0][4] = tier.intervals[0].end_time
            last_turn_start = tier.intervals[0].end_time
            curr_window = [tier.intervals[0].end_time, tier.intervals[0].end_time + window_size]

        sil_timestamps = [] 
        sil_avg_total = 0
        sil_count = 0

        # Começamos a percorrer a camada de fonemas do textgrid de input
        for enum, interval in enumerate(tier.intervals):
            # adicionamos à lista de trechos de silencio caso o fonema seja "sil"
            if interval.text == 'sil':
                sil_timestamps.append([interval.start_time, interval.end_time]) 
                sil_avg_total += abs(interval.end_time - interval.start_time)
                # atualiza início da próxima janela e da próxima palavra
                curr_window = [interval.end_time, interval.end_time + window_size]
                turn_until_word[i][3] = interval.end_time
            else:
                # substituimos fonemas 'w' por 'v' (e 'y' por 'i') pois o alinhador joga fora (é uma solução tosca, mas o que dá pra fazer sem alterar o alinhador)
                aux = interval.text
                aux = aux.replace("w", "v") 
                aux = aux.replace("y", "i")
                constr += aux

                # adicionamos à lista de fonemas da janela o fonema atual se sua duração caso esteja dentro do tempo da janela
                if interval.end_time < curr_window[1]: 
                    window_phones.append([interval.text, interval.end_time - interval.start_time, interval.start_time, interval.end_time, curr_loc])
                
                # se os fonemas encontrados desde a última janela formam uma palavra, finalizamos a janela atual com os fonemas que cabem nela e seguimos para a próxima janela pois cada palavra precisa da sua janela
                if constr == curr_word:
                    index = enum+1
                    j = 0 # contador de palavras que cabem na janela
                    aux_word = ""
                    # verificando e adicionando os próximos fonemas que ainda cabem na janela atual
                    while index < len(tier.intervals) and tier.intervals[index].end_time < curr_window[1]:
                        #print(curr_word, i, j, locs_and_words[i+j])
                        if tier.intervals[index].text == 'sil': 
                            index += 1
                            continue
                        elif locs_and_words[i+j+1].split(';')[0] != curr_loc:
                            break
                        else:
                            window_phones.append([tier.intervals[index].text, tier.intervals[index].end_time - tier.intervals[index].start_time, tier.intervals[index].start_time, tier.intervals[index].end_time, curr_loc])

                        aux_word += tier.intervals[index].text
                        index += 1
        
                        if aux_word == g2p_words[i+1]:
                            j += 1
                    curr_turn += ' ' + curr_word_grapheme
        
                    if window_phones:
                        av = self.calculate_average_phone_duration(window_phones)
                        windows.append([window_phones, av])

                    # atualiza o turno atual até a palavra e o tempo do final do turno até agora
                    turn_until_word[i][1] = curr_turn
                    turn_until_word[i][5] = interval.end_time

                    # se a palavra atual foi concluída, pulamos para a próxima
                    i += 1
                    try:
                        curr_word = g2p_words[i]
                        curr_word_grapheme = sentences[i]

                        # atualiza a próxima palavra do turno e seu tempo de início
                        turn_until_word[i][0] = curr_word_grapheme
                        turn_until_word[i][4] = last_turn_start # inicio do tempo do turno
                        turn_until_word[i][3] = interval.end_time # INICIO do tempo da palavra, caso a próxima seja um silêncio, este valor será atualizado quando o silêncio for identificado

                    except:
                        print("lista de g2p acabou")

                        # primeira heurística
                        dsrs_1, dsr_windows_1 = self.dsr_threshold_1(windows, delta1)
                        #print("dsrs1",dsrs_1)
                        # segunda heurística
                        dsrs_2 = self.dsr_threshold_2(dsr_windows_1, delta2, interval_size, windows, min_words_h2)
                        #print("dsrs2", dsrs_2)
                        # junta todas as fronteiras identificadas pelas duas primeiras heuristicas aplicadas no turno em uma lista
                        timestamps = list(set(dsrs_1 + dsrs_2))
                        print("tamanho dsrs1:", len(dsrs_1))
                        print("tamanho dsrs2:", len(dsrs_2))
                        print("tamanho timestamps:", len(timestamps), "\n")
                        all_timestamps += timestamps
                        timestamps_dsrs_1 += dsrs_1
                        timestamps_dsrs_2 += dsrs_2

                        if curr_loc not in tier_names:

                            loc_tb_tier = tgt.core.IntervalTier(start_time=input_tg.start_time, end_time=input_tg.end_time, name="TB-"+curr_loc, objects=None)
                            loc_ntb_tier = tgt.core.IntervalTier(start_time=input_tg.start_time, end_time=input_tg.end_time, name="NTB-"+curr_loc, objects=None)

                            # Adds the new tiers to the textgrid file
                            output_tg.add_tier(loc_tb_tier) 
                            output_tg.add_tier(loc_ntb_tier) 

                        # limpa lista de janelas do turno
                        windows = []
                        break

                    # se há troca de turno chamamos as heurísticas para as janelas do turno
                    if locs_and_words[i].split(';')[0] != curr_loc:
                        # primeira heurística
                        dsrs_1, dsr_windows_1 = self.dsr_threshold_1(windows, delta1)
                        print("dsrs1",dsrs_1)
                        # segunda heurística
                        dsrs_2 = self.dsr_threshold_2(dsr_windows_1, delta2, interval_size, windows, min_words_h2)
                        print("dsrs2",dsrs_2)

                        # junta todas as fronteiras identificadas pelas duas primeiras heuristicas aplicadas no turno em uma lista
                        timestamps = list(set(dsrs_1 + dsrs_2))
                        timestamps_dsrs_1 += dsrs_1
                        timestamps_dsrs_2 += dsrs_2
                        all_timestamps += timestamps
                        print("curr_loc", curr_loc)
                        print("tamanho dsrs1:", len(dsrs_1))
                        print("tamanho dsrs2:", len(dsrs_2))
                        print("tamanho timestamps:", len(timestamps), "\n")

                        # resgata ou cria camada do locutor atual no textgrid
                        if curr_loc in tier_names:
                            loc_tb_tier = output_tg.get_tier_by_name("TB-"+curr_loc) 
                            loc_ntb_tier = output_tg.get_tier_by_name("NTB-"+curr_loc) 
                        else:
                            # Creates TB and NTB tiers for the speaker
                            #print("new tiers: TB-"+curr_loc, "and", "NTB-"+curr_loc)
                            loc_tb_tier = tgt.core.IntervalTier(start_time=input_tg.start_time, end_time=input_tg.end_time, name="TB-"+curr_loc, objects=None)
                            loc_ntb_tier = tgt.core.IntervalTier(start_time=input_tg.start_time, end_time=input_tg.end_time, name="NTB-"+curr_loc, objects=None)

                            # Adds the new tiers to the textgrid file
                            output_tg.add_tier(loc_tb_tier) 
                            output_tg.add_tier(loc_ntb_tier) 
                            tier_names.append(curr_loc)

                        # limpa lista de janelas e texto do turno
                        windows = []
                        curr_turn = ""

                        # atualiza o começo do tempo do turno atual
                        aux = 0
                        if tier.intervals[enum+1].text == "sil":
                            aux = 1
                        last_turn_start = tier.intervals[enum+aux].end_time
                        turn_until_word[i][4] = last_turn_start

                    # atualiza locutor para a proxima palavra
                    curr_loc = locs_and_words[i].split(';')[0]
                    turn_until_word[i][2] = curr_loc 

                    # limpamos a string que guarda a palavra sendo construída pelos fonemas
                    constr = ""
                    # janelas de 300 ms
                    curr_window = [interval.end_time, interval.end_time + window_size]
                    # limpamos a lista de fonemas para a próxima janela
                    window_phones = []

        # terceira heurística
        sil_boundaries = []
        silences = self.print_silences(sil_timestamps, silence_threshold, sil_boundaries)

        # junta todas as fronteiras identificadas das primeiras heuristicas com a terceira
        all_timestamps = list(set(all_timestamps + silences))
        all_timestamps.sort()
        print(all_timestamps)
        print("Tamanho do all_timestamps:",len(all_timestamps), "\n")

        # preenche tier de boundaries juntando as 3 heurísticas
        self.fill_boundaries_tier(all_timestamps, boundaries_tier, sil_boundaries)

        last_c = 0
        last_text = ""
        last_loc = turn_until_word[0][2]
        last_b = turn_until_word[0][4]
        missing_text = ""

        new_intervals = []
        index_atual_lista_palavras = 0
        tamanho_lista_palavras = len(turn_until_word)
        current_stretch = "" # trecho atual

        # aqui vamos inserir as informações das fronteiras identificadas pelo método nas tiers correspondentes de cada turno no textgrid
        for boundary in boundaries_tier.intervals:
            if boundary.text == "...":
                new_intervals.append([boundary, last_loc])
                current_stretch = ""
            # itera pelas palavras
            else:
                for index_infos_palavra in range(index_atual_lista_palavras,tamanho_lista_palavras):
                    # se o fim da fronteira é menor ou igual ao fim da palavra
                    if boundary.end_time <= turn_until_word[index_infos_palavra][5]:
                        # se trocou de turno, troca o locutor
                        if turn_until_word[index_infos_palavra][2] != last_loc:
                            last_loc = turn_until_word[index_infos_palavra][2]
                        if boundary.end_time < turn_until_word[index_infos_palavra][5]: #fim da fronteira é menor que fim da palavra, logo deve ser igual ao tempo de início da palavra atual
                            missing_text = turn_until_word[index_infos_palavra][0] + " "
                        else:
                            current_stretch += turn_until_word[index_infos_palavra][0]
                            missing_text = ""
                        i_text = current_stretch
                        i = [tgt.core.Interval(start_time=boundary.start_time, end_time=boundary.end_time, text=i_text), turn_until_word[index_infos_palavra][2]] 
                        new_intervals.append(i)
                        current_stretch = missing_text
                        # para de iterar pelas palavras para essa fronteira pois já foi encontrada
                        index_atual_lista_palavras = index_infos_palavra + 1
                        break
                    current_stretch += turn_until_word[index_infos_palavra][0] + " "

        # adiciona intervalos nas duas camadas do turno adequado
        for ni in new_intervals:
            tb_turn_tier = output_tg.get_tier_by_name("TB-"+ni[1])
            ntb_turn_tier =  output_tg.get_tier_by_name("NTB-"+ni[1])
            try:
                tb_turn_tier.add_interval(ni[0])
                ntb_turn_tier.add_interval(ni[0])
            except:
                print("overlap")



        # adiciona tier para comentários dos anotadores
        comments1_tier = tgt.core.IntervalTier(start_time=input_tg.start_time, end_time=input_tg.end_time, name="comentarios-anotacao", objects=None)
        output_tg.add_tier(comments1_tier) 

        comments2_tier = tgt.core.IntervalTier(start_time=input_tg.start_time, end_time=input_tg.end_time, name="comentarios", objects=None)
        output_tg.add_tier(comments2_tier) 

        #for name in output_tg.get_tier_names():
        #    print(name)

        #print("Vou escrever o textgrid no arquivo agr e fim da função")
        tgt.io.write_to_file(output_tg, output_tg_file, format='long', encoding='utf-8') 
        return silences, timestamps_dsrs_1, timestamps_dsrs_2

    def ser(self, annot_tg, method_tg, boundary_type, timestamps_silences, timestamps_dsrs_1, timestamps_dsrs_2, hits_threshold):
        if boundary_type not in ["TB", "NTB"]:
            print("boundary_type inválido")
            return 0

        Annot_tg = tgt.io.read_textgrid(annot_tg, self.predict_encoding(annot_tg), include_empty_intervals=True)
        Method_tg = tgt.io.read_textgrid(method_tg, self.predict_encoding(method_tg), include_empty_intervals=True)

        agreement_tier = tgt.core.IntervalTier(start_time=Annot_tg.start_time, end_time=Annot_tg.end_time, name="concordancia", objects=None)

        names_annot = Annot_tg.get_tier_names()
        names_method = Method_tg.get_tier_names()

        method_boundaries = []

        for name in names_method:
            tier = Method_tg.get_tier_by_name(name)
            print("Biron tier - Name:",name,"Qtd intervalos", len(tier.intervals))
            if boundary_type in name:
                for interval in tier.intervals:
                    method_boundaries.append([interval, '0', ''])
                Annot_tg.add_tier(tier)
        print("")
        end_flag = False
        I = 0
        R = 0
        for name in names_annot:
            tier = Annot_tg.get_tier_by_name(name)
            print("Annot tier, Name:",name,"Qtd intervalos", len(tier.intervals))
            if boundary_type in name:
                for interval in tier.intervals:
                    if interval.start_time < method_boundaries[0][0].start_time or interval.start_time > method_boundaries[-1][0].end_time:
                        continue
                    for mb in method_boundaries:
                        if abs(interval.start_time - mb[0].start_time) < hits_threshold:
                            mb[1] = '1'
                            mb[2] = ''
                            if mb[0].start_time in timestamps_dsrs_1:
                              mb[2]  = 'dsrs 1 '
                            if mb[0].start_time in timestamps_dsrs_2:
                              mb[2] += 'dsrs 2 '
                            if mb[0].start_time in silences:
                              mb[2] += 'sil '
                        else:
                            R += 1
                        if mb[0].end_time == Method_tg.end_time: # ENTRA TODA VEZ NESSA CONDIÇÃO, PRECISA OTIMIZAR
                            if abs(interval.end_time - mb[0].end_time) < hits_threshold:
                                end_flag = True
                        if (mb[0].start_time + hits_threshold > interval.start_time):
                          break
        print("")
        hits = 0
        hits_dsrs_1 = 0
        hits_dsrs_2 = 0
        hits_silences = 0
        print("method boundaries", len(method_boundaries))
        for mb in method_boundaries:
            if mb[1] == '1':
                hits += 1
                print("mb", mb)
                if 'dsrs 1' in mb[2]:
                  hits_dsrs_1  += 1
                if 'dsrs 2' in mb[2]:
                  hits_dsrs_2  += 1
                if 'sil' in mb[2]:
                  hits_silences  += 1
            if mb[1] == '0':
                I += 1

        if end_flag:
            hits += 1
            hits_silences +=1
        else:
            I += 1

        C = hits

        print("Acurácia:",C, "/ ?")
        print("Hits da heurística 1:", hits_dsrs_1,"/")
        print("Hits da heurística 2:", hits_dsrs_2,"/")
        print("Hits da heurística de silêncios", hits_silences,"/")
        return C # acurácia
        #return SER

    def metrics(self, annot_rg, method_tg, timestamps_silences, timestamps_dsrs_1, timestamps_dsrs_2, hits_threshold):
        boundary_types = ["TB", "NTB"]
        Annot_tg = tgt.io.read_textgrid(annot_tg, self.predict_encoding(annot_tg), include_empty_intervals=True)
        Method_tg = tgt.io.read_textgrid(method_tg, self.predict_encoding(method_tg), include_empty_intervals=True)

        all_timestamps = timestamps_dsrs_1 + timestamps_dsrs_2 + timestamps_silences
        names_annot = Annot_tg.get_tier_names()
        names_method = Method_tg.get_tier_names()
        print(names_annot, names_method)
        hits_list = []
        total_hits = 0
        aux_hits = 0
        TB_hits = 0
        NTB_hits = 0

        for name in names_annot:
            print(name)
            try:
                tier_annot = Annot_tg.get_tier_by_name(name)
                tier_method = Method_tg.get_tier_by_name(name)
                index_method = 0
                index_annot = 0
                # Basta que um textgrid acabe para que possa ser feito um último hit entre o tempo final do textgrid acabado e algum próximo intervalo do textgrid que não acabou:
                # ARRUMAR ESSA CONDIÇÃO
                while index_method < len(tier_method): #and tier_annot.intervals[index_annot].start_time < tier_method.intervals[-1].end_time:
                    #print(tier_annot.intervals[index_annot].start_time, tier_method.intervals[index_method].start_time)
                    # Se for um hit, ambos os índices avançam
                    if abs (tier_annot.intervals[index_annot].start_time - tier_method.intervals[index_method].start_time) < hits_threshold:
                        aux_hits += 1
                        #print("HIT", tier_annot.intervals[index_annot].start_time, tier_method.intervals[index_method].start_time)
                        #print(" ")
                        index_annot += 1
                        index_method += 1
                    # Se não for hit, o índice que apontar para o intervalo que começa primeiro avança
                    elif tier_annot.intervals[index_annot].start_time < tier_method.intervals[index_method].start_time:
                        index_annot += 1
                    else:
                        index_method += 1
                # Verifica se o tempo final da camada do método também é um hit
                print("Final timestamps compared:",tier_annot.intervals[index_annot].start_time, tier_method.intervals[-1].end_time)
                print("")
                if abs (tier_annot.intervals[index_annot].start_time - tier_method.intervals[-1].end_time) < hits_threshold: # Verifica se o tempo final é um hit
                    aux_hits += 1
                    print("HIT", tier_annot.intervals[-1].end_time, tier_method.intervals[-1].end_time)
                    print(" ")
                print("Hits layer "+name+":", aux_hits)
                print("(method)Quantity of intervals layer "+name+":", len(tier_method.intervals))
                print("(annot)Quantity of intervals compared(index):layer "+name+":", index_annot)
                if "NTB" in name:
                    NTB_hits += aux_hits
                elif "TB" in name:
                    TB_hits += aux_hits
                #total_hits += aux_hits
                aux_hits = 0
            except:
                continue
        
        print("TB Hits:", TB_hits)
        print("NTB Hits:", NTB_hits)
        #print("Total hits:",total_hits)


# Organizando caminhos

# Inquérito selecionado
#inq = "SP_EF_156"
#inq = "SP_D2_255"
#inq = "SP_DID_242"
inq = "SP_D2_012"
i = 1
segments_quantity = 8
alignment_tg_list = []
locs_files_list = []
rel_path_inq = "Mestrado/" + inq + "_segmentado/"
concatenated_tg_file = rel_path_inq + inq + "_concatenated.TextGrid"
concatenated_locs_file = rel_path_inq + inq + "_locutores.txt"
concatenated_locs_words_file = rel_path_inq + inq + "_locutores_palavras.txt"
output_tg_file = rel_path_inq + inq + "_OUTPUT.TextGrid"
for i in range (1,segments_quantity+1):
    segment_number = str(i)
    path = rel_path_inq + inq + "_" + segment_number + "/" + inq 
    clipped_path = path + "_clipped_" + segment_number
    locs_file =  clipped_path + "_locutores.txt"
    #locs_words_file = clipped_path + "_locutores_palavras.txt"
    alignment_tg = clipped_path + ".TextGrid"
    #output_tg_file = clipped_path + "_OUTPUT.TextGrid"
    annot_tg = path + ".TextGrid"

    #Criando a classe com todas as funções que serão utilizadas
    Segmentation = AutomaticSegmentation(path, locs_file)

    # Pré-processando text grid de entrada
    Segmentation.remove_overlaps(alignment_tg)
    alignment_tg_list.append(alignment_tg)
    locs_files_list.append(locs_file)

# Juntando todos os textgrids de entrada
Segmentation.concatenate_textgrids(alignment_tg_list, concatenated_tg_file)


# Juntando todos os arquivos de locutores
Segmentation.concatenate_locs_file(locs_files_list, concatenated_locs_file)

# Gera arquivo de palavras por locutor
Segmentation.generate_words_file(concatenated_locs_file, concatenated_locs_words_file)

# Parâmetros
window_size = 0.3
delta1 = 0.88
delta2 = 0.70
interval_size = 3
silence_threshold = 0.3
min_words_h2 = 10
hits_threshold = 0.25

# Aplicando o método
silences, dsrs_1, dsrs_2 = Segmentation.find_boundaries(concatenated_locs_words_file, concatenated_tg_file, annot_tg, output_tg_file, window_size, delta1, delta2, silence_threshold, interval_size, min_words_h2)

# Imprimindo alguns dados
print("silences", silences)
print("Quantity of boundaries obtained with the silences heuristic:",len(silences))
print("dsrs1", dsrs_1)
print("Quantity of boundaries obtained with the first heuristic:",len(dsrs_1))
print("dsrs2", dsrs_2)
print("Quantity of boundaries obtained with the second heuristic:",len(dsrs_2))
print("Total:", len(silences)+len(dsrs_1)+len(dsrs_2))
print(output_tg_file, "SUCCESS" )

# Métricas
#Segmentation.ser(annot_tg, output_tg_file, "NTB", silences, dsrs_1, dsrs_2, hits_threshold)
Segmentation.metrics(annot_tg, output_tg_file, silences, dsrs_1, dsrs_2, hits_threshold)

# 6 parâmetros: tamanho da janela: 0.3                      (em s, deve ser positivo e não deve ser grande, talvez no max 1s)
#               threshold da 1a heurística (porcentagem
#                   da maior diferença de taxas de fala
#                   de janelas consecutivas para caracterizar
#                   DSR): 0.88                              (no intervalo [0, 1] e não muito pequeno, talvez no min 0.5 ou 0.6)
#               threshold da 2a heurística: 0.70            (no intervalo [0, 1], sempre abaixo do parâmetro anterior)
#               duração de segundos sem DSRs na primeira heuristica para contemplar a 2a: 3 (em s, positivo)
#               duração de silêncio para caracterizar pausa: 0.3 (em s, talvez no mínimo 0.15 e no max 1s)
#               palavras consecutivas sem DSR na 1a heuristica para contemplar a 2a: 10 (número inteiro talvez entre 5 e 20)
