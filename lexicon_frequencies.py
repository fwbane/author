import requests
import time
import re
import pandas as pd
from nltk import tokenize


EAP_start = 1827
EAP_end = 1849
EAP_major_works = [1831, 1839, 1843, 1845, 1846, 1849]
HPL_start = 1910
HPL_end = 1943
HPL_major_works = [1928, 1936, 1943]
MWS_start = 1818
MWS_end = 1837
MWS_major_works = [1818, 1823, 1826, 1830, 1835, 1837]
import string
from nltk.corpus import stopwords

ngram_data_regex = re.compile(r'imeseries\": \[(\d|\.|\s|,|e|-)*\]')

def get_ngram_frequency_from_request(request, regex):
    results = []
    frequencies = regex.search(request.text).group()
    frequencies = frequencies[13:-1]
    frequencies = frequencies.split(", ")
    frequencies = [float(f) for f in frequencies]
    results.append(np.mean([f for f in frequencies[EAP_start-1818:EAP_end-1818] if f>0]))
    results.append(np.mean([f for f in frequencies[HPL_start-1818:HPL_end-1818] if f>0]))
    results.append(np.mean([f for f in frequencies[MWS_start-1818:MWS_end-1818] if f>0]))
    return results

def get_lexicon_frequencies(lexicon):
    lexicon_frequencies = []
    error_counter = 0
    sleep_time = 1
    BASEURL = "https://books.google.com/ngrams/graph?content={}&year_start=1818&year_end=1943&corpus=16&smoothing=0"
    for n, word in enumerate(lexicon):
        if n%1378 == 1:
            print("{}% done!".format(round(len(lexicon)/n), 2))
        if error_counter > 8:
            print("too many errors. Sleep time = {}".format(sleep_time))
            break
        request = requests.get(BASEURL.format(word))
        if request.status_code != 200:
            error_counter += 1
            time.sleep(sleep_time)
            request = requests.get(BASEURL.format(word))
            if request.status_code != 200:
                error_counter += 1
                print("two consecutive errors")
                print(request.status_code, request.text)
                time.sleep(300)
                sleep_time *= 2
        ngram_frequencies = get_ngram_frequency_from_request(request, ngram_data_regex)
        lexicon_frequencies.append((word, ngram_frequencies))
        time.sleep(sleep_time)
    return lexicon_frequencies

def main():
	df = pd.read_csv(filename, index_col='id')
	token_list = []
	stopWords = set(stopwords.words('english'))
	for sentence in df.text:
		tokens = tokenize.word_tokenize(sentence)
		tokens = [t for t in tokens if len(t)>1 and t not in stopWords]
		token_list.append(tokens)
	lexicon = set()
	for sentence in token_list:
		lexicon.update(sentence)
	lexicon_frequencies = get_lexicon_frequencies(lexicon)
	pd.DataFrame(lexicon_frequencies).to_csv('lexicon_frequencies.csv')


if __name__ == '__main__':
	main()