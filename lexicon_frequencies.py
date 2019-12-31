import requests
import time
import re
import pandas as pd
from nltk import tokenize
import numpy as np
from collections import Counter

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
    if not regex.search(request.text):
        print(request.text)
        return None
    frequencies = regex.search(request.text).group()
    frequencies = frequencies[13:-1]
    frequencies = frequencies.split(", ")
    frequencies = [float(f) for f in frequencies]
    results.append(np.mean([f for f in frequencies[EAP_start-1818:EAP_end-1818] if f>0]))
    results.append(np.mean([f for f in frequencies[HPL_start-1818:HPL_end-1818] if f>0]))
    results.append(np.mean([f for f in frequencies[MWS_start-1818:MWS_end-1818] if f>0]))
    return results

def get_lexicon_frequencies(lexicon):
    filename = 'train.csv'
    lexicon_frequencies = []
    error_counter = 0
    sleep_time = 0.1
    BASEURL = "https://books.google.com/ngrams/graph?content={}&year_start=1818&year_end=1943&corpus=16&smoothing=0"
    for n, word in enumerate(lexicon):
        print(n)
        if n%1378 == 0:
            print("{}% done!".format(round(n/len(lexicon)*100), 2))
        if n%75 == 74:
            time.sleep(360)
        request = requests.get(BASEURL.format(word))
        if re.search(r'No valid ngrams to plot!', request.text):
            pass
        if request.status_code != 200:
            error_counter += 1
            time.sleep(sleep_time)
            request = requests.get(BASEURL.format(word))
            if request.status_code != 200:
                error_counter += 1
                print("two consecutive errors")
                print(request.status_code, request.text)
                time.sleep(120)
        ngram_frequencies = get_ngram_frequency_from_request(request, ngram_data_regex)
        if ngram_frequencies:
            lexicon_frequencies.append((word, ngram_frequencies))
        time.sleep(sleep_time)
    return lexicon_frequencies

def main():
	df = pd.read_csv('train.csv', index_col='id')
	token_list = []
	stopWords = set(stopwords.words('english'))
	for sentence in df.text:
		tokens = tokenize.word_tokenize(sentence)
		tokens = [t for t in tokens if len(t)>1 and t not in stopWords]
		token_list.append(tokens)
	wordcounts = Counter()
	for sentence in token_list:
		wordcounts.update([t.lower() for t in sentence])
	wordlist = []
	for word, count in wordcounts.items():
		if 5 < count < 200:
			wordlist.append(word)
	lexicon_frequencies = get_lexicon_frequencies(wordlist)
	pd.DataFrame(lexicon_frequencies).to_csv('lexicon_frequencies.csv')


if __name__ == '__main__':
	main()
