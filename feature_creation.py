import requests
import re
import numpy as np

def get_ngram_frequency(word):
    results = []
    EAP_start = 1827
    EAP_end = 1849
    EAP_major_works = [1831, 1839, 1843, 1845, 1846, 1849]
    HPL_start = 1910
    HPL_end = 1943
    HPL_major_works = [1928, 1936, 1943]
    MWS_start = 1818
    MWS_end = 1823
    MWS_major_works = [1818, 1823, 1826, 1830, 1835, 1837]
    request = requests.get("https://books.google.com/ngrams/graph?content={}&year_start=1818&year_end=1943&corpus=16&smoothing=0".format(word))
    regex = re.compile(r'imeseries\": \[(\d|\.|\s|,|e|-)*\]')
    frequencies = regex.search(request.text).group()
    frequencies = frequencies[13:-1]
    frequencies = frequencies.split(", ")
    frequencies = [float(f) for f in frequencies]
    results.append(np.mean([f for f in frequencies[EAP_start-1818:EAP_end-1818] if f>0]))
    results.append(np.mean([f for f in frequencies[HPL_start-1818:HPL_end-1818] if f>0]))
    results.append(np.mean([f for f in frequencies[MWS_start-1818:MWS_end-1818] if f>0]))
    return results