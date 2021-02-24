import os
from os import getcwd
from bs4 import BeautifulSoup

# import matplotlib.pyplot as plt
# from nltk.tokenize import word_tokenize
# from string import punctuation
# from nltk.stem.snowball import SnowballStemmer
# from wordcloud import WordCloud

# uncomment to refetch data from wikipedia
from process_data import process_data, pre_process_text
try:
    os.mkdir(f"{getcwd()}/data/")
    os.mkdir(f"{getcwd()}/data/pages")
except OSError:
    pass

all_page_content_file = f"{getcwd()}/data/AllContent.txt"
root_page_content_file = f"{getcwd()}/data/RootContent.txt"
lemma_content_file = f"{getcwd()}/data/lemmas.txt"

process_data(root_page_content_file, all_page_content_file)
pre_process_text(all_page_content_file, lemma_content_file)