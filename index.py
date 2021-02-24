import os
from os import getcwd
# from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from process_data import process_data, pre_process_text, get_word_frequency
from utils import build_frequency, get_most_freqeuent_words
try:
    os.mkdir(f"{getcwd()}/data/")
    os.mkdir(f"{getcwd()}/data/pages")
except OSError:
    pass

all_page_content_file = f"{getcwd()}/data/AllContent.txt"
root_page_content_file = f"{getcwd()}/data/RootContent.txt"
lemma_content_file = f"{getcwd()}/data/lemmas.txt"

# uncomment to refetch data from wikipedia
# process_data(root_page_content_file, all_page_content_file)
# pre_process_text(all_page_content_file, lemma_content_file)
# pre_process_text(root_page_content_file, lemma_content_file)

word_frequency,top_words  = get_word_frequency(lemma_content_file)

wc = WordCloud()
wc.generate_from_frequencies(word_frequency)

plt.figure()
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()