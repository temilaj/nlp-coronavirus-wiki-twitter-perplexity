import os
from os import getcwd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from process_data import fetch_data, pre_process_text, get_word_frequency
from utils import build_frequency, get_most_freqeuent_words
try:
    os.mkdir(f"{getcwd()}/data/")
    os.mkdir(f"{getcwd()}/data/pages")
    os.mkdir(f"{getcwd()}/results/")
except OSError:
    pass

all_page_content_file = f"{getcwd()}/data/AllContent.txt"
root_page_content_file = f"{getcwd()}/data/RootContent.txt"
lemma_content_file = f"{getcwd()}/data/lemmas.txt"

# uncomment to refetch data from wikipedia
# fetch_data(root_page_content_file, all_page_content_file)
# uncomment to reprocess data
# pre_process_text(all_page_content_file, lemma_content_file)

word_frequency,top_words  = get_word_frequency(lemma_content_file)
print(top_words)
wc = WordCloud()
wc.generate_from_frequencies(word_frequency)
wc.to_file(f"{getcwd()}/results/wiki-covid-cloud.png")

plt.figure()
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# last 10000 tweets