import os
from os import getcwd
import matplotlib.pyplot as plt
from requests.api import get
from wordcloud import WordCloud

from process_data import fetch_data, pre_process_text, get_word_frequency, build_tweet_vocab, extract_sentences
from utils import build_frequency, get_most_freqeuent_words
try:
    os.mkdir(f"{getcwd()}/data/")
    os.mkdir(f"{getcwd()}/data/pages")
    os.mkdir(f"{getcwd()}/results/")
except OSError:
    pass

all_page_content_file = f"{getcwd()}/data/AllContent.txt"
root_page_content_file = f"{getcwd()}/data/RootContent.txt"
raw_root_page_content_file = f"{getcwd()}/data/RootContentRaw.txt"


lemma_content_file = f"{getcwd()}/data/lemmas.txt"
token_content_file = f"{getcwd()}/data/tokens.txt"

# uncomment to refetch data from wikipedia
# fetch_data(root_page_content_file, all_page_content_file, raw_root_page_content_file)
# uncomment to reprocess data
# pre_process_text(all_page_content_file, lemma_content_file, token_content_file)

######################################################
## Question 5.1                                    ###
######################################################

# word_frequency, top_words  = get_word_frequency(lemma_content_file)
# print(top_words)
# wc = WordCloud()
# wc.generate_from_frequencies(word_frequency)
# wc.to_file(f"{getcwd()}/results/wiki-covid-cloud.png")

# plt.figure()
# plt.imshow(wc, interpolation='bilinear')
# plt.axis("off")
# plt.show()

# last 10000 tweets

######################################################
## Question 5.2                                    ###
######################################################
# test_tweet_tokens, test_tweet_types = build_tweet_vocab("test")
# print("len test_tweet_tokens", len(test_tweet_tokens))
# print("len test_tweet_types", len(test_tweet_types))

# oov_tweet_types_counter = 0
# oov_tweet_tokens_counter = 0
# for tweet_type in test_tweet_types:
#     if (tweet_type not in word_frequency):
#         oov_tweet_types_counter += 1

# for tweet_token in test_tweet_tokens:
#     if (tweet_token not in word_frequency):
#         oov_tweet_tokens_counter += 1

# print("oov_tweet_types_counter count", oov_tweet_types_counter)
# print("oov_tweet_tokens_counter count", oov_tweet_tokens_counter)

# oov_tweet_types_norm = oov_tweet_types_counter/len(test_tweet_types)
# oov_tweet_tokens_norm = oov_tweet_tokens_counter/len(test_tweet_tokens)
# print("test oov_tweet_types_norm", oov_tweet_types_norm)
# print("test oov_tweet_tokens_norm", oov_tweet_tokens_norm)

# train_tweet_tokens, train_tweet_types = build_tweet_vocab("train")
# print("len train_tweet_tokens", len(train_tweet_tokens))
# print("len train_tweet_types", len(train_tweet_types))

# oov_tweet_tokens_counter = 0
# for tweet_token in test_tweet_tokens:
#     if (tweet_token not in train_tweet_types):
#         oov_tweet_tokens_counter += 1
# print("oov_tweet_tokens_counter", oov_tweet_tokens_counter)
# oov_tweet_tokens_norm = oov_tweet_tokens_counter/len(train_tweet_tokens)
# print("train oov_tweet_tokens_norm", oov_tweet_tokens_norm)


######################################################
## Question 5.3                                    ###
######################################################
sentences = extract_sentences(raw_root_page_content_file, 9000)
print(sentences)
print(len(sentences))