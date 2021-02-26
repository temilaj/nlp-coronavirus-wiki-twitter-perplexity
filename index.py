import os
from os import getcwd
import matplotlib.pyplot as plt
from requests.api import get
from wordcloud import WordCloud

from data_pipeline import fetch_data, pre_process_text, get_word_frequency, process_tweet_data, get_processed_tweets
from language_model import generate_model
from utils import build_frequency, get_most_freqeuent_words, write_to_file

try:
    os.mkdir(f"{getcwd()}/results/")
except OSError:
    pass

all_page_content_file = f"{getcwd()}/data/AllContent.txt"
root_page_content_file = f"{getcwd()}/data/RootContent.txt"
raw_root_page_content_file = f"{getcwd()}/data/RootContentRaw.txt"

lemma_content_file = f"{getcwd()}/data/lemmas.txt"
token_content_file = f"{getcwd()}/data/tokens.txt"

test_tweets_file = f"{getcwd()}/data/ProcessedTestTweets.json"
train_tweets_file = f"{getcwd()}/data/ProcessedTrainTweets.json"

# uncomment to refetch data from wikipedia
# fetch_data(root_page_content_file, all_page_content_file, raw_root_page_content_file)
# uncomment to reprocess data from wikipedia
# pre_process_text(all_page_content_file, lemma_content_file, token_content_file)

######################################################
## Question 5.1                                    ###
######################################################

word_frequency, top_words  = get_word_frequency(lemma_content_file)
# print(top_words)
# wc = WordCloud()
# wc.generate_from_frequencies(word_frequency)
# wc.to_file(f"{getcwd()}/results/wiki-covid-cloud.png")

# plt.figure()
# plt.imshow(wc, interpolation='bilinear')
# plt.axis("off")
# plt.show()

######################################################
## Question 5.2                                    ###
######################################################

# uncomment to reprocess data from scraped tweets
# process_tweet_data(test_tweets_file, train_tweets_file)

# test_tweets, train_tweets = get_processed_tweets(test_tweets_file, train_tweets_file)
# test_tweet_tokens = test_tweets["tokens"]
# test_tweet_types = test_tweets["types"] 

# train_tweet_tokens = train_tweets["tokens"]
# train_tweet_types = train_tweets["types"] 

# def get_oov_rate(corpus, vocab):
#     counter = 0
#     for word in corpus:
#         if (word not in vocab):
#             counter += 1
#     oov_rate = counter/len(corpus)
#     return oov_rate


# oov_tweet_types_norm = get_oov_rate(test_tweet_types, word_frequency)
# print("out of vocabulary Word types in tweets normalized by the number of word types tweet = ", oov_tweet_types_norm)

# oov_tweet_tokens_norm = get_oov_rate(test_tweet_tokens, word_frequency)
# print("out of vocabulary Word tokens in tweets normalized by the number of word tokens tweet = ", oov_tweet_tokens_norm)


# oov_tweet_tokens_norm = get_oov_rate(test_tweet_tokens, train_tweet_types)
# print("OOV-rate of your tweet test set = ", oov_tweet_tokens_norm)


######################################################
## Question 5.3                                    ###
######################################################
generate_model(raw_root_page_content_file)