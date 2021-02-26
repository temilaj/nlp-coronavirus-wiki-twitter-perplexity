import spacy
import nltk
from os import getcwd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import json

from utils import scrape_sub_pages, get_root_page_data, write_to_file, build_frequency, get_most_freqeuent_words

nltk.download('stopwords')
nltk.download('punkt')
spacy_nlp = spacy.load('en_core_web_sm')
all_stopwords = spacy_nlp.Defaults.stop_words


def clean_text(text):
    # replace new lines and carraige returns with space
    text = text.replace("\n", " ").replace("\r", " ")
    
    # replace numbers and puntuations (except single quote) with space
    punc_list = '!"#$%&()*+,-./:;<=>?@[\]^_{|}~' + '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.translate(t)
    
    # replace single quote with empty character
    t = str.maketrans(dict.fromkeys("'`", ""))
    text = text.translate(t)
    return text

def fetch_data(root_page_content_file, all_page_content_file, raw_page_content_file):
    # wikipedia has a HTTP 301 redirect from https://en.wikipedia.org/wiki/COVID-19  to https://en.wikipedia.org/wiki/Coronavirus_disease_2019
    linksToScrape, root_page_content = get_root_page_data(url="https://en.wikipedia.org/wiki/Coronavirus_disease_2019")
    all_wiki_content = scrape_sub_pages(root_page_content, linksToScrape)
    raw_page_content = root_page_content

    root_page_content = clean_text(root_page_content)
    all_wiki_content = clean_text(all_wiki_content)
    
    write_to_file(root_page_content, root_page_content_file)
    write_to_file(all_wiki_content, all_page_content_file)
    write_to_file(raw_page_content, raw_page_content_file)

def tokenize(doc):
    text_tokens = word_tokenize(doc)
    tokens = [word for word in text_tokens if not word in stopwords.words()]
    return tokens

def get_types(tokens):
	word_types = {}
	for word_type in tokens:
		if word_type not in word_types:
			word_types[word_type] = 0
		word_types[word_type] += 1
	return word_types 

def lemmatize(text):
    doc = spacy_nlp(text)
    lemmas = [word.lemma_.lower()  for word in doc if not (word.is_stop or word.is_punct) ]
    return lemmas

def pre_process_text(file, lemma_file, token_file):
    f = open(file, "r")
    wiki_content = f.read().split()
    word_lenth = len(wiki_content)
    index = 0
    step = 100000
    all_lemmas = []
    all_tokens = []
    while (index < word_lenth):
        print(f"index {index},  step {step}")
        text = " " 
        current_content = wiki_content[index:index+step]
        current_content = text.join(current_content)
        tokens = tokenize(current_content)
        all_tokens += tokens
        text = " ".join(tokens)
        lemmas = lemmatize(text)
        all_lemmas += lemmas
        index += step;
    
    write_to_file(" ".join(all_lemmas), lemma_file)
    write_to_file(" ".join(all_tokens), token_file)

def get_word_frequency(file):
    """
        generate vocabulary of words
    """
    f = open(file, "r")
    lemmas = f.read().split()
    word_frequency = build_frequency(lemmas)
    top_words = get_most_freqeuent_words(word_frequency, 20)
    return word_frequency, top_words


def extract_sentences(file, length):
    """
        extract senteces from text file
    """
    f = open(file, "r")
    content = f.read()

    tokens = nltk.sent_tokenize(content)
    sentences = tokens[:length]
    
    # doc = spacy_nlp(content)
    # sentences = list(doc.sents)[:length]
    return sentences

def get_raw_tweets(tweet_type):
    """
        extract tweets from csv file
    """
    tweet_file = f"{getcwd()}/data/cleaned_tweets.csv"
    tweets = pd.read_csv(tweet_file)
    if (tweet_type == "train"):
        return tweets['text_lemmatized'].tolist()[:9000]
    elif(tweet_type == "test"):
        return tweets['text_lemmatized'].tolist()[9000:]

def build_tweet_vocab(vocab_type):
    """
        build vocabulary from all the tweets supplied
    """
    test_tweets = get_raw_tweets(vocab_type)
    test_tweets = " ".join(test_tweets)
    tweet_tokens = tokenize(test_tweets)
    tweet_types = get_types(tweet_tokens)
    # tweet_types = build_frequency(tweet_tokens)
    return tweet_tokens, tweet_types

def process_tweet_data(test_tweets_file, train_tweets_file):
    """
        build vocabulary from all the tweets supplied
        and Save the processed tweets to file
    """
    test_tweet_tokens, test_tweet_types = build_tweet_vocab("test")
    test_tweets = {}
    test_tweets["types"] = test_tweet_types
    test_tweets["tokens"] = test_tweet_tokens
    test_tweets = json.dumps(test_tweets)
    write_to_file(test_tweets, test_tweets_file)

    train_tweet_tokens, train_tweet_types = build_tweet_vocab("train")
    train_tweets = {}
    train_tweets["types"] = train_tweet_types
    train_tweets["tokens"] = train_tweet_tokens
    train_tweets = json.dumps(train_tweets)
    write_to_file(train_tweets, train_tweets_file)


def get_processed_tweets(test_tweets_file, train_tweets_file):
    test_tweets = ""
    train_tweets = ""
    with open(test_tweets_file, "r") as f:
        print(f"read {test_tweets_file} complete")
        test_tweets = json.load(f)

    with open(train_tweets_file, "r") as f:
        print(f"read {train_tweets_file} complete")
        train_tweets = json.load(f)

    return test_tweets, train_tweets

