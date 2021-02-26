import pandas as pd

import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, KneserNeyInterpolated, Vocabulary

from data_pipeline import extract_sentences

AVG_PERPLEXITY = 0.0

def unigram_model(train_sentences):
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) 
                    for sent in train_sentences]
    n = 1
    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
    model = MLE(n)
    model.fit(train_data, padded_vocab)

    test_sentences = ['an apple', 'an ant']
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) 
                    for sent in test_sentences]

    test_data, _ = padded_everygram_pipeline(n, tokenized_text)
    for test in test_data:
        print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

    test_data, _ = padded_everygram_pipeline(n, tokenized_text)

    for i, test in enumerate(test_data):
        print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))

def bigram_model(train_sentences):
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]

    n = 2
    train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    words = [word for sent in tokenized_text for word in sent]
    words.extend(["<s>", "</s>"])
    padded_vocab = Vocabulary(words)
    model = MLE(n)
    model.fit(train_data, padded_vocab)

    test_sentences = ['an apple', 'an ant']
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]

    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    for test in test_data:
        print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    for i, test in enumerate(test_data):
        print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))

def generate_model(raw_root_page_content_file):
    sentences = extract_sentences(raw_root_page_content_file, 900)
    print('sentences[100]')
    print(sentences[100])
    print('sentences[200]')
    print(sentences[200])
    print('sentences[300]')
    print(sentences[300])
    print(len(sentences))
    # print("sentences", sentences)
    # sentences = ['an apple', 'an orange']
    # print("sentences", sentences)
    unigram_model(sentences)
    bigram_model(sentences)
