import pandas as pd

import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, KneserNeyInterpolated, Vocabulary

from data_pipeline import extract_sentences, get_raw_tweets

def unigram_model(tokenized_text, test_sentences, sentence_count):
    n = 1
    average_perplexity = 0.0

    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
    model = MLE(n)
    model.fit(train_data, padded_vocab)

    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) 
                    for sent in test_sentences]

    test_data, _ = padded_everygram_pipeline(n, tokenized_text)
    for test in test_data:
        print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

    test_data, _ = padded_everygram_pipeline(n, tokenized_text)

    # for i, test in enumerate(test_data):
    #     print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))
    
    nbr_ignored = 0
    for test in list(test_data):
        ngrams = list(test)
        if model.perplexity(ngrams) != float('inf'):
            average_perplexity += model.perplexity(ngrams)
        else:
            nbr_ignored += 1

    average_perplexity /= sentence_count
    print(f"Average Perplexity for Unigram model on Test tweets: {round(average_perplexity, 4)} **")
    print(f"** Number of tweets that were ignored due to unseen words is ({nbr_ignored}) **")

def bigram_model(tokenized_text, test_sentences, sentence_count):

    n = 2
    average_perplexity = 0.0
    train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    words = [word for sent in tokenized_text for word in sent]
    words.extend(["<s>", "</s>"])
    padded_vocab = Vocabulary(words)
    model = MLE(n)
    model.fit(train_data, padded_vocab)

    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]

    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    # for test in test_data:
    #     print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

    nbr_ignored = 0
    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    for test in list(test_data):
        ngrams = list(test)
        if model.perplexity(ngrams) != float('inf'):
            average_perplexity += model.perplexity(ngrams)
        else:
            nbr_ignored += 1

    average_perplexity /= sentence_count
    print(f"Average Perplexity for Bigram model on Test tweets: {round(average_perplexity, 4)} **")
    print(f"** Number of tweets that were ignored due to unseen words is ({nbr_ignored}) **")

def trigram_model(tokenized_text, test_sentences, sentence_count):

    n = 3
    average_perplexity = 0.0
    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
    model = KneserNeyInterpolated(n)
    model.fit(train_data, padded_vocab)

    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]
    
    test_data, _ = padded_everygram_pipeline(n, tokenized_text)

    nbr_ignored = 0
    for test in list(test_data):
        ngrams = list(test)
        if model.perplexity(ngrams) != float('inf'):
            average_perplexity += model.perplexity(ngrams)
        else:
            nbr_ignored += 1

    average_perplexity /= sentence_count
    print(f"Average Perplexity for Trigram model on Test tweets: {round(average_perplexity, 4)}")
    # print(f"** Number of tweets that were ignored due to unseen words is ({nbr_ignored}) **")

def generate_model(raw_root_page_content_file, n_gram):
    train_sentences = list(extract_sentences(raw_root_page_content_file, 9000))
    # train_sentences = ['an apple', 'an orange']
    print("len(train_sentences)", len(train_sentences))
    test_sentences = get_raw_tweets("test")
    # test_sentences = ['an apple', 'an ant']
    print("len(test_sentences)", len(test_sentences))
    # print(test_sentences[0])
    # print(test_sentences[10])
    # print(test_sentences[100])
    # print(test_sentences[300])
    print("len(test_sentences)", len(test_sentences))
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]
    print(len(tokenized_text))
    if (n_gram == 1):
        unigram_model(tokenized_text, test_sentences, len(test_sentences) )
    elif (n_gram == 2):
        bigram_model(tokenized_text, test_sentences, len(test_sentences))
    elif (n_gram == 3):
        trigram_model(tokenized_text, test_sentences, len(test_sentences))
    else:
        print("Unsupported N-Gram selected")
            
