import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, KneserNeyInterpolated

from data_pipeline import extract_sentences, get_raw_tweets

def trigram_model(tokenized_text, test_sentences, sentence_count):

    n = 3
    average_perplexity = 0.0
    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
    model = KneserNeyInterpolated(n)
    model.fit(train_data, padded_vocab)

    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]
    
    test_data, _ = padded_everygram_pipeline(n, tokenized_text)

    for test in list(test_data):
        ngrams = list(test)
        if model.perplexity(ngrams) != float('inf'):
            average_perplexity += model.perplexity(ngrams)

    average_perplexity /= sentence_count
    print(f"Average Perplexity for Trigram model on Test tweets: {round(average_perplexity, 4)}")

def generate_model(raw_root_page_content_file):
    train_sentences = list(extract_sentences(raw_root_page_content_file, 9000))
    test_sentences = get_raw_tweets("test")
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]
    trigram_model(tokenized_text, test_sentences, len(test_sentences))
