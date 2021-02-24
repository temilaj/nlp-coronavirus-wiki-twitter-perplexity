import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils import scrape_sub_pages, get_root_page_data, write_to_file

# spacy_nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
nltk.download('stopwords')
spacy_nlp = spacy.load('en_core_web_sm')
all_stopwords = spacy_nlp.Defaults.stop_words


def process_data(root_page_content_file, all_page_content_file):
    # wikipedia has a HTTP 301 redirect from https://en.wikipedia.org/wiki/COVID-19  to https://en.wikipedia.org/wiki/Coronavirus_disease_2019
    linksToScrape, root_page_content = get_root_page_data(url="https://en.wikipedia.org/wiki/Coronavirus_disease_2019")
    all_wiki_content = scrape_sub_pages(root_page_content, linksToScrape)

    write_to_file(root_page_content, root_page_content_file)
    write_to_file(all_wiki_content, all_page_content_file)

def tokenize(doc):
    text_tokens = word_tokenize(doc)
    tokens = [word for word in text_tokens if not word in stopwords.words()]
    return tokens

def lemmatize(doc):
    lemmas = [word.lemma_.lower()  for word in doc if not (word.is_stop or word.is_punct) ]
    return lemmas

def pre_process_text(file, lemma_file):
    f = open(file, "r")
    wiki_content = f.read().split()
    word_lenth = len(wiki_content)
    index = 0
    step = 100000
    all_lemmas = []
    while (index < word_lenth):
        print(f"index {index},  step {step}")
        text = " " 
        current_content = wiki_content[index:index+step]
        current_content = text.join(current_content)
        tokens = tokenize(current_content)
        text = " ".join(tokens)
        doc = spacy_nlp(text)
        lemmas = [word.lemma_.lower()  for word in doc if not (word.is_stop or word.is_punct) ]
        all_lemmas += lemmas
        index += step;

    
    write_to_file(" ".join(all_lemmas), lemma_file)
