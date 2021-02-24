import requests
from bs4 import BeautifulSoup
import requests
from collections import Counter

def get_root_page_data(url):
    response = requests.get(url)
    # Get all links in root page
    parsed_page = BeautifulSoup(response.content, 'html.parser')
    
    references = parsed_page.find("div", {"class": "mw-references-wrap mw-references-columns"})
    references.extract()
    
    page_content = parsed_page.find(id="bodyContent").getText()

    # get only links within the main body content
    allLinks = parsed_page.find(id="bodyContent").find_all("a")
    linksToScrape = []
    for link in allLinks:
        href = link.get('href')
        # select backlinks to only wikipedia articles
        if (href is not None and href.find("/wiki/") == 0 and "identifier" not in href):
            class_name = link.get('class')
            # remove image URLS
            if(class_name is not None and "image" not in class_name and "internal" not in class_name):
                linksToScrape.append(href)
    return linksToScrape, page_content


def get_page_content(url):
    response = requests.get(url)
    parsed_page = BeautifulSoup(response.content, 'html.parser')
    
    references = parsed_page.find("div", {"class": "mw-references-wrap mw-references-columns"})
    if (references is not None):
        references.extract()
    
    page_content = parsed_page.find(id="bodyContent").getText()
    return page_content


def scrape_sub_pages(root_page_content, linksToScrape):
    base_url = "https://en.wikipedia.org"
    all_wiki_content = root_page_content
    for link in linksToScrape:
        full_url = f"{base_url}{link}"
        print(f"Parsing page content for {full_url}")
        current_page_content = get_page_content(full_url)
        all_wiki_content = all_wiki_content + " " + current_page_content
    print("Parsing content complete")
    return all_wiki_content


def write_to_file(document, file):
    with open(file, 'w') as f:
        f.write(document)
    print("file write to disk complete")

def build_frequency(token_array):
    freqs = {}
    for token in token_array:
        if token in freqs:
            freqs[token] += 1
        else:
            freqs[token] = 1
    return freqs

def get_most_freqeuent_words(word_frequency, count):
    top_tokens = dict(Counter(word_frequency).most_common(count))
    return top_tokens