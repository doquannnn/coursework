from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.corpus import stopwords
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import requests

# graphic terms
with open ("graphics_glossary.txt", 'r') as rf:
    glossary = set([w.strip() for w in rf.readlines()])


def word_frequencies(year, url, plotting=False):
    headers = {"User-Agent" : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}
    page = requests.get(url, headers=headers).text
    doc = BeautifulSoup(page, 'html.parser')


    keywords = defaultdict(lambda: 0)
    stop_words = set(stopwords.words('english'))

    content = doc.find(class_="content-item")
    a_tags = content.find_all("a")

    current_year = []
    for a_tag in a_tags:
        if re.search(re.compile(year), a_tag.text):
            current_year.append(a_tag['href'])

    current_year.pop(2)  # the third one is a blog
    conferences = [os.path.join(url, conference) for conference in current_year]

    for conference in conferences:
        headers = {"User-Agent" : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}
        page = requests.get(conference, headers=headers).text
        doc = BeautifulSoup(page, 'html.parser')

        dt_tags = doc.find_all("dt")
        for title in dt_tags:
            sentence = title.text.strip().lower().split()
            for word in sentence:
                if word.endswith("s"):
                    w = word[:-1]
                else:
                    w = word
                if w in glossary:
                    keywords[w] += 1

    if plotting:
        df = pd.DataFrame(list(keywords.items()), columns=['words', 'counts'])
        plt.rcParams["figure.figsize"] = (15,10)

        # 20 most common words in titles
        ax = df.sort_values('counts').tail(20).plot.barh(x='words', y='counts', color='blue')
        fig = ax.get_figure()
        plt.title(f"Top common words in titles of graphic conference - {year}")
        fig.savefig('word_frequencies.png')

    return keywords


# year = "2021"
# url = "https://kesen.realtimerendering.com/"
# current_year_keywords = word_frequencies(year, url)
# print(current_year_keywords)




