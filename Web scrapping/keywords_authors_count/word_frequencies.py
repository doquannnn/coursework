from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.corpus import stopwords
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import requests

url = "https://kesen.realtimerendering.com/"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}
page = requests.get(url, headers=headers, timeout=5, allow_redirects=True).text
doc = BeautifulSoup(page, 'html.parser')


vocab = defaultdict(lambda: 0)
stop_words = set(stopwords.words('english'))


content = doc.find(class_="content-item")
a_tags = content.find_all("a")

current_year = []
for a_tag in a_tags:
    if re.search(re.compile("2021"), a_tag.text):
        current_year.append(a_tag['href'])

current_year.pop(2)  # the third one is a blog

conferences = [os.path.join(url, conference) for conference in current_year]

for conference in conferences:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}
    page = requests.get(conference, headers=headers, timeout=5, allow_redirects=True).text
    doc = BeautifulSoup(page, 'html.parser')

    dt_tags = doc.find_all("dt")
    for title in dt_tags:
        words = title.text.split(" ")
        for word in words:
            if word == "-":
                continue
            w = word.lower()

            if w.endswith(":"):
                w = w[:-1]
            while w.endswith("\n"):
                w = w[:-1]
            if re.search("[\n]+", w):  # corner case
                w = re.sub("[\n]+", "", w)[:-4]
            if not w or w in stop_words:
                continue
            if w.endswith("(") or w.endswith(")"):
                w = w[:-1]

            vocab[w] += 1

# print(len(vocab))  # 1176
# print(vocab)  # pcâ\x80\x90mri: PCâ€ - due to format

df = pd.DataFrame(list(vocab.items()), columns=['words', 'counts'])

plt.rcParams["figure.figsize"] = (15,10)

# 20 most common words in titles
ax = df.sort_values('counts').tail(20).plot.barh(x='words', y='counts', color='blue')
fig = ax.get_figure()
plt.title("Most common words in titles")
fig.savefig('word_frequencies.png')










