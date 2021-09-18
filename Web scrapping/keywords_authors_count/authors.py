from bs4 import BeautifulSoup
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import requests

authors = defaultdict(lambda: 0)

def extract_names(dd_tag: list):
    for tag in dd_tag:
        temp = tag

        if not temp or temp.isspace():  # empty string or only whitespace
            continue
        if temp.endswith("*"):  # name ends with *
            temp = temp[:-1]
        if len(re.findall("\((.+)\)", temp)) != 0:  # remove university, retain author
            temp = re.sub("\((.+)\)", "", temp)
        if re.search("university", temp, re.IGNORECASE): # only university
            continue

        temp = temp.split(',') # if multiple authors
        if len(temp) >= 2:
            for name in temp:
                if not name or name.isspace():
                    continue
                authors[name.strip()] += 1
        else:
            if temp[0]:
                authors[temp[0].strip()] += 1

# a = ['(MIT CSAIL and Google Research)', 'Jiamin Xu, Xiuchao Wu, Zihan Zhu,', 'Weiwei Xu',
# '(Zhejiang University),', 'Yin Yang', '(Clemson University),', 'Hujun Bao',
# '(Zhejiang University),','Qixing Huang', '(The University of Texas at Austin)', '']
# extract_names(a)
# print(authors)
# exit()

url = "https://kesen.realtimerendering.com/"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"}
page = requests.get(url, headers=headers, timeout=5, allow_redirects=True).text
doc = BeautifulSoup(page, 'html.parser')

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

    dd_tags = doc.find_all("dd")
    for dd_tag in dd_tags:
        # print(dd_tag.text.split("\n"))
        extract_names(dd_tag.text.split("\n"))

authors.pop("()")
# print(len(authors))  # 1143


df = pd.DataFrame(list(authors.items()), columns=['authors', 'citations'])

plt.rcParams["figure.figsize"] = (15,10)

# 20 authors with most citations
ax = df.sort_values('citations').tail(20).plot.barh(x='authors', y='citations', color='blue')
fig = ax.get_figure()
plt.title("Authors get most citations")
fig.savefig('author_citations.png')
