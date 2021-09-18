from word_frequency import word_frequencies
import json
import matplotlib.pyplot as plt
import numpy as np

# previous_year, current_year = "2020", "2021"
# url = "https://kesen.realtimerendering.com/"
# current_year_keywords = word_frequencies(current_year, url)
# previous_year_keywords = word_frequencies(previous_year, url)

# # Save keywords for not re-running multiple times
# with open('current_year.txt', 'w') as outfile:
#     json.dump(current_year_keywords, outfile)

# with open('previous_year.txt', 'w') as outfile:
#     json.dump(previous_year_keywords, outfile)

with open('current_year.txt', 'r') as json_file:
    current_year_kws = json.load(json_file)

with open('previous_year.txt', 'r') as json_file:
    previous_year_kws = json.load(json_file)

# print(len(current_year_kws))  # 35
# print(len(previous_year_kws))  # 33

combined = []

for kw in current_year_kws:
    if kw in previous_year_kws:
        combined.append((kw, current_year_kws[kw], previous_year_kws[kw]))

# keywords increasing most between last year and current year

n = len(combined)
x1 = [combined[i][0] for i in range(n)]
y1 = [combined[i][1] for i in range(n)]
y2 = [combined[i][2] for i in range(n)]

c1, c2 = max(combined, key=lambda x: x[1])[1:]
lim = max(c1, c2) + 5

plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = (20, 17)

plt.scatter(x1, y1, marker="*", s=40, c='blue', label="2021")
plt.scatter(x1, y2, marker="o", s=40, c='green', label="2020")
for i in range(len(x1)):
    if y1[i] - y2[i] == 0:
        continue
    elif y1[i] - y2[i] > 0:  # increasing
        plt.arrow(i, y2[i], 0, y1[i] - y2[i] - 0.6, head_width=0.2, width=0.02,
        color='red')
    else:  # decreasing
        plt.arrow(i, y2[i], 0, y1[i] - y2[i] + 0.6, head_width=0.2, width=0.02,
        color='yellow')

plt.xticks(rotation=50, size=15)
plt.yticks(np.arange(0, lim, 2))
plt.xlabel('keywords', fontsize=22)
plt.ylabel('counts', fontsize=22)
plt.title("Most likely trending keywords", fontsize=30)


plt.legend()
plt.savefig("Keywords trending.png")



