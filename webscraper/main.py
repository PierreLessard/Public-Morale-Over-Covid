from nytimes_scraper.nyt_api import NytApi
from nytimes_scraper.comments import fetch_comments
import requests
import csv
from datetime import datetime

api = NytApi('1OPTR7sah2S2i34yl1vdNoci9K1OLnns')

id_and_url_list = []
for i in range(200):
    url = f'https://api.nytimes.com/svc/search/v2/articlesearch.json?q=covid&api-key=1OPTR7sah2S2i34yl1vdNoci9K1OLnns&page=0'
    r = requests.get(url)
    article = r.text.split('web_url\":\"')
    for x in range(1, 11):
        end = article[x].find(".html") + len(".html")
        article_link = article[x][0:end]
        start = article[x].find("_id\":\"") + len("_id\":\"")
        end = article[x].find("\",\"word_count")
        article_id = article[x][start:end]
        id_and_url_list.append((article_id, article_link))

comments = fetch_comments(api, article_ids_and_urls=id_and_url_list)

f = open('comments.csv', 'w', encoding="utf-8", newline='')
writer = csv.writer(f)

for comment in comments:
    writer.writerow(
        [datetime.utcfromtimestamp(int(comment['createDate'])).strftime('%Y-%m-%d %H:%M:%S'), comment['commentBody']])
