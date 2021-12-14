""" NYT Comments Scraper
Scrapes the NYT API to get comments from the articles in the API website

This file made by Zhi Yu
"""
from nytimes_scraper.nyt_api import NytApi  # pip install nytimtes_scraper
from nytimes_scraper.comments import fetch_comments
import requests  # pip install requests
import csv
from datetime import datetime


def write_comments_to_csv(year: int, month: int, startdate: int, enddate: int) -> None:
    """
    Writes all comments from 50 articles in the range of year startdate to year enddate to a .csv
    file that is named comments_year_startdate_enddate.csv

    Preconditions:
    - 20 <= year <= 21
    - 1 <= month <= 12
    - 1 <= startdate <= 30
    - 1 <= enddate <= 30
    """

    # API key from NYT API website
    api = NytApi('1OPTR7sah2S2i34yl1vdNoci9K1OLnns')

    id_and_url_list = []

    # Each api search returns 10 articles, which means the loop returns 50 articles
    for i in range(0, 5):

        url = f'https://api.nytimes.com/svc/search/v2/' \
               f'articlesearch.json?q=covid&api-key=1OPTR7sah2S2i34yl1vdNoci9K1OLnns&page={i}' \
               f'&begin_date=20{year}{month}{startdate}&end_date=20{year}{month}{enddate}'

        # Turns the api search page into a string
        r = requests.get(url)

        # This splits the string into 11 different sections, with an article url and uri in 10 of the 11 sections
        article = r.text.split('web_url\":\"')

        # Loops 10 times to get all article 10 url and uri
        for x in range(1, 11):
            # Gets the article url
            end = article[x].find(".html") + len(".html")
            article_link = article[x][0:end]

            # Gets the article uri
            start = article[x].find("_id\":\"") + len("_id\":\"")
            end = article[x].find("\",\"word_count")
            article_id = article[x][start:end]

            # Adds the article uri and url to list
            id_and_url_list.append((article_id, article_link))

    # fetches all the comments in the 50 articles
    comments = fetch_comments(api, article_ids_and_urls=id_and_url_list)

    # writes to a csv file
    f = open(f'comments_{year}_{month}_{startdate}_{enddate}.csv', 'w', encoding="utf-8", newline='')
    writer = csv.writer(f)

    # write each comment to the csv file
    for comment in comments:

        # only 2 columns: date and comment body
        writer.writerow(
            [datetime.utcfromtimestamp(int(comment['createDate'])).strftime('%Y-%m-%d %H:%M:%S'),
             comment['commentBody']])
