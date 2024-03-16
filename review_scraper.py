import re
import os
import sys

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup


def get_continuation_text(continuation_url):
    response = requests.get(continuation_url)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')

    read_elements = soup.find_all('p', class_='read')
    continuation_text = [element.get_text() for element in read_elements]

    return continuation_text


def get_total_pages(pagination_links):
    page_numbers = [int(re.search(r'\d+', link.get_text()).group()) for link in pagination_links if re.search(r'\d+', link.get_text())]
    total_pages = max(page_numbers, default=1)

    return total_pages


def get_pagination_urls(base_url, total_pages):
    pagination_urls = [base_url]
    for page in range(2, total_pages + 1):
        pagination_urls.append(f"{base_url}?page={page}")

    return pagination_urls


def scrape_and_save(url, file):
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')

    read_more_links = soup.find_all('a', class_='cmn-viewmore')

    for link in read_more_links:
        continuation_url = link['href']
        continuation_text = get_continuation_text(continuation_url)
        for text in continuation_text:
            file.write(text)


def main(url, directory_name):
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, "html.parser")

    pagination_links = soup.find_all('ul', class_='number')[0].find_all('a')
    total_pages = get_total_pages(pagination_links)

    pagination_urls = get_pagination_urls(url, total_pages)

    output_directory = os.path.join(os.path.dirname(__file__), 'data/{}'.format(directory_name))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    with open(os.path.join(output_directory, 'review.txt'), 'w', encoding='utf-8') as file:
        for url in tqdm(pagination_urls):
            scrape_and_save(url, file)


if __name__ == "__main__":
    url = sys.argv[1]
    directory_name = sys.argv[2]
    main(url, directory_name)
