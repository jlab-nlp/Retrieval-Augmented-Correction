import requests
from bs4 import BeautifulSoup
import re
if __name__ == '__main__':
    link = "https://en.wikipedia.org/wiki/Focus..."
    link_result = requests.get(link, timeout=3)
    cleantext = BeautifulSoup(link_result.text, "lxml").text
    cleantext = re.sub(r'\n+', '\n', cleantext)
    print(cleantext)