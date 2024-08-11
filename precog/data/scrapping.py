from bs4 import BeautifulSoup
import requests
import re
#link='https://huggingface.co/datasets/roneneldan/TinyStories?row='
link='https://huggingface.co/datasets/philschmid/easyrag-mini-wikipedia?row='
'''for row in range(500):
    url=link+str(row)
    html_text=requests.get(url).text
    soup=BeautifulSoup(html_text,'lxml')
    story=soup.find_all('div' class dir='auto')'''

token_count=0
for i in range(10):
    url=link+str(i)
    html_text=requests.get(url).text
    soup=BeautifulSoup(html_text,'lxml')
    story_elements=soup.find_all('div', dir='auto', class_='')
    story_texts=[elem.get_text() for elem in story_elements]
    story_text=' '.join(story_texts)
    cleaned_text=re.sub(r'</div>\s*,\s*<div class="" dir="auto">', ' ', story_text)
    cleaned_text=re.sub(r'</div>\s*', ' ', cleaned_text)
    cleaned_text=re.sub(r'<div class="" dir="auto">\s*', ' ', cleaned_text)
    token_count+=len(cleaned_text.split(' '))
    with open('scrapped_data.txt','w') as file:
        file.write(cleaned_text+'\n')
print(token_count)