import re
import requests
import lxml
from bs4 import BeautifulSoup
import urllib
import pandas as pd


def convert_birds_dataset_to_bi(data_path):
    data_csv = pd.read_csv(data_path)
    data_csv.drop(columns=[data_csv.columns[3], data_csv.columns[4]], inplace=True)
    data_csv['label'] = 1
    data_csv.to_csv('../Save/bi_bird_data.csv', index=False)


if __name__ == '__main__':
    # u = 'https://pixabay.com/zh/photos/?&pagi='
    # for i in range(2, 1001):
    #     url = u + str(i)
    #     page = requests.get(url).text
    #     pagesoup = BeautifulSoup(page, 'lxml')
    #
    #     img_id = 0
    #     for link in pagesoup.find_all(name='img'):
    #         img_link = None
    #         if 'data-lazy' in str(link):
    #             img_link = link['data-lazy']
    #         elif 'src' in str(link) and 'static' not in str(link):
    #             img_link = link['src']
    #
    #         if img_link is not None:
    #             try:
    #                 urllib.request.urlretrieve(img_link, '../data/img_' + str(i) + '_' + str(img_id) + '.jpg')
    #                 print(str(i)+'_'+str(img_id))
    #                 img_id += 1
    #             except:
    #                 print('img 404')

    convert_birds_dataset_to_bi('../Save/data.csv')
