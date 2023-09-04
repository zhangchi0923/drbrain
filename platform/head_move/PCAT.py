import datetime
import pandas as pd
import requests
from settings import PREFIX, URL_PREFIX

class Pcat(object):
    def __init__(self, id, type, url):
        self.id = id
        self.type = type
        self.url = url
    
    def text2DF(self, txt):
        lines = txt.strip().split('\n')
        head = lines[0].split(',')
        arr = [x.split(',') for x in lines[1:] if not not x]
        df = pd.DataFrame(arr, columns=head)
        df['timestamp'] = df['timestamp'].astype('uint64', errors='raise')
        df['timestamp'] = df['timestamp'] - df.loc[0, 'timestamp']
        df['team'] = df['team'].astype(int, errors='raise')
        df['id'] = df['id'].astype(int, errors='raise')
        df['x'] = df['x'].astype(float, errors='raise')
        df['y'] = df['y'].astype(float, errors='raise')

        df.dropna(how='any', axis=0, inplace=True)
        return df

    def get_url_and_img_num(self):

        try:
            with requests.get(self.url) as url_data:
                assert url_data.status_code == 200
                txt = url_data.text
            df = self.text2DF(txt)
            # TODO: 根据眼动数据格式确定 img_num
            img_num = df.loc[df['team'] >=0, 'id'].unique().shape[0]
            return img_num

        except Exception as e:
            if isinstance(e, AssertionError) or isinstance(e, requests.ConnectionError):
                raise ConnectionError("Error during fetching eye tracking url data.")
            elif isinstance(e, KeyError):
                raise KeyError("Mismatched eye tracking data columns!")
            else:
                raise e
    
    def make_cos_urls(self):
        img_num = self.get_url_and_img_num()
        cos_urls = []
        now = datetime.datetime.now()
        year, month, day = str(now.year), str(now.month), str(now.day)
        base_key = '/'.join([PREFIX, year, month, day, str(self.id), self.type])
        for i in range(1, img_num + 1):
            key = base_key + '/{}.jpg'.format(i)
            cos_urls.append(URL_PREFIX + key)
        return cos_urls

