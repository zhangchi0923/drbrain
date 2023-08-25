import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, datetime
from qcloud_cos import CosConfig, CosS3Client
from settings import SECRET_ID, SECRET_KEY, REGION, PREFIX, BUCKET_NAME, URL_PREFIX

class Drawer(object):
    def __init__(self, id, type):
        self.id = id
        self.type = type
    
    def text2DF(self, text):
        lines = text.split('\n')
        head = lines[0].split(',')
        arr = [x.split(',') for x in lines[1:] if not not x]
        df = pd.DataFrame(arr, columns=head)
        df['timestamp'] = df['timestamp'].astype('uint64', errors='ignore')
        df['timestamp'] = df['timestamp'] - df.loc[0, 'timestamp']
        df['team'] = df['team'].astype(int, errors='ignore')
        df['id'] = df['id'].astype(int, errors='ignore')
        df['x'] = df['x'].astype(float, errors='ignore')
        df['y'] = df['y'].astype(float, errors='ignore')

        df.dropna(how='any', axis=0, inplace=True)
        self.team = df['team'].iloc[-1]
        self.img_num = df['id'].iloc[-1]
        return df
    
    def draw_and_save_cos(self): ...
    
    '''
    返回一张图片的字节流
    '''
    def draw(self,) -> io.BytesIO: ...
    
    '''
    将一张图片的字节流上传cos
    '''
    def save2cos(self, bio, client, key):
        response = client.put_object(
            Bucket=BUCKET_NAME,
            Body=bio.read(),
            Key=key,
        )
        if response['ETag']:
            return True
        else:
            return False
        
    # def deSpike(arr, threshold=1):
    #     x = arr[:, 1:3]
    #     dx = np.diff(x, axis=0)
    #     idx0 = np.abs(dx) < threshold
    #     idx1 = idx0[:, 1] & idx0[:, 0]
    #     idx2 = np.array([True]+list(idx1))
    #     newArr = arr[idx2, :]
    #     return newArr

class SymbolSearchDrawer(Drawer):
    def __init__(self, id, type):
        super().__init__(id, type)
    
    def draw(self, q_id, x, y) -> io.BytesIO:
        img = plt.imread('./pcat-design/symbol/{}.png'.format(q_id))
        plt.imshow(img, extent=[-3.84, 3.84, -2.16, 2.16])
        plt.plot(x, y, 'r-', linewidth=0.5)
        plt.axis('off')
        bio = io.BytesIO()
        plt.savefig(bio, format='jpg', dpi=200, bbox_inches='tight')
        bio.seek(0)
        return bio
    
    def draw_and_save_cos(self, txt) -> list:
        cos_urls = []

        config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY, Token=None, Scheme="https")
        client = CosS3Client(config)
        now = datetime.datetime.now()
        year, month, day = str(now.year), str(now.month), str(now.day)
        base_key = '/'.join([PREFIX, year, month, day, str(self.id), self.type])
        df = self.text2DF(txt)
        for i in range(1, self.img_num + 1):
            q_id = i
            x = df.loc[df['id'] == i, 'x']
            y = df.loc[df['id'] == i, 'y']
            bio = self.draw(q_id, x, y)
            key = base_key + '/{}.jpg'.format(i)
            self.save2cos(bio, client, key)
            cos_urls.append(URL_PREFIX + key)
        return cos_urls
    
class VocabularyDrawer(Drawer):
    def __init__(self, id, type):
        super().__init__(id, type)
    
    def draw(self, q_id, x, y) -> io.BytesIO:
        img = plt.imread('./pcat-design/vocab/{}/{}.png'.format(self.team, q_id))
        plt.imshow(img, extent=[-3.84, 3.84, -2.16, 2.16])
        plt.plot(x, y, 'r-', linewidth=0.5)
        plt.axis('off')
        bio = io.BytesIO()
        plt.savefig(bio, format='jpg', dpi=200, bbox_inches='tight')
        bio.seek(0)
        return bio
    
    def draw_and_save_cos(self, txt) -> list:
        cos_urls = []

        config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY, Token=None, Scheme="https")
        client = CosS3Client(config)
        now = datetime.datetime.now()
        year, month, day = str(now.year), str(now.month), str(now.day)
        base_key = '/'.join([PREFIX, year, month, day, str(self.id), self.type])
        df = self.text2DF(txt)
        for i in range(1, self.img_num + 1):
            q_id = i
            x = df.loc[df['id'] == i, 'x']
            y = df.loc[df['id'] == i, 'y']
            bio = self.draw(q_id, x, y)
            key = base_key + '/{}.jpg'.format(i)
            self.save2cos(bio, client, key)
            cos_urls.append(URL_PREFIX + key)
        return cos_urls