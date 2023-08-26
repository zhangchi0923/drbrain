import sys
import pandas as pd
import matplotlib.pyplot as plt
import io, datetime
from qcloud_cos import CosConfig, CosS3Client
import requests
from settings import SECRET_ID, SECRET_KEY, REGION, PREFIX, BUCKET_NAME



class Drawer(object):
    def __init__(self, id, type, txt):
        self.id = id
        self.type = type
        self.txt = txt

    def text2DF(self):
        lines = self.txt.strip().split('\n')
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
        self.df = df
        return
    
    def draw(self): ...

    def async_draw(self):
        config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY, Token=None, Scheme="https")
        client = CosS3Client(config)
        now = datetime.datetime.now()
        year, month, day = str(now.year), str(now.month), str(now.day)
        base_key = '/'.join([PREFIX, year, month, day, str(self.id), self.type])
        for i in range(1, self.img_num + 1):
            q_id = i
            x = self.df.loc[self.df['id'] == i, 'x']
            y = self.df.loc[self.df['id'] == i, 'y']
            bio = self.draw(q_id, x, y)
            key = base_key + '/{}.jpg'.format(i)
            self.save2cos(bio, client, key)
        return
    
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

class SymbolSearchDrawer(Drawer):
    def __init__(self, id, type, txt):
        super().__init__(id, type, txt)

    def draw(self, q_id, x, y) -> io.BytesIO:
        img = plt.imread('./pcat-design/symbol/{}.png'.format(q_id))
        plt.imshow(img, extent=[-3.84, 3.84, -2.16, 2.16])
        plt.plot(x, y, 'r-', linewidth=0.5)
        plt.axis('off')
        bio = io.BytesIO()
        plt.savefig(bio, format='jpg', dpi=200, bbox_inches='tight')
        bio.seek(0)
        return bio
    
    
    
class VocabularyDrawer(Drawer):
    def __init__(self, id, type, txt):
        super().__init__(id, type, txt)

    def draw(self, q_id, x, y) -> io.BytesIO:
        img = plt.imread('./pcat-design/vocab/{}/{}.png'.format(self.team, q_id))
        plt.imshow(img, extent=[-3.84, 3.84, -2.16, 2.16])
        plt.plot(x, y, 'r-', linewidth=0.5)
        plt.axis('off')
        bio = io.BytesIO()
        plt.savefig(bio, format='jpg', dpi=200, bbox_inches='tight')
        bio.seek(0)
        return bio

if __name__ == '__main__':
    id = sys.argv[1]
    type = sys.argv[2]
    url = sys.argv[3]
    with requests.get(url) as req:
        txt = req.text

    if type == "SYMBOL_SEARCH":
        drawer = SymbolSearchDrawer(id, type, txt)
    elif type == "VOCABULARY_TEST":
        drawer = VocabularyDrawer(id, type, txt)
    
    drawer.text2DF()
    drawer.async_draw()
