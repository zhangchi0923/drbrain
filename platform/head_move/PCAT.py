import datetime
from settings import PREFIX, URL_PREFIX

class Pcat(object):
    def __init__(self, id, type, team, img_num):
        self.id = id
        self.type = type
        self.team = int(team)
        self.img_num = int(img_num)
    
    def make_cos_urls(self):
        cos_urls = []
        now = datetime.datetime.now()
        year, month, day = str(now.year), str(now.month), str(now.day)
        base_key = '/'.join([PREFIX, year, month, day, str(self.id), self.type])
        for i in range(1, self.img_num + 1):
            key = base_key + '/{}.jpg'.format(i)
            cos_urls.append(URL_PREFIX + key)
        return cos_urls

