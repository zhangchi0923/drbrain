import requests
import json
import hashlib
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from config.settings import settings


def get_md5(d):
    d = dict(sorted(d.items()))
    s = ''
    for k, v in d.items():
        s += str(k) + str(v)
    s = settings.salt + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()


if __name__ == '__main__':
    data2post = {
        "id":  1,
        # "url": "https://cos.drbrain.net/profile/tj/2023/9/21/25efef47-caa6-4820-8617-4dac133c1231.txt", # symbol
        "url": "E:/zc/GitProjects/algorithm-platform/platform/head_move/local_data/sd/sd_data.txt", # symbol
    }
    crypt2post = get_md5(data2post)
    with requests.post(
        url='http://127.0.0.1:8101/rehab/sd/cervical',
        data=json.dumps(data2post),
        headers={'Authorization': ''.join(crypt2post)}
    ) as r:
        print(r.text)
