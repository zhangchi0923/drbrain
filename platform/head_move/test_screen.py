import requests
import json
import hashlib
import os

from settings import SALT


def get_md5(d):
    d = dict(sorted(d.items()))
    s = ''
    for k, v in d.items():
        s += str(k) + str(v)
    s = SALT + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()


if __name__ == '__main__':
    data2post = {
        "sex":  "女",
        "age": 52,
        "education": "专科",
        # "url": "https://cos.drbrain.net/profile/tj/2023/5/16/3c6b49df-3d93-40ec-8bce-70390174becf.txt",
        "url": "https://cos.drbrain.net/profile/tj/2023/5/18/e80eea4d-0ef6-4675-86ff-e73593d25121.txt",
        # "backupResources": "/usr/local/project/algorithm/backup/utiles/design/",
        "questionVersion": 'A',
        "saveResourcesPath": "./eyescreen_log/"
    }
    crypt2post = get_md5(data2post)
    with requests.post(
        url='http://127.0.0.1:8101/eye/screen',
        data=json.dumps(data2post),
        headers={'Authorization': ''.join(crypt2post)}
    ) as r:
        print(r.text)
        print(r.status_code)
