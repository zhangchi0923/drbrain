import requests
import json
import hashlib

from settings import SALT
def get_md5(d):
    d = dict(sorted(d.items()))
    s=''
    for k, v in d.items():
        s += k+v
    s = SALT + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()


if __name__ == '__main__':
    data2post = {
        "sex":  "男",
        "age": 10,
        "education": "本科",
        "url": "https://cos.drbrain.net/profile/tj/2023/2/18/568c2380-30e0-4fcf-b165-a347e007606b.txt",
        "backupResources": "/usr/local/project/algorithm/backup/utiles/design/",
        "saveResourcesPath": "/usr/local/project/eye_image/y0001"
    }
    crypt2post = get_md5(data2post)
    with requests.post(
        url='http://127.0.0.1:8101/eye/screen', 
        data=json.dumps(data2post), 
        headers={'Authorization':''.join(crypt2post)}
    ) as r:
        print(r.text)
        print(r.status_code)