import requests
import json
import hashlib

from settings import SALT
def get_md5(d):
    d = dict(sorted(d.items()))
    s=''
    for k, v in d.items():
        s += str(k) + str(v)
    s = SALT + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()


if __name__ == '__main__':
    data2post = {
        "sex":  "男",
        "age": 68,
        "education": "初中",
        "url": "https://cos.drbrain.net/profile/tj/2023/3/13/764bd08c-a798-4e15-9268-a3dd1e6f64ba.txt",
        # "url": "https://cos.drbrain.net/profile/tj/2023/4/9/3d560422-bf7b-483d-9283-87ad0115d992.txt",
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