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
        "sex":  "女",
        "age": 60,
        "education": "高中",
        "url": "https://cos.drbrain.net/profile/tj/2023/4/3/81fadc7e-6d8e-4693-84ac-d9291175541e.txt",
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