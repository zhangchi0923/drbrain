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
        "url":  "https://cos.drbrain.net/profile/tj/2023/3/29/0626dfee-6bdd-4b48-af3e-d6545094d1d6.txt",
        "savePath": './firefly_log/',
    }
    crypt2post = get_md5(data2post)
    with requests.post(
        url='http://127.0.0.1:8101/eye/train/firefly', 
        data=json.dumps(data2post), 
        headers={'Authorization':''.join(crypt2post)}
    ) as r:
        print(r.text)
        print(r.status_code)