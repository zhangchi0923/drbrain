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


if __name__ == "__main__":
    data2post = {
        "urlHead":"https://cos.drbrain.net/profile/tj/2022/11/25/888d6ccc-b137-414c-9a8c-a56f751cfc5d.txt",
        "urlBall":"https://cos.drbrain.net/profile/tj/2022/11/25/16d2aed6-86ab-4ce8-92d6-096341abd520.txt",
        "outPath":"./",
        "mode":"stand"
    }
    crypt2post = get_md5(data2post)
    print(crypt2post)
    
    with requests.post(
        url="http://127.0.0.1:8101/balance", 
        data=json.dumps(data2post), 
        headers={"Authorization":"".join(crypt2post)}
    ) as r:
        print(r.text)
        print(r.status_code)
